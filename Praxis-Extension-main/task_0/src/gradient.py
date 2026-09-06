"""The policy-gradient extractor: one code path, two input modalities.

    g_hat = (1/|B|) sum_i (1/G) sum_j A_hat_ij grad_theta log pi_theta(a_ij | s_i)

Per item: build the prompt (image OR text), sample G answers from the current policy,
score each 0/1 against the known answer, centre within the group to get advantages,
backprop the advantage-weighted log-probs, accumulate. No optimizer step.

The four places this silently goes wrong, and what is done about each
---------------------------------------------------------------------
1. Sampling from the wrong distribution. The score function grad log pi is differentiated
   against the *untruncated* softmax. Qwen2.5-VL-Instruct ships a generation_config with
   top_k=20, top_p=0.8, temperature=0.7 baked in; leaving those on means sampling from a
   truncated distribution while differentiating the full one, which makes g_hat biased,
   not merely noisy. `_generation_config` builds a config from scratch that forces
   top_k=0, top_p=1.0, repetition_penalty=1.0, and `sanity_checks.check_sampling_matches_policy`
   empirically verifies the realised next-token histogram against softmax(logits).

2. Prompt tokens leaking into the gradient. Only generated tokens may contribute. The
   logits are sliced at `prompt_len - 1` (the position that *predicts* the first response
   token) through `-1`, so the response targets line up one-for-one and no prompt target
   is ever scored.

3. Padding tokens counted as generated. `generate` right-pads finished sequences. We
   locate the first EOS per row and *trim* the sequence there (the EOS itself is part of
   the action and keeps its log-prob), so nothing downstream has to remember to mask.
   `response_lengths` implements this and is unit-tested against hand-built cases.

4. Advantage applied at the wrong granularity. The advantage is a scalar per sample, so
   `aggregate_coeffs` reduces every GRPO convention to (per-sample coefficient, one final
   scale). That single function is used by the cached path and the direct path alike,
   which is what makes them provably the same estimator.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Sequence

from .rewards import group_advantages, group_is_degenerate, score_completion

QUESTION = "Given the situation, which of the following actions is the most appropriate?"


# ------------------------------------------------------------------ data records


@dataclass
class CalibItem:
    """One paired calibration example: the same situation via image and via text."""

    index: int
    image_path: str
    situation: str                     # the text channel (VIVA situation_description)
    action_list: list[str]
    answer: str
    category: str = ""
    _image: Any = field(default=None, repr=False, compare=False)

    def image(self):
        from PIL import Image

        if self._image is None:
            self._image = Image.open(self.image_path).convert("RGB")
        return self._image

    def release(self) -> None:
        self._image = None


@dataclass
class GroupSamples:
    """The K generations drawn once for one (item, modality), plus their scores."""

    seqs: list          # list of 1-D LongTensor, each already trimmed at EOS
    prompt_len: int
    lengths: list[int]  # response token count per sample (>= 1)
    rewards: list[float]
    parsed_ok: list[bool]
    texts: list[str]
    # Vision tensors for this prompt, carried so the backward passes can reuse them
    # instead of re-running the image processor once per sample.
    vis: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.seqs)


# ------------------------------------------------------------------ GRPO aggregation


def aggregate_coeffs(advantages: Sequence[float], lengths: Sequence[int],
                     loss_agg: str, n_items: int, G: int) -> tuple[list[float], float]:
    """Reduce a GRPO convention to per-sample coefficients and one final scale.

    All three conventions are the same vector reweighted, because the advantage and the
    token counts are constants with respect to theta:

      seq_sum          g = (1/n) sum_i (1/G) sum_j A_ij v_ij            v_ij = grad sum_t log pi
      seq_mean         g = (1/n) sum_i (1/G) sum_j (A_ij / L_ij) v_ij
      batch_token_mean g = (sum_ij A_ij v_ij) / (sum_ij L_ij)           (verl's default)

    Returned as (coeffs, scale) with g = scale * sum_ij coeffs_ij * v_ij, so the cached
    path and the direct path cannot drift apart.
    """
    A = [float(a) for a in advantages]
    L = [max(int(x), 1) for x in lengths]
    if loss_agg == "seq_sum":
        return A, 1.0 / max(n_items * G, 1)
    if loss_agg == "seq_mean":
        return [a / l for a, l in zip(A, L)], 1.0 / max(n_items * G, 1)
    if loss_agg == "batch_token_mean":
        # scale is applied by the caller once the whole batch's token total is known
        return A, float("nan")
    raise ValueError(f"unknown loss_agg {loss_agg!r}")


# ------------------------------------------------------------------ gradient sinks


class SketchSink:
    """Accumulates sum_ij coef_ij * S(v_ij) in the m-dimensional sketch space."""

    def __init__(self, sketch):
        import torch

        self.sketch = sketch
        self.buf = torch.zeros(sketch.m, dtype=torch.float32, device=sketch.device)

    def reset(self) -> None:
        self.buf.zero_()

    def add(self, params, coef: float) -> None:
        self.sketch.project_grads(params, out=self.buf, scale=float(coef), accumulate=True)

    def scale(self, s: float) -> None:
        self.buf.mul_(float(s))

    def vector(self):
        return self.buf


class FullSink:
    """Accumulates sum_ij coef_ij * v_ij over the full d-dimensional backbone, fp32.

    d = 3.09e9, so one buffer is 12.4 GB. Two live at once (one per modality). On a 40 GB
    card that will not co-exist with the model plus its gradients, so `device="cpu"` (with
    high-RAM) is the default resolution and the per-item copy cost is accepted.
    """

    def __init__(self, numels: Sequence[int], device: str = "cpu"):
        import torch

        self.torch = torch
        self.numels = list(numels)
        self.d = sum(self.numels)
        self.device = torch.device(device)
        self.buf = torch.zeros(self.d, dtype=torch.float32, device=self.device)
        self.offsets = []
        acc = 0
        for n in self.numels:
            self.offsets.append(acc)
            acc += n

    def reset(self) -> None:
        self.buf.zero_()

    def add(self, params, coef: float) -> None:
        coef = float(coef)
        if coef == 0.0:
            return
        for p, off, n in zip(params, self.offsets, self.numels):
            g = p.grad
            if g is None:
                continue
            self.buf[off:off + n].add_(
                g.reshape(-1).to(device=self.device, dtype=self.torch.float32), alpha=coef
            )

    def scale(self, s: float) -> None:
        self.buf.mul_(float(s))

    def vector(self):
        return self.buf


def chunked_dot(a, b, chunk: int = 1 << 24) -> float:
    """Inner product accumulated in float64 over float32 chunks.

    A naive torch.dot over 3.09e9 float32 terms accumulates rounding badly enough to
    matter when the true cosine is small; chunking and promoting the partial sums keeps
    the error far below the sampling noise we are trying to measure.
    """
    import torch

    if a.numel() != b.numel():
        raise ValueError(f"dot dimension mismatch: {a.numel()} vs {b.numel()}")
    total = 0.0
    for lo in range(0, a.numel(), chunk):
        hi = min(lo + chunk, a.numel())
        total += float(torch.dot(a[lo:hi].float(), b[lo:hi].float()).double())
    return total


# ------------------------------------------------------------------ the extractor


class PolicyGradientExtractor:
    def __init__(self, model, processor, backbone, cfg, *, device: str = "cuda"):
        import torch

        self.torch = torch
        self.model = model
        self.proc = processor
        self.tok = getattr(processor, "tokenizer", processor)
        self.backbone = backbone
        self.params = backbone.params
        self.cfg = cfg
        self.device = torch.device(device)

        self.temperature = float(cfg.get_path("policy.temperature", 1.0))
        self.max_new_tokens = int(cfg.get_path("policy.max_new_tokens", 256))
        self.system_prompt = cfg.get_path("policy.system_prompt") or None
        self.gen_batch = int(cfg.get_path("sampling.gen_batch", 16))

        self._ltk_kw = self._detect_logits_to_keep()
        self._eos_ids = self._collect_eos_ids()
        self._gen_cfg = self._generation_config()

    # -- model interface probing ----------------------------------------------
    def _detect_logits_to_keep(self) -> str | None:
        """Ask the model to compute the lm_head only over the tail we score.

        Renamed across transformers versions, and absent in older ones. Optional: without
        it we slice the full logits instead, at the cost of materialising
        (prompt+response) x 152k logits per forward.
        """
        try:
            params = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return None
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in params:
                return name
        return None

    def _collect_eos_ids(self) -> list[int]:
        ids: set[int] = set()
        for src in (getattr(self.model, "generation_config", None), self.model.config):
            v = getattr(src, "eos_token_id", None) if src is not None else None
            if isinstance(v, int):
                ids.add(v)
            elif isinstance(v, (list, tuple)):
                ids.update(int(x) for x in v)
        t = getattr(self.tok, "eos_token_id", None)
        if isinstance(t, int):
            ids.add(t)
        if not ids:
            raise RuntimeError("no eos_token_id found; response trimming would be unsafe")
        return sorted(ids)

    def _pad_id(self) -> int:
        for src in (getattr(self.model, "generation_config", None), self.tok, self.model.config):
            v = getattr(src, "pad_token_id", None) if src is not None else None
            if isinstance(v, int):
                return v
        return self._eos_ids[0]

    def _generation_config(self):
        """A generation config built from scratch so nothing is inherited.

        Every knob that would reshape the sampling distribution is pinned to its
        identity value. If any of these drift, g_hat stops being an estimate of the
        gradient of the objective the policy actually defines.
        """
        from transformers import GenerationConfig

        return GenerationConfig(
            do_sample=True,
            temperature=self.temperature,
            top_p=float(self.cfg.get_path("policy.top_p", 1.0)),
            top_k=int(self.cfg.get_path("policy.top_k", 0)),
            repetition_penalty=float(self.cfg.get_path("policy.repetition_penalty", 1.0)),
            min_p=None,
            typical_p=1.0,
            epsilon_cutoff=0.0,
            eta_cutoff=0.0,
            no_repeat_ngram_size=0,
            length_penalty=1.0,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._pad_id(),
            eos_token_id=self._eos_ids,
            # The KV cache must be ON for generation. load_model sets
            # model.config.use_cache=False for the checkpointed backward passes; without
            # re-enabling it here, sampling would run without a cache and cost O(T^2).
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )

    def make_gen_config(self, num_return_sequences: int, max_new_tokens: int | None = None):
        """A per-call copy of the pinned config.

        transformers deprecates passing generation kwargs *alongside* a generation_config,
        so anything that varies per call has to be set on a copy rather than handed to
        generate() separately.
        """
        import copy

        gc = copy.deepcopy(self._gen_cfg)
        gc.num_return_sequences = int(num_return_sequences)
        if max_new_tokens is not None:
            gc.max_new_tokens = int(max_new_tokens)
        return gc

    # -- prompt construction ---------------------------------------------------
    def _messages(self, item: CalibItem, modality: str) -> list[dict]:
        opts = "\n".join(item.action_list)
        body = f"## Question:\n{QUESTION}\n{opts}\n\nNow answer the question."
        if modality == "image":
            content = [
                {"type": "image"},
                {"type": "text", "text": "## Situation:\nShown in the given image.\n" + body},
            ]
        elif modality == "text":
            content = [
                {"type": "text", "text": "## Situation:\n" + item.situation + "\n" + body},
            ]
        else:
            raise ValueError(f"modality must be 'image' or 'text', got {modality!r}")
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        msgs.append({"role": "user", "content": content})
        return msgs

    def build_prompt(self, item: CalibItem, modality: str):
        """Returns (inputs, prompt_len, vis) with everything already on device."""
        torch = self.torch
        text = self.proc.apply_chat_template(
            self._messages(item, modality), tokenize=False, add_generation_prompt=True
        )
        if modality == "image":
            inputs = self.proc(text=[text], images=[item.image()], return_tensors="pt")
        else:
            try:
                inputs = self.proc(text=[text], return_tensors="pt")
            except Exception:
                inputs = self.tok([text], return_tensors="pt")

        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        vis = {k: inputs[k] for k in
               ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
               if k in inputs}
        if modality == "text" and vis:
            raise RuntimeError(f"text prompt produced vision inputs {sorted(vis)} — the two "
                               "modalities are not actually separated")
        return inputs, prompt_len, vis

    # -- sampling --------------------------------------------------------------
    def response_lengths(self, resp):
        """Token count per row up to and including the first EOS.

        Rows that ran to max_new_tokens without emitting EOS keep their full length. The
        EOS token itself is *kept*: emitting it is part of the action, so its log-prob
        belongs in log pi(a|s).
        """
        torch = self.torch
        eos = torch.tensor(self._eos_ids, device=resp.device, dtype=resp.dtype)
        is_eos = torch.isin(resp, eos)                       # (B, T)
        has_eos = is_eos.any(dim=-1)
        first = is_eos.to(torch.int64).argmax(dim=-1)        # 0 where no EOS; guarded below
        T = resp.shape[1]
        return torch.where(has_eos, first + 1, torch.full_like(first, T))

    def sample_group(self, item: CalibItem, modality: str, K: int, *, seed: int | None = None
                     ) -> GroupSamples:
        """Draw K completions from pi_theta and score them. Pure inference, no grad."""
        torch = self.torch
        inputs, prompt_len, vis = self.build_prompt(item, modality)

        if seed is not None:
            torch.manual_seed(int(seed))

        seqs: list = []
        lengths: list[int] = []
        texts: list[str] = []
        remaining = K
        while remaining > 0:
            b = min(remaining, self.gen_batch)
            with torch.no_grad():
                out = self.model.generate(**inputs, generation_config=self.make_gen_config(b))
            full = out.sequences                              # (b, prompt_len + T)
            if int(full.shape[1]) <= prompt_len:
                raise RuntimeError("generate returned no response tokens")
            # The prompt must be reproduced verbatim in every row; if it is not, the
            # prompt_len slice used for the log-prob would be measuring the wrong tokens.
            if not bool((full[:, :prompt_len] == inputs["input_ids"]).all()):
                raise RuntimeError(
                    "generated sequences do not begin with the prompt — generate() is "
                    "padding or reordering, and prompt_len can no longer be trusted"
                )
            resp = full[:, prompt_len:]
            lens = self.response_lengths(resp)
            for r in range(full.shape[0]):
                L = int(lens[r])
                seqs.append(full[r, : prompt_len + L].detach().clone())
                lengths.append(L)
                texts.append(self.tok.decode(resp[r, :L], skip_special_tokens=True))
            del out, full, resp
            remaining -= b

        scored = [score_completion(t, item.answer, item.action_list) for t in texts]
        rewards = [s[0] for s in scored]
        parsed = [s[1] for s in scored]
        return GroupSamples(seqs=seqs, prompt_len=prompt_len, lengths=lengths,
                            rewards=rewards, parsed_ok=parsed, texts=texts, vis=vis)

    # -- log-probability -------------------------------------------------------
    def sequence_logprob(self, seq_1d, prompt_len: int, vis: dict, *,
                         per_token: bool = False):
        """sum_t log pi_theta(a_t | s, a_<t) over response tokens only, with grad.

        `seq_1d` is already trimmed at EOS, so every position after `prompt_len` is a real
        generated token and no mask is required. The logits slice is the whole trick:
        position p-1 predicts token p, so logits[prompt_len-1 : -1] aligns exactly with
        targets seq[prompt_len:].
        """
        torch = self.torch
        import torch.nn.functional as F

        seq = seq_1d.unsqueeze(0) if seq_1d.dim() == 1 else seq_1d
        total = int(seq.shape[1])
        T = total - prompt_len
        if T <= 0:
            raise ValueError(f"no response tokens: total={total} prompt_len={prompt_len}")

        fwd: dict[str, Any] = {
            "input_ids": seq,
            "attention_mask": torch.ones_like(seq),
            "use_cache": False,
        }
        fwd.update(vis)
        if self._ltk_kw:
            fwd[self._ltk_kw] = T + 1

        out = self.model(**fwd)
        logits = out.logits
        if int(logits.shape[1]) == total:
            sel = logits[:, prompt_len - 1: total - 1, :]
        elif int(logits.shape[1]) == T + 1:
            sel = logits[:, :-1, :]
        else:
            raise RuntimeError(
                f"unexpected logits length {int(logits.shape[1])}; expected {total} or {T+1}"
            )
        if int(sel.shape[1]) != T:
            raise RuntimeError(f"logit/target misalignment: {int(sel.shape[1])} vs {T}")

        sel = sel.float()
        if self.temperature != 1.0:
            sel = sel / self.temperature
        tgt = seq[:, prompt_len:]

        if per_token:
            logp = torch.log_softmax(sel, dim=-1)
            tok = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            return tok.sum(), T, tok.squeeze(0)
        # cross_entropy is the memory-lean, numerically stable route to the same number.
        nll = F.cross_entropy(sel.reshape(-1, sel.shape[-1]), tgt.reshape(-1), reduction="sum")
        return -nll, T

    # -- one accumulated gradient ---------------------------------------------
    def backward_weighted(self, seq_1d, prompt_len: int, vis: dict, coef: float) -> int:
        """Backprop coef * log pi(a|s), letting p.grad accumulate. No sink, no zeroing.

        Returns the response length, or 0 if the sample was skipped. A zero advantage
        contributes nothing, so skipping it saves a whole backward pass — and with binary
        rewards, groups where every answer scores the same are common enough that this is
        a large saving, not a micro-optimisation.
        """
        if float(coef) == 0.0:
            return 0
        logp, T = self.sequence_logprob(seq_1d, prompt_len, vis)
        (float(coef) * logp).backward()
        return T

    def accumulate_sample(self, seq_1d, prompt_len: int, vis: dict, coef: float, sink) -> int:
        """Backprop one sample and add coef * its gradient into `sink`, in isolation.

        One sink write per sample. Used where that granularity is the point (the cache
        linearity check); `modality_gradient` batches by item instead, because a FullSink
        write moves 6.2 GB across the PCIe bus and doing that per sample rather than per
        item costs a factor of G in wall-clock.
        """
        if float(coef) == 0.0:
            return 0
        self.model.zero_grad(set_to_none=True)
        logp, T = self.sequence_logprob(seq_1d, prompt_len, vis)
        logp.backward()
        sink.add(self.params, coef)
        self.model.zero_grad(set_to_none=True)
        return T

    def per_sample_gradient(self, seq_1d, prompt_len: int, vis: dict, sketch, out=None):
        """S(v_ij) for one sample: the *unweighted* grad sum_t log pi, sketched.

        Unweighted on purpose — the advantage is applied later as a scalar, which is what
        lets one cached vector serve every (n, G, convention, resample) combination.
        """
        self.model.zero_grad(set_to_none=True)
        logp, T = self.sequence_logprob(seq_1d, prompt_len, vis)
        logp.backward()
        vec = sketch.project_grads(self.params, out=out)
        self.model.zero_grad(set_to_none=True)
        return vec, T

    # -- a full modality gradient, generated fresh -----------------------------
    def modality_gradient(self, items: Sequence[CalibItem], modality: str, G: int, *,
                          grpo: dict, sink, seed: int | None = None) -> dict:
        """g_hat_v or g_hat_t from fresh generations. Used by the exact anchor and the
        acid test; the sweep uses the cache instead, via alignment.combine_from_cache."""
        sink.reset()
        n = len(items)
        total_tokens = 0
        stats = {"degenerate_groups": 0, "reward_sum": 0.0, "reward_n": 0,
                 "parse_fail": 0, "backwards": 0}

        for i, item in enumerate(items):
            grp = self.sample_group(item, modality, G,
                                    seed=None if seed is None else seed * 100003 + i)
            adv = group_advantages(grp.rewards,
                                   normalize_by_std=grpo["normalize_by_std"],
                                   eps=grpo["eps"], std_floor=grpo["std_floor"])
            coeffs, _ = aggregate_coeffs(adv, grp.lengths, grpo["loss_agg"], n, G)

            stats["reward_sum"] += sum(grp.rewards)
            stats["reward_n"] += len(grp.rewards)
            stats["parse_fail"] += sum(1 for ok in grp.parsed_ok if not ok)
            if group_is_degenerate(grp.rewards):
                stats["degenerate_groups"] += 1

            # Accumulate this item's G samples into p.grad (autograd sums them for us,
            # and G terms of bf16 accumulation is well inside the noise), then make one
            # sink write. The token total counts every sampled token, including those in
            # skipped zero-advantage groups: that is verl's batch denominator.
            self.model.zero_grad(set_to_none=True)
            wrote = 0
            for seq, coef, L in zip(grp.seqs, coeffs, grp.lengths):
                if self.backward_weighted(seq, grp.prompt_len, grp.vis, coef):
                    wrote += 1
                total_tokens += L
            if wrote:
                sink.add(self.params, 1.0)
                stats["backwards"] += wrote
            self.model.zero_grad(set_to_none=True)
            item.release()

        if grpo["loss_agg"] == "batch_token_mean":
            sink.scale(1.0 / max(total_tokens, 1))
        else:
            sink.scale(1.0 / max(n * G, 1))

        stats["mean_reward"] = stats["reward_sum"] / max(stats["reward_n"], 1)
        stats["total_response_tokens"] = total_tokens
        return stats


__all__ = ["CalibItem", "GroupSamples", "PolicyGradientExtractor", "SketchSink", "FullSink",
           "aggregate_coeffs", "chunked_dot", "QUESTION"]
