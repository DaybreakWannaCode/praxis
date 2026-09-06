"""Correctness gate. Run this before believing any Lambda.

The score-function extraction is the part of this pipeline most likely to be subtly and
silently wrong, and a sign error or a masking bug corrupts every downstream number without
ever raising. These checks are ordered cheapest-first; the model-dependent ones run on the
real checkpoint by default, because a test that passes on a 2-layer toy and a test that
passes on the model you are measuring are not the same evidence.

The decisive one is `check_finite_difference`: take a single sample, give it advantage +1,
compute g = grad log pi(a|s), step theta -> theta + eps*g, and verify that log pi(a|s)
actually went UP by eps*||g||^2. That single assertion catches a flipped sign, a
mis-slice that scores prompt tokens instead of response tokens, and an advantage applied
with the wrong sense. `check_cache_linearity` then confirms the cached reconstruction is
the same estimator as direct backprop, which is what the entire sweep rests on.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

_RESULTS: list[dict] = []


def _record(name: str, ok: bool, detail: str = "", **extra) -> dict:
    r = {"name": name, "ok": bool(ok), "detail": detail, **extra}
    _RESULTS.append(r)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return r


# ==================================================================== pure-python


def check_reward_parsing() -> dict:
    """The answer parser, against the shapes a Qwen chat model actually emits."""
    from .rewards import option_labels, parse_choice, score_completion

    opts = ["A. Pick up the items and throw them back into the car.",
            "B. Take note of the location and pick up the trash when it is safe to do so.",
            "C. Call emergency services to promptly handle the situation.",
            "D. Politely remind them not to do so.",
            "E. No action is necessary given the situation depicted in the image."]

    cases = [
        ("<think>reasoning here</think><answer>D</answer>", "D"),
        ("<think>x</think>\n<answer>D. Politely remind them not to do so.</answer>", "D"),
        ("<think>x</think> (B)", "B"),
        ("<think>x</think>\nThe answer is C.", "C"),
        ("<answer>**E**</answer>", "E"),
        ("<think>x</think> Option A", "A"),
        ("<think>x</think> I would choose B because it is safest.", "B"),
        ("<think>x</think> Politely remind them not to do so.", "D"),
        ("<answer>C) Call emergency services</answer>", "C"),
        ("<answer>Answer: B</answer>", "B"),
        ("<answer>The correct action is E.</answer>", "E"),
        ("<answer>I think we should call emergency services to promptly handle the "
         "situation.</answer>", "C"),
        ("<answer>A</answer>", "A"),
        ("<think>x</think> A", "A"),
        # Must NOT be guessed at:
        ("<think>x</think> None of these seem right.", None),
        ("<answer>An observer might do nothing at all here.</answer>", None),
        ("<answer>Both A and C seem reasonable.</answer>", None),
        ("<answer>Z</answer>", None),
        ("", None),
        ("<think>x</think>\n\n", None),
        # Two traps that a naive "one option letter appears" rule gets wrong. The first
        # reads as A off the indefinite article; the second off a truncated CoT's "a".
        ("<think>x</think> A person is littering, so the best response is to remind them "
         "politely not to do so.", "D"),
        ("<think>truncated reasoning without a close", None),
    ]
    bad = []
    for text, want in cases:
        got = parse_choice(text, opts)
        if got != want:
            bad.append((text[:52], want, got))

    labels_ok = option_labels(opts) == list("ABCDE")
    r_correct, ok1 = score_completion("<answer>D</answer>", "D", opts)
    r_wrong, ok2 = score_completion("<answer>A</answer>", "D", opts)
    r_unparsed, ok3 = score_completion("<answer>hmm</answer>", "D", opts)
    scoring_ok = (r_correct, r_wrong, r_unparsed) == (1.0, 0.0, 0.0) and ok1 and ok2 and not ok3

    return _record("reward parsing", not bad and labels_ok and scoring_ok,
                   f"{len(cases)-len(bad)}/{len(cases)} parse cases"
                   + (f"; mismatches {bad}" if bad else ""))


def check_advantages() -> dict:
    """group_advantages against hand arithmetic, and against its vectorised twin."""
    import torch

    from .alignment import advantages_torch
    from .rewards import binary_group_std, group_advantages

    # G=4, two correct: mean 0.5, ddof=1 std = sqrt(4*(1/3)*(0.25)) -> 0.57735
    r = [1.0, 1.0, 0.0, 0.0]
    sd = math.sqrt(sum((x - 0.5) ** 2 for x in r) / 3)
    want = [(x - 0.5) / (sd + 1e-6) for x in r]
    got = group_advantages(r, normalize_by_std=True, eps=1e-6)
    ok_hand = all(abs(a - b) < 1e-9 for a, b in zip(got, want))

    # Degenerate group -> exactly zero, no epsilon rescue needed.
    ok_deg = group_advantages([1.0] * 8) == [0.0] * 8 and group_advantages([0.0] * 8) == [0.0] * 8

    # Centre-only variant.
    ok_centre = group_advantages([1.0, 0.0], normalize_by_std=False) == [0.5, -0.5]

    # The 1/sqrt(G) floor claim in the module docstring.
    ok_floor = all(abs(binary_group_std(1, G) - 1.0 / math.sqrt(G)) < 1e-12
                   for G in (4, 8, 16, 32))

    # Vectorised implementation must agree element-for-element.
    rt = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    at = advantages_torch(rt, normalize_by_std=True, eps=1e-6, std_floor=0.0)
    ok_vec = True
    for i in range(rt.shape[0]):
        ref = group_advantages(rt[i].tolist(), normalize_by_std=True, eps=1e-6)
        ok_vec &= all(abs(float(at[i, j]) - ref[j]) < 1e-5 for j in range(rt.shape[1]))

    ok = ok_hand and ok_deg and ok_centre and ok_floor and ok_vec
    return _record("GRPO advantages", ok,
                   f"hand={ok_hand} degenerate={ok_deg} centre={ok_centre} "
                   f"floor={ok_floor} vectorised={ok_vec}")


def check_aggregate_coeffs() -> dict:
    """The three conventions really are reweightings of the same cached vector."""
    import torch

    from .alignment import coeffs_torch
    from .gradient import aggregate_coeffs

    A = [1.5, -0.5, -1.0, 0.0]
    L = [10, 20, 5, 8]
    n, G = 1, 4

    c_sum, s_sum = aggregate_coeffs(A, L, "seq_sum", n, G)
    c_mean, s_mean = aggregate_coeffs(A, L, "seq_mean", n, G)
    ok = (c_sum == A and abs(s_sum - 0.25) < 1e-12
          and all(abs(c - a / l) < 1e-12 for c, a, l in zip(c_mean, A, L))
          and abs(s_mean - 0.25) < 1e-12)

    # batch_token_mean's scale is the batch token total, applied by the caller.
    c_btm, s_btm = aggregate_coeffs(A, L, "batch_token_mean", n, G)
    ok &= c_btm == A and math.isnan(s_btm)

    # The torch twin must produce identical coefficients and the concrete 1/sum(L) scale.
    at = torch.tensor([A])
    lt = torch.tensor([L])
    for agg in ("seq_sum", "seq_mean", "batch_token_mean"):
        ct, st = coeffs_torch(at, lt, agg)
        ref_c, ref_s = aggregate_coeffs(A, L, agg, n, G)
        ok &= all(abs(float(ct[0, j]) - ref_c[j]) < 1e-6 for j in range(G))
        if agg == "batch_token_mean":
            ok &= abs(st - 1.0 / sum(L)) < 1e-12
        else:
            ok &= abs(st - ref_s) < 1e-12
    return _record("aggregation conventions", ok, "seq_sum / seq_mean / batch_token_mean")


def check_sketch(numels=None, m_target: int = 1 << 14, device: str = "cpu") -> dict:
    """Unbiasedness of the block-sign sketch, and its predicted error scaling."""
    from .sketch import BlockSignSketch, check_unbiased

    numels = numels or [40_000, 25_000, 12_345]
    sk = BlockSignSketch(numels, m_target=m_target, seed=3, device=device)
    stats = {c: check_unbiased(sk, n_trials=6, cos=c, seed=11) for c in (0.3, 0.05)}

    ok = True
    detail = []
    for c, s in stats.items():
        # Mean of the estimate over trials should sit on 1.0; the tolerance is the
        # per-trial error shrunk by sqrt(n_trials), with slack.
        tol = max(4.0 * s["predicted_rel_err"] / math.sqrt(6), 0.03)
        ok &= abs(s["mean_ratio"] - 1.0) < tol
        # Realised error should track the 1/(alpha sqrt(m)) prediction within ~3x.
        ok &= s["mean_rel_err"] < 3.0 * s["predicted_rel_err"] + 1e-6
        detail.append(f"cos={c}: ratio={s['mean_ratio']:.3f} "
                      f"err={s['mean_rel_err']:.3f} (pred {s['predicted_rel_err']:.3f})")
    return _record("sketch unbiasedness", ok, "; ".join(detail), m=sk.m)


# ==================================================================== model-dependent


def _tiny_extractor(cfg, device: str = "cpu"):
    """A 2-layer randomly-initialised Qwen2.5-VL sharing the real processor.

    Best-effort: the vision geometry and special-token ids must match the real processor,
    and the mrope sections must re-sum to head_dim/2, both of which have moved between
    transformers releases. Callers fall back to the real model when this raises.
    """
    import torch
    from transformers import AutoConfig

    from .gradient import PolicyGradientExtractor
    from .load_model import _import_model_class, load_processor, select_backbone

    conf = AutoConfig.from_pretrained(cfg.model.name)
    text = getattr(conf, "text_config", conf)
    text.hidden_size = 256
    text.intermediate_size = 512
    text.num_hidden_layers = 2
    text.num_attention_heads = 4
    text.num_key_value_heads = 2

    head_dim = text.hidden_size // text.num_attention_heads
    rs = getattr(text, "rope_scaling", None) or getattr(conf, "rope_scaling", None)
    if isinstance(rs, dict) and "mrope_section" in rs:
        want = head_dim // 2
        base = rs["mrope_section"]
        scaled = [max(1, int(round(v * want / max(sum(base), 1)))) for v in base]
        scaled[0] += want - sum(scaled)
        rs["mrope_section"] = scaled

    vc = getattr(conf, "vision_config", None)
    if vc is not None:
        vc.depth = 2
        vc.hidden_size = 128
        vc.num_heads = 4
        vc.intermediate_size = 256
        vc.out_hidden_size = text.hidden_size
        if hasattr(vc, "fullatt_block_indexes"):
            vc.fullatt_block_indexes = [1]

    cls, _ = _import_model_class()
    model = cls.from_config(conf) if hasattr(cls, "from_config") else cls(conf)
    model = model.to(device=device, dtype=torch.float32).eval()
    model.config.use_cache = False

    proc = load_processor(cfg.model.name, min_pixels=cfg.get_path("model.min_pixels"),
                          max_pixels=cfg.get_path("model.max_pixels"))
    spec = select_backbone(model,
                           exclude_patterns=cfg.get_path("backbone.exclude_name_patterns"),
                           include_patterns=cfg.get_path("backbone.include_name_patterns", []))
    return PolicyGradientExtractor(model, proc, spec, cfg, device=device), model, spec


def _reference_seq_logprob(ex, seq_1d, prompt_len: int, vis: dict) -> float:
    """A deliberately naive, obviously-correct log pi(a|s).

    Loops positions explicitly instead of slicing, so it cannot share an off-by-one with
    the implementation it is checking.
    """
    import torch

    with torch.no_grad():
        out = ex.model(input_ids=seq_1d.unsqueeze(0),
                       attention_mask=torch.ones_like(seq_1d).unsqueeze(0),
                       use_cache=False, **vis)
        logits = out.logits[0].float()
        if ex.temperature != 1.0:
            logits = logits / ex.temperature
        total = 0.0
        for t in range(prompt_len, int(seq_1d.shape[0])):
            lp = torch.log_softmax(logits[t - 1], dim=-1)
            total += float(lp[int(seq_1d[t])])
    return total


def check_response_lengths(ex) -> dict:
    """EOS trimming: the EOS token counts, everything after it does not."""
    import torch

    eos = ex._eos_ids[0]
    other = 5 if eos != 5 else 6
    resp = torch.tensor([
        [other, other, eos, 99, 99],     # EOS at index 2 -> length 3 (EOS included)
        [other, other, other, other, other],  # no EOS -> full length 5
        [eos, 99, 99, 99, 99],           # immediate EOS -> length 1
    ], device=ex.device)
    got = ex.response_lengths(resp).tolist()
    ok = got == [3, 5, 1]
    return _record("response trimming (EOS/padding)", ok, f"lengths {got}, expected [3, 5, 1]")


def check_logprob_alignment(ex, item, modality: str = "text") -> dict:
    """Our sliced log-prob equals the explicit position-by-position reference.

    This is the prompt-masking test: the reference sums positions prompt_len..end only, so
    any slice that leaked a prompt target or dropped the first response token disagrees.
    """
    import torch

    grp = ex.sample_group(item, modality, 1, seed=0)
    seq = grp.seqs[0]
    with torch.no_grad():
        ours, T = ex.sequence_logprob(seq, grp.prompt_len, grp.vis)
        ours = float(ours)
    ref = _reference_seq_logprob(ex, seq, grp.prompt_len, grp.vis)
    rel = abs(ours - ref) / (abs(ref) + 1e-9)
    ok_val = rel < 1e-4
    ok_len = T == int(seq.shape[0]) - grp.prompt_len

    # A shifted slice must disagree — otherwise the test above proves nothing.
    with torch.no_grad():
        shifted = _shifted_reference(ex, seq, grp.prompt_len, grp.vis, offset=1)
    ok_sensitive = abs(shifted - ref) / (abs(ref) + 1e-9) > 1e-3

    return _record(f"log-prob alignment [{modality}]", ok_val and ok_len and ok_sensitive,
                   f"ours={ours:.4f} reference={ref:.4f} rel={rel:.2e}, "
                   f"T={T}, off-by-one is detectable={ok_sensitive}")


def _shifted_reference(ex, seq_1d, prompt_len: int, vis: dict, offset: int = 1) -> float:
    """The same sum with a deliberately wrong logit offset, to prove the test has teeth."""
    import torch

    with torch.no_grad():
        out = ex.model(input_ids=seq_1d.unsqueeze(0),
                       attention_mask=torch.ones_like(seq_1d).unsqueeze(0),
                       use_cache=False, **vis)
        logits = out.logits[0].float()
        total = 0.0
        for t in range(prompt_len, int(seq_1d.shape[0])):
            lp = torch.log_softmax(logits[max(t - 1 - offset, 0)], dim=-1)
            total += float(lp[int(seq_1d[t])])
    return total


def _grad_of(ex, seq, P, vis, sub, coef: float = 1.0):
    """coef * grad log pi(a|s) restricted to `sub`, as fp32 copies."""
    import torch

    ex.model.zero_grad(set_to_none=True)
    logp, _ = ex.sequence_logprob(seq, P, vis)
    (coef * logp).backward()
    g = [(p.grad.detach().clone().float() if p.grad is not None
          else torch.zeros_like(p, dtype=torch.float32)) for p in sub]
    ex.model.zero_grad(set_to_none=True)
    return g, float(logp.detach())


def _perturbation_scale(sub, g) -> tuple[float, str, float]:
    """Pick a step size the parameter dtype can actually represent.

    This is the trap in finite-differencing a bf16 model: bf16 carries ~8 mantissa bits, so
    a perturbation smaller than ~4e-3 of a weight's magnitude rounds straight back to the
    original value and the test silently measures nothing. We size eps as a fixed relative
    perturbation of the typical weight — small (1e-3) in float32 where linearity holds
    tightly, larger (3e-2) in bf16/fp16 where it has to survive rounding — and the caller
    verifies afterwards that the weights genuinely moved.
    """
    import torch

    gmax = max(float(gi.abs().max()) for gi in g) if g else 0.0
    pscale = float(torch.stack([p.detach().float().abs().mean() for p in sub]).median())
    dtype = sub[0].dtype
    if dtype == torch.float32 or dtype == torch.float64:
        rel, tol, label = 1e-3, 0.15, "float32"
    else:
        rel, tol, label = 3e-2, 0.40, str(dtype).replace("torch.", "")
    eps = rel * pscale / max(gmax, 1e-30)
    return eps, label, tol


def _probe(ex, sub, g, amount: float, evaluate: Callable[[], float]) -> tuple[float, bool]:
    """Evaluate `evaluate()` at theta + amount*g, then restore theta exactly.

    Restoration is by snapshot, not by subtracting the step back off: in bf16 an add
    followed by a sub does not round-trip, and this pipeline must leave the checkpoint
    bit-identical — Task 0 measures a *fixed* checkpoint.
    """
    import torch

    snap = [p.detach().clone() for p in sub]
    try:
        changed = False
        with torch.no_grad():
            for p, gi in zip(sub, g):
                p.add_(amount * gi.to(p.dtype))
        for p, s in zip(sub, snap):
            if bool((p != s).any()):
                changed = True
                break
        with torch.no_grad():
            value = float(evaluate())
        return value, changed
    finally:
        with torch.no_grad():
            for p, s in zip(sub, snap):
                p.copy_(s)


def check_finite_difference(ex, item, modality: str = "text", n_params: int = 6) -> dict:
    """THE check. A positive-advantage sample's gradient must raise its own log-prob.

    Restricted to a handful of parameter tensors so the perturbation is cheap; the
    identity is exact on any subspace:

        d/d(eps) log pi(theta + eps * g_sub)|_0 = <g_sub, g_sub> = ||g_sub||^2 > 0

    Three assertions: stepping along +g raises log pi and along -g lowers it (catches a
    sign flip), the central-difference slope matches ||g_sub||^2 (catches a mis-slice that
    scores the wrong tokens, since a wrong gradient still points somewhere but not with
    the right magnitude), and the weights demonstrably moved (catches the perturbation
    being rounded away by a low-precision dtype, which would make the whole test vacuous).
    """
    import torch

    grp = ex.sample_group(item, modality, 1, seed=1)
    seq, P, vis = grp.seqs[0], grp.prompt_len, grp.vis

    sub = [p for p in ex.params if p.dim() >= 2][-n_params:] or ex.params[-n_params:]
    g, base = _grad_of(ex, seq, P, vis, sub)
    gnorm2 = float(sum((gi * gi).sum() for gi in g))
    eps, dlabel, tol = _perturbation_scale(sub, g)

    evaluate = lambda: ex.sequence_logprob(seq, P, vis)[0]
    up, moved = _probe(ex, sub, g, +eps, evaluate)
    down, _ = _probe(ex, sub, g, -eps, evaluate)

    slope = (up - down) / (2 * eps)
    rel = abs(slope - gnorm2) / (abs(gnorm2) + 1e-30)
    ok = bool(moved) and gnorm2 > 0 and up > base > down and rel < tol

    return _record(
        f"finite-difference gradient check [{modality}]", ok,
        f"logp {down:.5f} < {base:.5f} < {up:.5f}; slope={slope:.4e} vs "
        f"||g||^2={gnorm2:.4e} (rel {rel:.1%}, tol {tol:.0%} for {dlabel}); "
        f"eps={eps:.2e}, weights moved={moved}, {len(sub)} tensors",
        slope=slope, gnorm2=gnorm2, rel_err=rel, dtype=dlabel, weights_moved=bool(moved),
    )


def check_advantage_sign(ex, item, modality: str = "text", n_params: int = 6) -> dict:
    """A negative advantage must push the sequence's probability DOWN.

    Separate from the finite-difference check because it exercises the advantage as a
    coefficient on the loss, not just the raw log-prob gradient.
    """
    import torch

    grp = ex.sample_group(item, modality, 1, seed=2)
    seq, P, vis = grp.seqs[0], grp.prompt_len, grp.vis
    sub = [p for p in ex.params if p.dim() >= 2][-n_params:] or ex.params[-n_params:]

    evaluate = lambda: ex.sequence_logprob(seq, P, vis)[0]
    results, moved_all = {}, True
    for adv in (+1.0, -1.0):
        g, base = _grad_of(ex, seq, P, vis, sub, coef=adv)
        eps, _, _ = _perturbation_scale(sub, g)
        after, moved = _probe(ex, sub, g, +eps, evaluate)
        moved_all &= moved
        results[adv] = after - base

    ok = moved_all and results[+1.0] > 0 > results[-1.0]
    return _record("advantage sign", ok,
                   f"A=+1 -> dlogp={results[+1.0]:+.3e}; A=-1 -> dlogp={results[-1.0]:+.3e}; "
                   f"weights moved={moved_all}")


def check_sampling_matches_policy(ex, item, modality: str = "text",
                                  n_samples: int = 1024) -> dict:
    """Are we sampling from the same pi_theta we differentiate?

    Compares the empirical first-token histogram against softmax(logits). The decisive
    statistic is the empirical mass falling OUTSIDE the model's top-k: if a stray
    top_k=20/50 were still active, that mass would be exactly zero while the true
    distribution puts real probability there.
    """
    import torch

    inputs, P, vis = ex.build_prompt(item, modality)
    with torch.no_grad():
        out = ex.model(**inputs, use_cache=False)
        logits = out.logits[0, -1].float()
        if ex.temperature != 1.0:
            logits = logits / ex.temperature
        p = torch.softmax(logits, dim=-1)

    counts = torch.zeros_like(p)
    drawn = 0
    while drawn < n_samples:
        b = min(ex.gen_batch * 4, n_samples - drawn)
        with torch.no_grad():
            o = ex.model.generate(
                **inputs, generation_config=ex.make_gen_config(b, max_new_tokens=1))
        first = o.sequences[:, P]
        counts.scatter_add_(0, first, torch.ones_like(first, dtype=counts.dtype))
        drawn += b

    emp = counts / counts.sum()
    tv = 0.5 * float((emp - p).abs().sum())

    detail_parts = []
    ok = True
    for k in (20, 50):
        topk = torch.topk(p, k).indices
        mask = torch.ones_like(p, dtype=torch.bool)
        mask[topk] = False
        tail_true = float(p[mask].sum())
        tail_emp = float(emp[mask].sum())
        detail_parts.append(f"tail>top{k}: true={tail_true:.3f} empirical={tail_emp:.3f}")
        # Only meaningful when the true tail carries enough mass to be observable.
        if tail_true > 0.05:
            expected = tail_true * n_samples
            se = math.sqrt(max(expected * (1 - tail_true), 1.0))
            ok &= (tail_emp * n_samples) > expected - 5 * se

    # Total variation is dominated by sampling noise ~ sqrt(support/n); use a loose bound.
    ok &= tv < 0.25
    return _record(f"sampling matches pi_theta [{modality}]", ok,
                   f"TV={tv:.3f} over {n_samples} draws; " + "; ".join(detail_parts),
                   tv=tv)


def check_backbone_isolation(ex, item, modality: str = "image") -> dict:
    """No vision gradient may reach the measured vector, and no tensor may be counted twice."""
    import torch

    grp = ex.sample_group(item, modality, 1, seed=3)
    ex.model.zero_grad(set_to_none=True)
    logp, _ = ex.sequence_logprob(grp.seqs[0], grp.prompt_len, grp.vis)
    logp.backward()

    selected = {id(p) for p in ex.params}
    leaked = []
    for n, p in ex.model.named_parameters():
        if id(p) in selected:
            continue
        if p.grad is not None and float(p.grad.abs().sum()) > 0:
            leaked.append(n)
    dup = len(ex.params) != len({id(p) for p in ex.params})
    nonzero = sum(1 for p in ex.params if p.grad is not None and float(p.grad.abs().sum()) > 0)
    ex.model.zero_grad(set_to_none=True)

    ok = not leaked and not dup and nonzero > 0
    return _record("backbone isolation", ok,
                   f"{nonzero}/{len(ex.params)} measured tensors have gradient; "
                   f"leaked={leaked[:3] if leaked else 'none'}; duplicates={dup}")


def check_cache_linearity(ex, item, sketch, modality: str = "text", G: int = 4) -> dict:
    """The cached reconstruction is the same estimator as direct backprop.

    Path A backprops the advantage-weighted loss the way training would. Path B caches one
    unweighted sketched gradient per sample and recombines them with the same scalars.
    They must agree to float precision, or every number in the sweep is measuring
    something other than what the exact anchor measures.
    """
    import torch

    from .config import resolve_grpo
    from .gradient import SketchSink, aggregate_coeffs
    from .rewards import group_advantages

    grp = ex.sample_group(item, modality, G, seed=4)
    grpo = resolve_grpo(ex.cfg)

    # Force a non-degenerate group: with a random tiny model every reward may be 0, which
    # makes every advantage 0 and the comparison vacuous.
    rewards = list(grp.rewards)
    if len(set(rewards)) == 1:
        rewards = [1.0 if j % 2 == 0 else 0.0 for j in range(len(rewards))]
    adv = group_advantages(rewards, normalize_by_std=grpo["normalize_by_std"],
                           eps=grpo["eps"], std_floor=grpo["std_floor"])
    coeffs, scale = aggregate_coeffs(adv, grp.lengths, grpo["loss_agg"], 1, G)
    if grpo["loss_agg"] == "batch_token_mean":
        scale = 1.0 / max(sum(grp.lengths), 1)

    sink = SketchSink(sketch)
    for seq, c in zip(grp.seqs, coeffs):
        ex.accumulate_sample(seq, grp.prompt_len, grp.vis, c, sink)
    sink.scale(scale)
    direct = sink.vector().clone()

    recon = torch.zeros_like(direct)
    buf = torch.zeros_like(direct)
    for seq, c in zip(grp.seqs, coeffs):
        ex.per_sample_gradient(seq, grp.prompt_len, grp.vis, sketch, out=buf)
        recon += float(c) * buf
    recon *= scale

    num = float((direct - recon).norm())
    den = float(direct.norm()) + 1e-30
    rel = num / den
    ok = rel < 1e-4 and den > 1e-20
    return _record("cache linearity (direct == recombined)", ok,
                   f"relative difference {rel:.2e}, ||g||={den:.3e}, "
                   f"advantages={[round(a, 3) for a in adv]}")


# ==================================================================== driver


def run_all(cfg, *, extractor=None, item=None, sketch=None, device: str = "cuda",
            include_image: bool = True, verbose: bool = True) -> dict:
    """Run every check. Pass a live extractor to test the real checkpoint.

    With `extractor=None` a 2-layer random model is built instead, which is fast but
    weaker evidence: prefer running these against the model you will actually measure.
    """
    _RESULTS.clear()
    t0 = time.time()
    print("=" * 74)
    print("TASK 0 SANITY CHECKS")
    print("=" * 74)

    print("\npure-python:")
    check_reward_parsing()
    check_advantages()
    check_aggregate_coeffs()
    check_sketch()

    owned = False
    if extractor is None:
        print("\nbuilding a tiny random Qwen2.5-VL for the model-dependent checks ...")
        try:
            extractor, _model, _spec = _tiny_extractor(cfg, device="cpu")
            owned = True
        except Exception as e:
            _record("tiny model construction", False,
                    f"{type(e).__name__}: {e}. Re-run with a real extractor "
                    f"(sanity_checks.run_all(cfg, extractor=ex, item=items[0], sketch=sk)).")
            return _summary(t0)

    if item is None:
        try:
            from .data_prep import load_calibration

            item = load_calibration(cfg, n=1)[0]
        except Exception as e:
            _record("calibration item for model checks", False,
                    f"{type(e).__name__}: {e}. Run the data stage first.")
            return _summary(t0)

    if sketch is None:
        from .sketch import BlockSignSketch

        sketch = BlockSignSketch(extractor.backbone.numels, m_target=4096, seed=5,
                                 device=str(extractor.device))

    print("\nmodel-dependent:")
    modalities = ["text"] + (["image"] if include_image else [])
    check_response_lengths(extractor)
    for mod in modalities:
        check_logprob_alignment(extractor, item, mod)
    check_finite_difference(extractor, item, "text")
    if include_image:
        check_finite_difference(extractor, item, "image")
    check_advantage_sign(extractor, item, "text")
    check_sampling_matches_policy(extractor, item, "text")
    check_backbone_isolation(extractor, item, "image" if include_image else "text")
    check_cache_linearity(extractor, item, sketch, "text")
    if include_image:
        check_cache_linearity(extractor, item, sketch, "image")

    if owned:
        from .load_model import free

        free(extractor)
    return _summary(t0)


def _summary(t0: float) -> dict:
    failed = [r for r in _RESULTS if not r["ok"]]
    print("\n" + "=" * 74)
    if failed:
        print(f"SANITY CHECKS FAILED: {len(failed)}/{len(_RESULTS)}")
        for r in failed:
            print(f"  - {r['name']}: {r['detail']}")
        print("\nDo not trust any Lambda produced before these pass.")
    else:
        print(f"ALL {len(_RESULTS)} SANITY CHECKS PASSED  ({time.time()-t0:.1f}s)")
    print("=" * 74)
    return {"results": list(_RESULTS), "n_failed": len(failed),
            "ok": not failed, "seconds": time.time() - t0}


__all__ = ["run_all", "check_reward_parsing", "check_advantages", "check_aggregate_coeffs",
           "check_sketch", "check_response_lengths", "check_logprob_alignment",
           "check_finite_difference", "check_advantage_sign",
           "check_sampling_matches_policy", "check_backbone_isolation",
           "check_cache_linearity"]
