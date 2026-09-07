"""One-response-at-a-time backend interface, synthetic and Qwen LoRA implementations."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .core import isolated_rng, trainables


@dataclass
class Response:
    payload: object
    text: str
    reward: float
    parsed: bool
    answer: object
    length: int
    truncated: bool = False


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.05, -0.03], [-0.02, 0.04]], dtype=torch.float64))
        self.register_buffer("reference_weight", self.weight.detach().clone())


class TinyBackend:
    """Two-action toy problem. This is never evidence about VLMs."""
    def __init__(self):
        self.model = TinyPolicy()
        self.metadata = {"backend": "synthetic", "reference": "immutable registered buffer"}

    def distribution(self, item, modality, reference=False):
        w = self.model.reference_weight if reference else self.model.weight
        x = torch.tensor(item.feature, dtype=w.dtype)
        if modality == "image":
            x = x * torch.tensor([0.9, 1.1], dtype=w.dtype)
        return torch.log_softmax(w @ x, dim=0)

    def sample(self, item, modality, count, seed, *, greedy=False):
        with isolated_rng(seed), torch.no_grad():
            lp = self.distribution(item, modality)
            choices = [int(lp.argmax())] if greedy else torch.multinomial(lp.exp(), count, replacement=True).tolist()
        result = []
        for choice in choices:
            answer = "AB"[choice]
            result.append(Response((item, modality, choice), f"<answer>{answer}</answer>",
                                   float(answer == item.answer), True, answer, 1))
        return result

    def logprobs(self, response, *, reference=False):
        item, modality, choice = response.payload
        return self.distribution(item, modality, reference)[choice:choice+1]


class QwenBackend:
    """Single-device LoRA engineering backend, NOT distributed Praxis parity.

    Uses the existing Task 0 sequence/EOS implementation. Only decoder LoRA
    parameters are optimized. Adapter-disabled base weights define the fixed KL
    reference. No mutable second model is needed for this reference policy.
    """
    def __init__(self, cfg):
        import re
        self.answer_parser = cfg.get('answer_parser', 'legacy')
        if self.answer_parser not in ('legacy', 'explicit_final_v2', 'explicit_final_v3'):
            raise ValueError('Unknown answer parser profile')
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from task_0.src.config import Cfg
        from task_0.src.gradient import PolicyGradientExtractor
        from task_0.src.load_model import enable_gradient_checkpointing

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen pilot requires CUDA; use --backend synthetic for CPU checks")
        model_name, revision = cfg["model"], cfg["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Set a 40-character model commit SHA before a Qwen run")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=False,
            local_files_only=cfg.get("local_files_only", True),
        ).to("cuda")
        targets = [n for n, m in base.named_modules()
                   if isinstance(m, torch.nn.Linear) and ".layers." in n
                   and not any(s in n for s in ("visual", "vision"))
                   and n.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
        if not targets:
            raise RuntimeError("No language attention modules matched; refusing ambiguous LoRA placement")
        self.model = get_peft_model(base, LoraConfig(
            r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        ))
        for n, p in trainables(self.model).items():
            if "lora_" not in n or "visual" in n:
                raise RuntimeError(f"Unexpected trainable coordinate {n}")
            p.data = p.data.float()  # Small adapter steps must not round back in BF16.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        enable_gradient_checkpointing(self.model)
        self.model.train()  # Non-reentrant checkpointing active; all dropout disabled.
        proc = AutoProcessor.from_pretrained(
            model_name, revision=revision, local_files_only=cfg.get("local_files_only", True),
            min_pixels=4*28*28, max_pixels=cfg["max_pixels"],
        )
        gen_batch = cfg.get("generation_batch_size", 1)
        if not isinstance(gen_batch, int) or not 1 <= gen_batch <= 8:
            raise ValueError("generation_batch_size must be between 1 and 8")
        extractor_cfg = Cfg({"policy": {"temperature": 1.0, "max_new_tokens": cfg["max_new_tokens"],
                              "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                              "system_prompt": cfg["system_prompt"]}, "sampling": {"gen_batch": gen_batch}})

        class Extractor(PolicyGradientExtractor):
            def _messages(self, item, modality):
                body = item.question + "\n" + "\n".join(item.action_list)
                if modality == "image":
                    content = [{"type": "image"}, {"type": "text", "text": body}]
                else:
                    content = [{"type": "text", "text": item.situation + "\n" + body}]
                return [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                        {"role": "user", "content": content}]

        self.extractor = Extractor(self.model, proc, SimpleNamespace(params=list(trainables(self.model).values())),
                                   extractor_cfg, device="cuda")
        # Avoid forwarding generic PEFT **kwargs-only signatures to Task 0's
        # logits-tail auto-detection. The full-output fallback is correct but slower.
        self.metadata = {"backend": "qwen_lora_engineering", "model": model_name,
                         "revision": revision, "reference": "adapter-disabled pinned base",
                         "target_modules": targets, "config": cfg, "answer_parser": self.answer_parser,
                         "gpu": torch.cuda.get_device_name(0)}

    def sample(self, item, modality, count, seed, *, greedy=False):
        from .answer_parsing import parse_choice, score_completion
        if self.answer_parser=='explicit_final_v3':
            from .answer_parsing_v3 import parse_choice, score_completion
        ex = self.extractor
        was_training = self.model.training
        try:
            self.model.eval()
            with isolated_rng(seed):
                if not greedy:
                    group = ex.sample_group(item, modality, count)
                    seqs, texts, lengths, vis, plen = group.seqs, group.texts, group.lengths, group.vis, group.prompt_len
                else:
                    inputs, plen, vis = ex.build_prompt(item, modality)
                    gc = ex.make_gen_config(1)
                    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
                    with torch.no_grad():
                        full = self.model.generate(**inputs, generation_config=gc).sequences
                    length = int(ex.response_lengths(full[:, plen:])[0])
                    seqs = [full[0, :plen+length]]
                    texts = [ex.tok.decode(seqs[0][plen:], skip_special_tokens=True)]
                    lengths = [length]
                result = []
                for seq, text, length in zip(seqs, texts, lengths):
                    reward, parsed = score_completion(text, item.answer, item.action_list, profile=self.answer_parser)
                    result.append(Response((seq, plen, vis), text, reward, parsed,
                                           parse_choice(text, item.action_list, profile=self.answer_parser), length,
                                           int(seq[-1]) not in ex._eos_ids))
                return result
        finally:
            self.model.train(was_training)

    def logprobs(self, response, *, reference=False):
        seq, plen, vis = response.payload
        if reference:
            with self.model.disable_adapter(), torch.no_grad():
                return self.extractor.sequence_logprob(seq, plen, vis, per_token=True)[2]
        return self.extractor.sequence_logprob(seq, plen, vis, per_token=True)[2]
