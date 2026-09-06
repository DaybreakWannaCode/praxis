"""Load a frozen Qwen2.5-VL and isolate the language-backbone parameter block.

Two jobs, both of which have a trap in them.

1. Loading. transformers has renamed Qwen2.5-VL's classes and its submodule paths more
   than once (`Qwen2_5_VLForConditionalGeneration`, then the generic
   `AutoModelForImageTextToText`; `visual.*` at top level, then `model.visual.*`). We try
   a cascade and then *verify* what we got rather than trusting a naming convention.

2. Backbone isolation. Excluding the vision tower is not cosmetic: grad J_t is identically
   zero on the encoder and merger, so those coordinates contribute nothing to
   <g_v, g_t> while doubling gradient memory. The trap is `tie_word_embeddings`, which is
   True for Qwen2.5-3B: `lm_head.weight` and `model.embed_tokens.weight` are the *same*
   tensor (~311M coordinates). Walking named_parameters() naively and concatenating would
   put that tensor into the flat gradient vector twice, double-weighting 10% of the
   coordinates in every inner product. We dedupe by `id(p)` and report what was tied.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

# Modules whose gradients we never want. Applied as substring tests against the fully
# qualified parameter name, so "visual" catches "model.visual.blocks.0.attn.qkv.weight"
# and "visual.merger.mlp.0.weight" alike.
_DEFAULT_EXCLUDE = ("visual", "vision", "merger", "patch_embed", "image_newline")


@dataclass
class BackboneSpec:
    """The measured parameter block, plus enough bookkeeping to audit it."""

    params: list = field(repr=False)
    names: list[str]
    numels: list[int]
    d: int
    n_tensors: int
    excluded_d: int
    excluded_n_tensors: int
    tied_dropped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"backbone block: {self.n_tensors} tensors, d = {self.d:,} parameters "
            f"({self.d * 2 / 1e9:.2f} GB of bf16 gradient)",
            f"excluded      : {self.excluded_n_tensors} tensors, {self.excluded_d:,} parameters",
        ]
        if self.tied_dropped:
            lines.append(
                f"tied weights deduped ({len(self.tied_dropped)}): "
                + ", ".join(self.tied_dropped[:4])
                + (" ..." if len(self.tied_dropped) > 4 else "")
            )
        return "\n".join(lines)


def _import_model_class():
    """Return (class, label) for the best available Qwen2.5-VL model class."""
    import transformers

    for name in ("Qwen2_5_VLForConditionalGeneration",
                 "AutoModelForImageTextToText",
                 "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls, name
    raise ImportError(
        "transformers exposes none of Qwen2_5_VLForConditionalGeneration / "
        "AutoModelForImageTextToText / AutoModelForVision2Seq. Upgrade transformers."
    )


def load_processor(model_name: str, *, min_pixels: int | None = None,
                   max_pixels: int | None = None):
    """Processor with the vision token budget applied.

    min_pixels/max_pixels bound Qwen2.5-VL's dynamic resolution. Capping max_pixels is the
    single biggest lever on prompt length: it decides how many vision tokens each image
    becomes, which drives both activation memory and the cost of every forward pass.
    """
    from transformers import AutoProcessor

    kw: dict[str, Any] = {}
    if min_pixels is not None:
        kw["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        kw["max_pixels"] = int(max_pixels)
    try:
        return AutoProcessor.from_pretrained(model_name, **kw)
    except TypeError:
        # Older processors take the bounds on the image processor instead.
        proc = AutoProcessor.from_pretrained(model_name)
        for k, v in kw.items():
            if hasattr(proc, "image_processor"):
                setattr(proc.image_processor, k, int(v))
        return proc


def load_model(cfg, *, device: str = "cuda", for_generation_only: bool = False):
    """Load the checkpoint frozen. No optimizer, no training mode, no weight updates.

    `for_generation_only=True` skips gradient plumbing entirely (used by the captioner).
    """
    import torch

    cls, label = _import_model_class()
    dtype = getattr(torch, str(cfg.model.dtype))

    kw: dict[str, Any] = {"dtype": dtype}
    attn = cfg.get_path("model.attn_implementation")
    if attn:
        kw["attn_implementation"] = attn

    try:
        model = cls.from_pretrained(cfg.model.name, **kw)
    except TypeError:
        # transformers < 4.56 spells it torch_dtype.
        kw["torch_dtype"] = kw.pop("dtype")
        model = cls.from_pretrained(cfg.model.name, **kw)

    model = model.to(device)
    # eval() everywhere: dropout must be off or pi_theta is not the distribution we
    # think we are differentiating, and two calls on the same input would disagree.
    model.eval()
    model.config.use_cache = False

    proc = load_processor(
        cfg.model.name,
        min_pixels=cfg.get_path("model.min_pixels"),
        max_pixels=cfg.get_path("model.max_pixels"),
    )

    if for_generation_only:
        for p in model.parameters():
            p.requires_grad_(False)
        return model, proc, None

    spec = select_backbone(
        model,
        exclude_patterns=cfg.get_path("backbone.exclude_name_patterns", _DEFAULT_EXCLUDE),
        include_patterns=cfg.get_path("backbone.include_name_patterns", []),
        dedupe_tied=bool(cfg.get_path("backbone.dedupe_tied", True)),
    )

    if cfg.get_path("model.gradient_checkpointing", True):
        enable_gradient_checkpointing(model)

    return model, proc, spec


def enable_gradient_checkpointing(model) -> None:
    """Turn on checkpointing and make sure gradient can actually flow through it.

    use_reentrant=False is the non-legacy implementation and is what plays well with
    frozen submodules. enable_input_require_grads() is the standard companion: with the
    reentrant path (and some module graphs) a checkpointed block whose inputs do not
    require grad silently produces no gradient at all. embed_tokens is inside the measured
    block here so it is already satisfied, but the hook costs nothing and removes the
    failure mode if someone narrows `include_name_patterns`.
    """
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False


def select_backbone(model, *, exclude_patterns=_DEFAULT_EXCLUDE, include_patterns=(),
                    dedupe_tied: bool = True) -> BackboneSpec:
    """Pick the language-backbone parameters and freeze everything else.

    Iterates with remove_duplicate=False so tied tensors are *visible* and can be reported,
    then dedupes by identity. Anything not selected gets requires_grad_(False) so a stray
    p.grad can never leak into the measured vector.
    """
    exclude = tuple(exclude_patterns or ())
    include = tuple(include_patterns or ())

    params, names, numels = [], [], []
    seen: dict[int, str] = {}
    tied_dropped: list[str] = []
    excluded_d = 0
    excluded_n = 0

    for n, p in model.named_parameters(remove_duplicate=False):
        keep = not any(pat in n for pat in exclude)
        if keep and include:
            keep = any(pat in n for pat in include)
        if not keep:
            if id(p) not in seen:
                excluded_d += p.numel()
                excluded_n += 1
            continue
        if dedupe_tied and id(p) in seen:
            tied_dropped.append(f"{n} (tied to {seen[id(p)]})")
            continue
        seen[id(p)] = n
        params.append(p)
        names.append(n)
        numels.append(p.numel())

    selected_ids = {id(p) for p in params}
    for p in model.parameters():
        p.requires_grad_(id(p) in selected_ids)

    if not params:
        raise RuntimeError(
            "backbone selection matched zero parameters. Check "
            "backbone.exclude_name_patterns / include_name_patterns against the actual "
            "parameter names (print a few with model.named_parameters())."
        )

    spec = BackboneSpec(
        params=params, names=names, numels=numels, d=sum(numels), n_tensors=len(params),
        excluded_d=excluded_d, excluded_n_tensors=excluded_n, tied_dropped=tied_dropped,
    )
    _audit_backbone(spec)
    return spec


def _audit_backbone(spec: BackboneSpec) -> None:
    """Sanity-check the selection instead of trusting the name patterns.

    Catches the two ways this silently goes wrong: the vision tower not actually being
    excluded (a renamed submodule that no pattern matches), and the whole thing being
    empty or absurdly small because a rename broke the language-side names too.
    """
    leaked = [n for n in spec.names if any(k in n.lower() for k in ("visual", "vision", "merger"))]
    if leaked:
        raise RuntimeError(
            f"vision parameters leaked into the backbone block: {leaked[:5]}. "
            "Update backbone.exclude_name_patterns for this transformers version."
        )
    if spec.excluded_n_tensors == 0:
        raise RuntimeError(
            "nothing was excluded — the vision tower was not found under any of the "
            "exclude patterns. Print model.named_parameters() and fix the patterns; "
            "measuring alignment over the encoder would add pure zeros from grad J_t."
        )
    has_layer = any(".layers." in n for n in spec.names)
    if not has_layer:
        raise RuntimeError(
            "no transformer decoder layers in the selected block; the include/exclude "
            "patterns are almost certainly wrong for this transformers version."
        )


def free(*objs) -> None:
    """Drop references and reclaim GPU memory between stages."""
    import torch

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cuda_report() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cuda: unavailable"
    free_b, total_b = torch.cuda.mem_get_info()
    return (f"cuda: {torch.cuda.get_device_name(0)} | "
            f"free {free_b/1e9:.1f} / {total_b/1e9:.1f} GB | "
            f"allocated {torch.cuda.memory_allocated()/1e9:.1f} GB | "
            f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


__all__ = ["BackboneSpec", "load_model", "load_processor", "select_backbone",
           "enable_gradient_checkpointing", "free", "cuda_report"]