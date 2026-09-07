"""Shared numerical primitives; no model downloads or GPU dependencies."""
from __future__ import annotations

import copy
import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derived_seed(seed, *parts):
    key = repr((int(seed), parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG topology differs from this host")
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng(seed):
    state = rng_state()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(state)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def digest(value):
    """Stable content digest, also usable for optimizer/RNG replay checks."""
    h = hashlib.sha256()

    def visit(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            # Feed the contiguous buffer directly: full-model tensors can be GBs.
            # tobytes() would allocate another equally large temporary copy.
            h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode())
            h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=repr):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for x in v:
                visit(x)
        else:
            h.update(repr((type(v).__name__, v)).encode())
        h.update(b"\0")

    visit(value)
    return h.hexdigest()


def trainables(model):
    # named_parameters deduplicates tied tensors.
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def coordinate_manifest(model):
    return [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype),
             "numel": p.numel(), "trainable": p.requires_grad}
            for n, p in model.named_parameters()]


def capture(model, optimizer, scheduler, *, compact=False, scaler=None):
    """Full parent; compact child stores mutable params/buffers relative to parent.

    Reference state must be registered in model state, or immutable in the backend.
    Compact children are restored ONLY after their associated full parent.
    """
    names = set(trainables(model)) | {n for n, _ in model.named_buffers()}
    state = model.state_dict()
    if compact:
        state = {n: t for n, t in state.items() if n in names}
    return {
        "model": cpu_copy(state), "compact": compact,
        "coordinates": coordinate_manifest(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "scheduler": cpu_copy(scheduler.state_dict()),
        "rng": rng_state(), "modes": {n: m.training for n, m in model.named_modules()},
        "scaler": cpu_copy(scaler.state_dict()) if scaler is not None else None,
    }


def restore(state, model, optimizer, scheduler, *, scaler=None):
    if coordinate_manifest(model) != state["coordinates"]:
        raise ValueError("Checkpoint parameter names, shapes, precision or trainability differ")
    model.load_state_dict(state["model"], strict=not state["compact"])
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    if (scaler is None) != (state["scaler"] is None):
        raise ValueError("Checkpoint scaler mismatch")
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    for n, m in model.named_modules():
        m.training = state["modes"][n]
    model.zero_grad(set_to_none=True)
    restore_rng(state["rng"])


def assert_frozen_unchanged(model, parent):
    for n, p in model.named_parameters():
        if not p.requires_grad and not torch.equal(p.detach().cpu(), parent["model"][n]):
            raise AssertionError(f"Unmeasured frozen parameter moved: {n}")


def weights(model):
    return {n: p.detach().float().cpu().clone() for n, p in trainables(model).items()}


def displacement(model, before):
    if set(trainables(model)) != set(before):
        raise ValueError("Update coordinates changed")
    return {n: p.detach().float().cpu() - before[n] for n, p in trainables(model).items()}


def dot(a, b):
    if set(a) != set(b):
        raise ValueError("Dot-product coordinate mismatch")
    total = 0.0
    for name in sorted(a):
        if a[name].shape != b[name].shape:
            raise ValueError(f"Shape mismatch: {name}")
        x, y = a[name].reshape(-1), b[name].reshape(-1)
        # Promote inputs before multiplying/reducing, not the already rounded dot.
        for start in range(0, x.numel(), 262144):
            value = (x[start:start+262144].double() * y[start:start+262144].double()).sum()
            total += value.item()
    if not np.isfinite(total):
        raise FloatingPointError("Nonfinite dot product")
    return total


def alignment(g, delta):
    a = dot(g, delta)
    gn, dn = dot(g, g)**0.5, dot(delta, delta)**0.5
    return {"alignment": a, "visual_gradient_norm": gn, "update_norm": dn,
            "cosine": a / (gn * dn) if gn and dn else None}


def loo_advantages(rewards):
    """Unstandardized, detached RLOO weights for expected sequence reward."""
    r = torch.as_tensor(rewards, dtype=torch.float64).detach()
    if r.ndim != 1 or r.numel() < 2 or not torch.isfinite(r).all():
        raise ValueError("RLOO requires >=2 finite independent rewards")
    return r - (r.sum() - r) / (r.numel() - 1)


def grpo_advantages(rewards):
    r = torch.as_tensor(rewards, dtype=torch.float64).detach()
    if r.ndim != 1 or r.numel() < 2 or not torch.isfinite(r).all():
        raise ValueError("GRPO requires >=2 finite rewards")
    return (r - r.mean()) / (r.std(unbiased=True) + 1e-6)


def clipped_loss(logp, old_logp, ref_logp, advantage, *, clip, kl_coef):
    ratio = (logp - old_logp).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    ref_ratio_log = ref_logp - logp
    kl = ref_ratio_log.exp() - ref_ratio_log - 1
    return policy + kl_coef * kl
