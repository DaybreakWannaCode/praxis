"""Block-sign sketch: an unbiased, O(d)-time compression of gradient vectors.

Why this exists
---------------
Every estimate in the sweep is a linear reweighting of the same per-(item, sample)
gradients v_ij = grad_theta sum_t log pi(a_ijt). Caching those makes the whole noise study
free — but there are 2 * n_pool * K of them (25,600 at the defaults) and each is d = 3.09e9
floats. Caching them raw is 200 TB. Sketched to m = 2^18 floats it is 13 GB.

The sketch
----------
Fix random signs sigma_k in {-1,+1}, one per coordinate, and sum contiguous blocks of L
coordinates:

    (Sx)_b = sum_{k in block b} sigma_k x_k

Then, over the randomness of sigma,

    E<Sx, Sy> = sum_b sum_{k,l in b} E[sigma_k sigma_l] x_k y_l
              = sum_b sum_{k in b} x_k y_k                        (E[sigma_k sigma_l] = delta_kl)
              = <x, y>                                            exactly, no rescaling.

    Var<Sx, Sy> ~ (||x||^2 ||y||^2 + <x,y>^2) / m.

Two consequences that matter for Task 0:

  * The relative sketch error on Lambda is ~ 1 / (|alpha| sqrt(m)), where alpha is the true
    cosine. That is why the exact anchor runs first: it measures |alpha|, which is what
    tells us whether m is large enough. `recommended_m` turns that into a number.
  * The signs are FIXED across the whole study, so sketch error is not fresh noise per
    resample — but different resamples produce different g_v, g_t, so it does show up as
    extra spread. It therefore inflates the reported noise floor rather than hiding it,
    which is the safe direction. `check_unbiased` measures how much.

Blocks are contiguous in the flattened tensor, which means neighbouring weights (strongly
correlated in a gradient) land in the same block. The random signs are exactly what makes
that harmless: without them the cross terms sum_{k != l in b} x_k y_l do not cancel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SIGN_GEN_CHUNK = 1 << 26   # 64M elements per RNG call
_PROJ_CHUNK_ELEMS = 1 << 23  # ~8M elements of fp32 scratch per projection chunk


@dataclass
class SketchPlan:
    block_size: int
    n_blocks: list[int]     # per tensor
    offsets: list[int]      # per tensor, start index into the m-dim output
    m: int
    d: int


def plan_sketch(numels: list[int], m_target: int) -> SketchPlan:
    """Choose one global block size L and lay out per-tensor block ranges.

    Blocks never span tensors: a tensor with N elements gets ceil(N / L) blocks, its tail
    block zero-padded. Keeping blocks inside tensors means the realised m is slightly
    above d / L (one partial block per tensor) and, more usefully, that per-tensor
    contributions to Lambda stay separable for the per-block diagnostics.
    """
    d = sum(numels)
    if m_target >= d:
        block_size = 1
    else:
        block_size = max(1, int(round(d / max(m_target, 1))))
    n_blocks = [max(1, math.ceil(n / block_size)) for n in numels]
    offsets, acc = [], 0
    for nb in n_blocks:
        offsets.append(acc)
        acc += nb
    return SketchPlan(block_size=block_size, n_blocks=n_blocks, offsets=offsets, m=acc, d=d)


class BlockSignSketch:
    """Fixed random-sign block sketch over a list of parameter tensors."""

    def __init__(self, numels: list[int], *, m_target: int, seed: int,
                 device: str = "cuda"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.seed = int(seed)
        self.plan = plan_sketch(list(numels), int(m_target))
        self.numels = list(numels)
        self._signs = [self._make_signs(i, n) for i, n in enumerate(self.numels)]

    # -- properties ------------------------------------------------------------
    @property
    def m(self) -> int:
        return self.plan.m

    @property
    def d(self) -> int:
        return self.plan.d

    def sign_bytes(self) -> int:
        return sum(s.numel() for s in self._signs)

    def describe(self) -> str:
        return (f"block-sign sketch: d = {self.d:,} -> m = {self.m:,} "
                f"(block L = {self.plan.block_size:,}), signs {self.sign_bytes()/1e9:.2f} GB "
                f"int8 on {self.device}")

    # -- construction ----------------------------------------------------------
    def _make_signs(self, tensor_index: int, n: int):
        """Deterministic +-1 vector for one tensor.

        Seeded per tensor so the sketch is reproducible across processes and machines,
        and so a tensor's signs do not depend on how many tensors precede it.
        """
        torch = self.torch
        gen = torch.Generator(device=self.device)
        gen.manual_seed((self.seed * 1_000_003 + tensor_index) % (2**63 - 1))
        out = torch.empty(n, dtype=torch.int8, device=self.device)
        for lo in range(0, n, _SIGN_GEN_CHUNK):
            hi = min(lo + _SIGN_GEN_CHUNK, n)
            r = torch.rand(hi - lo, generator=gen, device=self.device)
            out[lo:hi] = torch.where(r < 0.5, -1, 1).to(torch.int8)
            del r
        return out

    # -- projection ------------------------------------------------------------
    def project_grads(self, params, *, out=None, scale: float = 1.0, accumulate: bool = False):
        """Sketch the .grad of each parameter into one m-dim float32 vector.

        Parameters whose grad is None contribute zeros (a legitimate outcome: a group with
        no reward variation produces no gradient at all).
        """
        torch = self.torch
        if out is None:
            out = torch.zeros(self.m, dtype=torch.float32, device=self.device)
        elif not accumulate:
            out.zero_()

        for i, p in enumerate(params):
            g = p.grad
            if g is None:
                continue
            self._project_one(g.reshape(-1), i, out, scale)
        return out

    def project_vector(self, flat, tensor_index_ranges=None, *, out=None):
        """Sketch an already-flattened d-vector. Used by the unit tests and by the
        exact-vs-sketched cross-check, where the vector comes from a buffer not a .grad."""
        torch = self.torch
        if out is None:
            out = torch.zeros(self.m, dtype=torch.float32, device=self.device)
        else:
            out.zero_()
        pos = 0
        for i, n in enumerate(self.numels):
            self._project_one(flat[pos:pos + n], i, out, 1.0)
            pos += n
        return out

    def _project_one(self, g_flat, i: int, out, scale: float) -> None:
        torch = self.torch
        L = self.plan.block_size
        nb = self.plan.n_blocks[i]
        off = self.plan.offsets[i]
        signs = self._signs[i]
        n = g_flat.numel()

        if g_flat.device != signs.device:
            g_flat = g_flat.to(signs.device, non_blocking=True)

        blocks_per_chunk = max(1, _PROJ_CHUNK_ELEMS // max(L, 1))
        for b0 in range(0, nb, blocks_per_chunk):
            b1 = min(b0 + blocks_per_chunk, nb)
            lo, hi = b0 * L, min(b1 * L, n)
            if lo >= hi:
                break
            seg = g_flat[lo:hi].to(torch.float32)
            seg = seg * signs[lo:hi].to(torch.float32)
            want = (b1 - b0) * L
            if seg.numel() < want:                       # zero-pad the tail block
                pad = torch.zeros(want - seg.numel(), dtype=torch.float32, device=seg.device)
                seg = torch.cat([seg, pad])
            contrib = seg.view(b1 - b0, L).sum(dim=1)
            if scale != 1.0:
                contrib = contrib * scale
            out[off + b0: off + b1] += contrib
            del seg, contrib


def recommended_m(alpha_abs: float, target_rel_error: float = 0.05,
                  safety: float = 2.0) -> int:
    """Sketch dimension needed so sketch error is `target_rel_error` of Lambda.

    Relative error of <Sx,Sy> as an estimate of <x,y> is ~ 1 / (|alpha| sqrt(m)); invert,
    apply a safety factor, round up to a power of two.
    """
    alpha_abs = max(abs(float(alpha_abs)), 1e-6)
    m = (safety / (alpha_abs * max(target_rel_error, 1e-6))) ** 2
    return int(2 ** math.ceil(math.log2(max(m, 1024.0))))


def sketch_rel_err(alpha_abs: float, m: int) -> float:
    """Relative error of the sketched Lambda: ~ 1 / (|alpha| sqrt(m))."""
    return 1.0 / (max(abs(float(alpha_abs)), 1e-9) * math.sqrt(max(m, 1)))


def cache_plan(alpha_abs: float, n_pool: int, K: int, budget_gb: float = 30.0,
               bytes_per: int = 2) -> dict:
    """What sketch dimension the cache budget affords, and the trade if it is not enough.

    Cache bytes = 2 (modalities) * n_pool * K * m * bytes_per. When the measured |alpha| is
    small, the m needed for a clean sketch can exceed any affordable cache — and the fix is
    not a bigger m alone but a smaller (n_pool, K), which buys m at the cost of the top of
    the n sweep and of bootstrap pool depth. This lays that trade out numerically instead
    of leaving it to be discovered after a 13 GB build.
    """
    def afford(np_, k_):
        return int(budget_gb * 1e9 / (2 * max(np_, 1) * max(k_, 1) * bytes_per))

    m_now = afford(n_pool, K)
    options = []
    for np_, k_ in ((n_pool, K), (n_pool // 2, K), (n_pool, K // 2), (n_pool // 2, K // 2)):
        if np_ < 8 or k_ < 4:
            continue
        m = afford(np_, k_)
        options.append({"n_pool": np_, "K": k_, "m_affordable": m,
                        "rel_err": sketch_rel_err(alpha_abs, m),
                        "max_n": np_, "cache_gb": budget_gb})
    return {
        "alpha_abs": float(alpha_abs),
        "m_for_5pct": recommended_m(alpha_abs, 0.05),
        "m_affordable_now": m_now,
        "rel_err_now": sketch_rel_err(alpha_abs, m_now),
        "options": options,
    }


def describe_cache_plan(plan: dict) -> str:
    lines = [
        f"measured |alpha| = {plan['alpha_abs']:.4f}",
        f"  m for 5% sketch error: {plan['m_for_5pct']:,}",
        f"  m affordable at the current (n_pool, K): {plan['m_affordable_now']:,} "
        f"-> {plan['rel_err_now']:.1%} error",
    ]
    if plan["rel_err_now"] > 0.10:
        lines.append("  budget does not reach 5%. Trades that buy m by shrinking the cache:")
        for o in plan["options"]:
            lines.append(f"    n_pool={o['n_pool']:>4} K={o['K']:>3} -> m={o['m_affordable']:>9,} "
                         f"({o['rel_err']:.1%} error, sweep n capped at {o['max_n']})")
        lines.append("  Or: keep the sweep as a relative measurement (how noise scales with "
                     "n and G) and quote the exact anchor for the absolute Lambda.")
    return "\n".join(lines)


def check_unbiased(sketch: "BlockSignSketch", n_trials: int = 8, cos: float = 0.05,
                   seed: int = 0) -> dict:
    """Empirically confirm E<Sx,Sy> = <x,y> and measure the realised relative error.

    Builds x, y with a prescribed cosine directly in the *sketched* space's source
    dimension, so this is a genuine end-to-end test of plan + signs + projection.
    """
    import torch

    g = torch.Generator(device=sketch.device)
    g.manual_seed(seed)
    rel_errs, ratios = [], []
    for t in range(n_trials):
        parts_x, parts_y, dots = [], [], 0.0
        for n in sketch.numels:
            x = torch.randn(n, generator=g, device=sketch.device)
            z = torch.randn(n, generator=g, device=sketch.device)
            y = cos * x + math.sqrt(max(1.0 - cos ** 2, 0.0)) * z
            parts_x.append(x)
            parts_y.append(y)
        x = torch.cat(parts_x)
        y = torch.cat(parts_y)
        true = float(x @ y)
        sx = sketch.project_vector(x)
        sy = sketch.project_vector(y)
        est = float(sx @ sy)
        ratios.append(est / true if true != 0 else float("nan"))
        rel_errs.append(abs(est - true) / (abs(true) + 1e-30))
        del x, y, sx, sy, parts_x, parts_y

    import statistics

    return {
        "m": sketch.m,
        "d": sketch.d,
        "target_cos": cos,
        "mean_ratio": statistics.fmean(ratios),
        "mean_rel_err": statistics.fmean(rel_errs),
        "predicted_rel_err": 1.0 / (cos * math.sqrt(sketch.m)),
    }


__all__ = ["SketchPlan", "plan_sketch", "BlockSignSketch", "recommended_m", "check_unbiased",
           "sketch_rel_err", "cache_plan", "describe_cache_plan"]