"""Lambda, alpha, and the two bias corrections that make them mean what they claim.

The estimands
-------------
    Lambda = <grad J_v, grad J_t>          alpha = Lambda / (||grad J_v|| ||grad J_t||)

both over the shared backbone block. The gradients on the right are *population*
gradients. What we can compute are noisy estimates g_hat, and plugging those in is not
innocent:

  Bias 1 — shared items.  The brief's Lambda_hat = <g_hat_v, g_hat_t> uses the same
  calibration items for both modalities. The per-item noise in the two channels is
  correlated (same situation drives both), so

      E<g_hat_v, g_hat_t> = <grad J_v, grad J_t> + (1/n) E<delta_v, delta_t>,

  a positive O(1/n) inflation. `crossfit` removes it by computing the two gradients on
  *disjoint* halves of the pool, then symmetrising:

      Lambda_cf = 1/2 ( <g_v^A, g_t^B> + <g_v^B, g_t^A> ),   A, B disjoint.

  Both are reported; `study.estimator` picks the headline. Their gap is itself the
  measurement of how big the bias is.

  Bias 2 — plug-in norms.  E||g_hat||^2 = ||grad J||^2 + E||noise||^2, and in a 3.09e9-dim
  space the noise term is enormous. So the plug-in cosine is biased *toward zero*, hard —
  this is the "cosine biased low" the estimator lemma warns about. The fix is the same
  independence trick applied to the norms:

      ||grad J_v||^2_hat = <g_v^A, g_v^B>          (unbiased: A and B independent)
      alpha_shc = Lambda_cf / sqrt( <g_v^A,g_v^B> <g_t^A,g_t^B> )

  `alpha_plugin` and `alpha_shc` are both reported. Expect alpha_plugin << alpha_shc; if
  they are close, the gradient estimates are unusually clean.

Everything is computed from the cache as a batched matrix product: a draw becomes a
coefficient row over the flat (item, sample) axis, and B draws become one (B, N*K) matrix
times the (N*K, m) cached gradient matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .cache import MODALITIES

_V_CHUNK = 512   # rows of the cached gradient matrix upcast to fp32 at a time


# ------------------------------------------------------------------ advantages


def advantages_torch(rewards, *, normalize_by_std: bool, eps: float, std_floor: float):
    """Vectorised GRPO advantage over a (n, G) reward block.

    Identical arithmetic to rewards.group_advantages, which sanity_checks asserts
    element-for-element. Degenerate groups (all rewards equal) give exactly zero, because
    the numerator is exactly zero — the epsilon never has to rescue anything.
    """
    import torch

    G = rewards.shape[-1]
    mean = rewards.mean(dim=-1, keepdim=True)
    centred = rewards - mean
    if not normalize_by_std:
        return centred
    if G < 2:
        return torch.zeros_like(centred)
    var = (centred * centred).sum(dim=-1, keepdim=True) / (G - 1)
    sd = var.clamp_min(0).sqrt()
    if std_floor > 0:
        sd = sd.clamp_min(std_floor)
    return centred / (sd + eps)


def coeffs_torch(adv, lengths, loss_agg: str):
    """(per-sample coefficient, per-draw scale) for a (n, G) block. See
    gradient.aggregate_coeffs — this is the same contract, batched."""
    import torch

    L = lengths.clamp_min(1).to(adv.dtype)
    n, G = adv.shape[-2], adv.shape[-1]
    if loss_agg == "seq_sum":
        return adv, 1.0 / max(n * G, 1)
    if loss_agg == "seq_mean":
        return adv / L, 1.0 / max(n * G, 1)
    if loss_agg == "batch_token_mean":
        return adv, 1.0 / float(L.sum().clamp_min(1))
    raise ValueError(f"unknown loss_agg {loss_agg!r}")


# ------------------------------------------------------------------ batched gradients


def gradients_from_coeffs(cache, modality: str, coef_rows):
    """(B, m) gradients from a (B, N*K) coefficient matrix.

    Chunked over the flat (item, sample) axis and upcast to fp32 per chunk: exact fp32
    arithmetic without ever materialising the 27 GB fp32 view of the cache.
    """
    import torch

    c = cache.mod[modality]
    NK = cache.N * cache.K
    V = c.vectors.reshape(NK, cache.m)
    S = c.scales.reshape(NK)
    coef_rows = coef_rows * S.unsqueeze(0)          # fold the fp16 dequant scale in

    out = torch.zeros(coef_rows.shape[0], cache.m, dtype=torch.float32, device=V.device)
    for lo in range(0, NK, _V_CHUNK):
        hi = min(lo + _V_CHUNK, NK)
        block = coef_rows[:, lo:hi]
        if not bool(block.any()):
            continue
        out += block @ V[lo:hi].float()
    return out


def scatter_coeffs(dest, row: int, item_idx, sample_idx, coeffs, K: int) -> None:
    """Accumulate a draw's per-sample coefficients into the flat coefficient row.

    scatter_add (not scatter) because bootstrap draws repeat (item, sample) pairs, and a
    repeated pair must contribute twice — that repetition is precisely the resampling
    noise we are trying to reproduce.
    """
    flat = (item_idx.unsqueeze(-1) * K + sample_idx).reshape(-1)
    dest[row].scatter_add_(0, flat, coeffs.reshape(-1).to(dest.dtype))


# ------------------------------------------------------------------ one measurement


@dataclass
class Draw:
    """Index bundle for one resample: two disjoint item halves, per-modality samples."""

    idx_a: object
    idx_b: object
    samp: dict          # modality -> {"a": (nA,G), "b": (nB,G)}


def make_draw(cache, n: int, G: int, gen, *, item_resample: str = "bootstrap",
              group_resample: str = "bootstrap") -> Draw:
    """Draw items and answers for one resample.

    The pool is split into two disjoint halves *first*, then items are drawn within each.
    That disjointness is what makes the cross-fit terms genuinely independent; drawing
    both halves from the whole pool would let the same item land in each and reintroduce
    the very correlation cross-fitting exists to remove.

    Items and answers are both drawn with replacement by default. For answers this is the
    group bootstrap: it reproduces infinite-population variance to first order (whereas
    drawing G of K without replacement understates it by a factor 1 - (G-1)/(K-1), ~2x at
    G=16, K=32). acid_test measures whether that first-order claim actually holds here.
    """
    import torch

    N, K = cache.N, cache.K
    perm = torch.randperm(N, generator=gen, device=cache.device)
    half = N // 2
    pool_a, pool_b = perm[:half], perm[half:]

    nA, nB = n // 2, n - n // 2
    if item_resample == "bootstrap":
        idx_a = pool_a[torch.randint(len(pool_a), (nA,), generator=gen, device=pool_a.device)]
        idx_b = pool_b[torch.randint(len(pool_b), (nB,), generator=gen, device=pool_b.device)]
    elif item_resample == "subsample":
        if nA > len(pool_a) or nB > len(pool_b):
            raise ValueError(f"n={n} needs {max(nA, nB)} items per pool half but the halves "
                             f"hold {len(pool_a)}/{len(pool_b)}; use item_resample=bootstrap "
                             f"or enlarge data.n_pool")
        idx_a = pool_a[torch.randperm(len(pool_a), generator=gen, device=pool_a.device)[:nA]]
        idx_b = pool_b[torch.randperm(len(pool_b), generator=gen, device=pool_b.device)[:nB]]
    else:
        raise ValueError(f"unknown item_resample {item_resample!r}")

    samp: dict = {}
    for mod in MODALITIES:
        entry = {}
        for tag, cnt in (("a", nA), ("b", nB)):
            if group_resample == "bootstrap":
                s = torch.randint(K, (cnt, G), generator=gen, device=idx_a.device)
            elif group_resample == "subsample_fpc":
                if G > K:
                    raise ValueError(f"G={G} > K={K}")
                s = torch.stack([
                    torch.randperm(K, generator=gen, device=idx_a.device)[:G] for _ in range(cnt)
                ]) if cnt else torch.zeros((0, G), dtype=torch.long, device=idx_a.device)
            else:
                raise ValueError(f"unknown group_resample {group_resample!r}")
            entry[tag] = s
        samp[mod] = entry
    return Draw(idx_a=idx_a, idx_b=idx_b, samp=samp)


def make_draw_fixed_items(cache, item_pos, G: int, gen, *,
                          group_resample: str = "bootstrap") -> Draw:
    """A draw over a FIXED item set, resampling only the answers.

    This is the acid test's comparison object. Holding the items fixed isolates exactly
    the quantity under test — whether bootstrapping a group of G from a pool of K
    reproduces the variance of drawing G genuinely fresh generations — with no item
    resampling noise mixed in to blur the comparison.
    """
    import torch

    K = cache.K
    idx = item_pos.to(cache.device)
    nA = len(idx) // 2
    idx_a, idx_b = idx[:nA], idx[nA:]

    samp: dict = {}
    for mod in MODALITIES:
        entry = {}
        for tag, sub in (("a", idx_a), ("b", idx_b)):
            cnt = int(sub.numel())
            if group_resample == "bootstrap":
                s = torch.randint(K, (cnt, G), generator=gen, device=cache.device)
            elif group_resample == "subsample_fpc":
                s = (torch.stack([torch.randperm(K, generator=gen, device=cache.device)[:G]
                                  for _ in range(cnt)])
                     if cnt else torch.zeros((0, G), dtype=torch.long, device=cache.device))
            else:
                raise ValueError(f"unknown group_resample {group_resample!r}")
            entry[tag] = s
        samp[mod] = entry
    return Draw(idx_a=idx_a, idx_b=idx_b, samp=samp)


def _rows_for(cache, draws: Sequence[Draw], modality: str, grpo: dict):
    """Build the (3B, N*K) coefficient matrix: rows [A, B, ALL] per draw."""
    import torch

    B = len(draws)
    NK = cache.N * cache.K
    rows = torch.zeros(3 * B, NK, dtype=torch.float32, device=cache.device)
    diag = {"degenerate": 0, "groups": 0, "reward_sum": 0.0, "reward_n": 0,
            "adv_abs_sum": 0.0, "adv_nonzero": 0, "parse_ok_sum": 0}

    for b, dr in enumerate(draws):
        halves = {"a": dr.idx_a, "b": dr.idx_b}
        all_items, all_samp = [], []
        for hi, tag in enumerate(("a", "b")):
            idx = halves[tag]
            s = dr.samp[modality][tag]
            if idx.numel() == 0:
                continue
            r, L = cache.meta_for(modality, idx, s)
            adv = advantages_torch(r, normalize_by_std=grpo["normalize_by_std"],
                                   eps=grpo["eps"], std_floor=grpo["std_floor"])
            co, sc = coeffs_torch(adv, L, grpo["loss_agg"])
            scatter_coeffs(rows, 3 * b + hi, idx, s, co * sc, cache.K)
            all_items.append(idx)
            all_samp.append(s)
            diag["degenerate"] += int((r == r[:, :1]).all(dim=1).sum())
            diag["groups"] += int(r.shape[0])
            diag["reward_sum"] += float(r.sum())
            diag["reward_n"] += int(r.numel())
            # How much signal each group actually carries. A group where every answer
            # scored the same has every advantage exactly zero and contributes no
            # gradient at all, so these are what explain a small ||g|| downstream.
            diag["adv_abs_sum"] += float(adv.abs().sum())
            diag["adv_nonzero"] += int((adv != 0).sum())
            diag["parse_ok_sum"] += int(
                cache.mod[modality].parsed_ok[idx.unsqueeze(-1), s].sum())

        # The "all items" row is recomputed rather than averaged from the halves: under
        # batch_token_mean the normaliser is the batch token total, so (g_A + g_B)/2 is
        # not the same estimator as g over the union.
        idx = torch.cat(all_items)
        s = torch.cat(all_samp)
        r, L = cache.meta_for(modality, idx, s)
        adv = advantages_torch(r, normalize_by_std=grpo["normalize_by_std"],
                               eps=grpo["eps"], std_floor=grpo["std_floor"])
        co, sc = coeffs_torch(adv, L, grpo["loss_agg"])
        scatter_coeffs(rows, 3 * b + 2, idx, s, co * sc, cache.K)

    return rows, diag


def estimate_batch(cache, draws: Sequence[Draw], grpo: dict) -> dict:
    """Evaluate Lambda/alpha for a batch of draws. Returns arrays of length len(draws)."""
    import torch

    B = len(draws)
    out_rows = {}
    diags = {}
    for mod in MODALITIES:
        rows, diag = _rows_for(cache, draws, mod, grpo)
        out_rows[mod] = gradients_from_coeffs(cache, mod, rows).reshape(B, 3, cache.m)
        diags[mod] = diag

    gv_a, gv_b, gv = out_rows["image"].unbind(dim=1)
    gt_a, gt_b, gt = out_rows["text"].unbind(dim=1)

    dot = lambda x, y: (x * y).sum(dim=-1)

    lam_paired = dot(gv, gt)
    lam_cf = 0.5 * (dot(gv_a, gt_b) + dot(gv_b, gt_a))

    nv = gv.norm(dim=-1)
    nt = gt.norm(dim=-1)
    alpha_plugin = lam_paired / (nv * nt).clamp_min(1e-30)

    vv = dot(gv_a, gv_b)          # unbiased for ||grad J_v||^2
    tt = dot(gt_a, gt_b)
    denom = (vv.clamp_min(0) * tt.clamp_min(0)).sqrt()
    alpha_shc = torch.where(denom > 0, lam_cf / denom.clamp_min(1e-30),
                            torch.full_like(lam_cf, float("nan")))

    to = lambda t: t.detach().float().cpu().numpy()
    res = {
        "lambda_paired": to(lam_paired),
        "lambda_crossfit": to(lam_cf),
        "alpha_plugin": to(alpha_plugin),
        "alpha_shc": to(alpha_shc),
        "grad_v_norm": to(nv),
        "grad_t_norm": to(nt),
        "sq_norm_v_unbiased": to(vv),
        "sq_norm_t_unbiased": to(tt),
    }
    for mod in MODALITIES:
        d = diags[mod]
        res[f"{mod}_degenerate_frac"] = d["degenerate"] / max(d["groups"], 1)
        res[f"{mod}_mean_reward"] = d["reward_sum"] / max(d["reward_n"], 1)
        res[f"{mod}_mean_abs_adv"] = d["adv_abs_sum"] / max(d["reward_n"], 1)
        res[f"{mod}_nonzero_adv_frac"] = d["adv_nonzero"] / max(d["reward_n"], 1)
        res[f"{mod}_parse_ok_frac"] = d["parse_ok_sum"] / max(d["reward_n"], 1)
    return res


def resample_cell(cache, n: int, G: int, grpo: dict, *, n_draws: int, seed: int,
                  item_resample: str = "bootstrap", group_resample: str = "bootstrap",
                  batch: int = 32) -> dict:
    """The sampling distribution of Lambda/alpha at one (n, G) cell."""
    import numpy as np
    import torch

    gen = torch.Generator(device=cache.device)
    gen.manual_seed(int(seed))

    chunks: list[dict] = []
    made = 0
    while made < n_draws:
        b = min(batch, n_draws - made)
        draws = [make_draw(cache, n, G, gen, item_resample=item_resample,
                           group_resample=group_resample) for _ in range(b)]
        chunks.append(estimate_batch(cache, draws, grpo))
        made += b

    out: dict = {}
    for k in chunks[0]:
        vals = [c[k] for c in chunks]
        if isinstance(vals[0], np.ndarray):
            out[k] = np.concatenate(vals)
        else:
            out[k] = float(np.mean(vals))
    out["n"] = n
    out["G"] = G
    out["n_draws"] = n_draws
    return out


# ------------------------------------------------------------------ exact path


def alignment_from_vectors(gv, gt) -> dict:
    """Lambda / alpha / norms from two full-length gradient vectors (the exact anchor)."""
    from .gradient import chunked_dot

    lam = chunked_dot(gv, gt)
    nv = math.sqrt(max(chunked_dot(gv, gv), 0.0))
    nt = math.sqrt(max(chunked_dot(gt, gt), 0.0))
    return {"lambda": lam, "alpha": lam / (nv * nt + 1e-30), "grad_v_norm": nv,
            "grad_t_norm": nt}


__all__ = ["advantages_torch", "coeffs_torch", "gradients_from_coeffs", "scatter_coeffs",
           "Draw", "make_draw", "make_draw_fixed_items", "estimate_batch", "resample_cell",
           "alignment_from_vectors"]