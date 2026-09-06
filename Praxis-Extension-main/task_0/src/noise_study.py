"""The point of Task 0: how noisy is Lambda_hat, and is that noise small enough?

Four things happen here.

1. `run_sweep` — the (n, G) grid. std(Lambda_hat), the bootstrap CI, and sign agreement
   for every cell.
2. `run_exact_anchor` — full-backbone, unsketched Lambda at the anchor cell. Anchors the
   absolute number, measures the true |alpha| (which is what justifies the sketch
   dimension), and validates the sketch by projecting the *same* exact vectors.
3. `run_acid_test` — does group-bootstrap variance match genuinely fresh generations?
   Items are held fixed on both sides so only answer sampling varies.
4. `verdict` — the go/no-go against the acceptance criteria.

On the two intervals, which are not the same thing
--------------------------------------------------
`ci_lo/ci_hi` are the 2.5/97.5 percentiles of the *sampling distribution* of Lambda_hat.
That is the interval one measurement at this (n, G) lands in, and its half-width is
~1.96 std — it does not shrink as we take more draws. This is what the acceptance
criterion is judged on, because the criterion is about whether a single measurement at a
checkpoint pins Lambda down.

`se_of_mean` and `mean_ci_*` describe the *mean over draws* and do shrink like 1/sqrt(R).
Judging acceptance on that would be gameable — take more draws, pass the test, learn
nothing about a single measurement. Both are reported; only the first is the gate.
"""

from __future__ import annotations

import json
import math
import os
from typing import Sequence

import numpy as np

from .alignment import (alignment_from_vectors, estimate_batch, make_draw_fixed_items,
                        resample_cell)
from .cache import MODALITIES
from .config import both_conventions, rel, resolve_grpo, stamp

_METRICS = ("lambda_paired", "lambda_crossfit", "alpha_plugin", "alpha_shc")

# The within-modality positive control. <g_v^A, g_v^B> over disjoint item halves is an
# unbiased estimate of ||grad J_v||^2, so its true value is strictly positive. If the
# apparatus cannot even resolve THAT sign at a given (n, G), a null result on the
# cross-modal Lambda says nothing about alignment — it just says the estimator is blind.
# Separating those two cases is the whole reason this is tracked.
_CONTROLS = ("sq_norm_v_unbiased", "sq_norm_t_unbiased")


def variance_decomposition(rows: list[dict], key: str, convention: str) -> dict:
    """Split the noise into item sampling and answer sampling, and say which knob to turn.

    To leading order the per-item gradient error has an item component and an answer
    component that averages over the group, so

        Var(Lambda_hat) ~ ( A + B / G ) / n.

    Fitting A and B over the grid gives the crossover G* = B / A, the group size at which
    answer noise stops dominating. Above it, more items; below it, more samples per item.
    """
    sel = [r for r in rows if r["convention"] == convention
           and np.isfinite(r.get(f"{key}.std", float("nan")))]
    if len(sel) < 3:
        return {"ok": False, "reason": "not enough cells"}

    X = np.array([[1.0 / r["n"], 1.0 / (r["n"] * r["G"])] for r in sel])
    y = np.array([r[f"{key}.std"] ** 2 for r in sel])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    A, B = (float(c) for c in coef)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    G_star = B / A if A > 0 else float("inf")
    Gs = sorted({r["G"] for r in sel})
    if A <= 0:
        advice = ("item-sampling variance fits as non-positive — answer sampling dominates "
                  "everywhere in this grid; raise G.")
    elif G_star > max(Gs):
        advice = (f"answer sampling dominates up to G={max(Gs)} (crossover G* = {G_star:.0f}); "
                  f"raising G buys more than raising n.")
    elif G_star < min(Gs):
        advice = (f"item sampling dominates from G={min(Gs)} up (crossover G* = {G_star:.3g}); "
                  f"raising n buys more than raising G.")
    else:
        advice = (f"crossover at G* = {G_star:.3g}: below it raise G, above it raise n.")
    return {"ok": True, "A_item": A, "B_answer": B, "G_star": G_star, "r2": r2,
            "advice": advice, "n_cells": len(sel)}


# ------------------------------------------------------------------ summaries


def summarize(values, *, sign_ref: float | None = None) -> dict:
    """Distribution summary for one metric across resamples."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        empty = {k: float("nan") for k in
                 ("mean", "std", "median", "ci_lo", "ci_hi", "ci_halfwidth",
                  "rel_halfwidth", "frac_positive", "sign_agreement", "se_of_mean")}
        empty.update({"n": 0, "ci_excludes_zero": False})
        return empty

    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    med = float(np.median(x))
    lo, hi = (float(v) for v in np.percentile(x, [2.5, 97.5]))
    ref = np.sign(med) if sign_ref is None else np.sign(sign_ref)
    if ref == 0:
        ref = 1.0
    half = 0.5 * (hi - lo)
    return {
        "n": int(x.size),
        "mean": mean,
        "std": std,
        "median": med,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_halfwidth": half,
        "rel_halfwidth": half / abs(med) if med != 0 else float("inf"),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "frac_positive": float((x > 0).mean()),
        "sign_agreement": float((np.sign(x) == ref).mean()),
        "se_of_mean": std / math.sqrt(x.size) if x.size > 1 else float("nan"),
    }


# ------------------------------------------------------------------ the sweep


def run_sweep(cache, cfg, *, conventions: dict | None = None, verbose: bool = True) -> dict:
    """Evaluate every (n, G) cell under every requested GRPO convention."""
    conventions = conventions or {cfg.grpo.convention: resolve_grpo(cfg)}
    R = int(cfg.study.n_resamples)
    B = int(cfg.study.n_boot)
    n_draws = max(R, B)

    rows: list[dict] = []
    raw: dict = {}
    for cname, grpo in conventions.items():
        for n in cfg.study.n_values:
            for G in cfg.study.G_values:
                seed = int(cfg.study.seed) * 1_000_003 + n * 97 + G
                res = resample_cell(
                    cache, n, G, grpo, n_draws=n_draws, seed=seed,
                    item_resample=cfg.study.item_resample,
                    group_resample=cfg.study.group_resample,
                )
                raw[f"{cname}|n{n}|G{G}"] = {k: np.asarray(v).tolist()
                                             for k, v in res.items()
                                             if isinstance(v, np.ndarray)}
                row = {"convention": cname, "n": n, "G": G, "n_draws": n_draws,
                       "R_spec": R}
                for metric in _METRICS:
                    full = summarize(res[metric])
                    spec = summarize(res[metric][:R], sign_ref=full["median"])
                    for k, v in full.items():
                        row[f"{metric}.{k}"] = v
                    row[f"{metric}.std_R{R}"] = spec["std"]
                    row[f"{metric}.sign_agreement_R{R}"] = spec["sign_agreement"]
                # Controls are summarised against the known-positive truth, not the
                # observed median, so "sign_agreement" here reads as "fraction correct".
                for ctrl in _CONTROLS:
                    s = summarize(res[ctrl], sign_ref=1.0)
                    row[f"{ctrl}.frac_positive"] = s["frac_positive"]
                    row[f"{ctrl}.median"] = s["median"]
                    row[f"{ctrl}.rel_halfwidth"] = s["rel_halfwidth"]
                for mod in MODALITIES:
                    for stat in ("degenerate_frac", "mean_reward", "mean_abs_adv",
                                 "nonzero_adv_frac", "parse_ok_frac"):
                        row[f"{mod}_{stat}"] = res[f"{mod}_{stat}"]
                row["grad_v_norm.mean"] = float(np.mean(res["grad_v_norm"]))
                row["grad_t_norm.mean"] = float(np.mean(res["grad_t_norm"]))
                rows.append(row)
                if verbose:
                    est = cfg.study.estimator
                    key = f"lambda_{est}"
                    print(f"  [{cname}] n={n:>3} G={G:>2}  "
                          f"{key}={row[key + '.median']:+.4e}  "
                          f"std={row[key + '.std']:.3e}  "
                          f"relCI={row[key + '.rel_halfwidth']:.2f}  "
                          f"sign={row[key + '.sign_agreement']:.2%}  "
                          f"alpha_shc={row['alpha_shc.median']:+.4f}", flush=True)
    return {"rows": rows, "raw": raw}


# ------------------------------------------------------------------ exact anchor


def _pick_accum_device(cfg, d: int) -> str:
    """Two fp32 buffers of d floats must fit. On a 40 GB card they will not sit alongside
    the model and its gradients, so CPU (high-RAM) is the usual answer."""
    import torch

    want = cfg.get_path("exact_anchor.accum_device", "auto")
    if want != "auto":
        return want
    need = 2 * d * 4 * 1.15
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        if free_b > need + 4e9:
            return "cuda"
    return "cpu"


def run_exact_anchor(cfg, extractor, sketch, items, *, grpo: dict, verbose: bool = True) -> dict:
    """Full-backbone Lambda at the anchor, plus a sketch-fidelity check.

    The sketch check projects the *same* exact gradient vectors, so the difference it
    reports is pure sketch error with no sampling noise mixed in.
    """
    import torch

    from .gradient import FullSink

    # Default to the study anchor so the exact number and the swept cell are comparable;
    # override only to shrink the exact run, which is the expensive one.
    n = int(cfg.get_path("exact_anchor.n") or cfg.study.anchor.n)
    G = int(cfg.get_path("exact_anchor.G") or cfg.study.anchor.G)
    reps = int(cfg.get_path("exact_anchor.n_reps", 3))
    d = extractor.backbone.d
    dev = _pick_accum_device(cfg, d)
    if verbose:
        print(f"exact anchor: n={n} G={G} reps={reps}, fp32 accumulators "
              f"({2*d*4/1e9:.1f} GB) on {dev}")

    rng = np.random.default_rng(int(cfg.study.seed) + 991)
    out: list[dict] = []
    for rep in range(reps):
        sub = [items[i] for i in rng.choice(len(items), size=n, replace=False)]
        rec: dict = {"rep": rep}
        sinks = {}
        for mod in MODALITIES:
            sink = FullSink(extractor.backbone.numels, device=dev)
            stats = extractor.modality_gradient(sub, mod, G, grpo=grpo, sink=sink,
                                                seed=int(cfg.study.seed) * 13 + rep * 7)
            sinks[mod] = sink
            rec[f"{mod}_mean_reward"] = stats["mean_reward"]
            rec[f"{mod}_degenerate_groups"] = stats["degenerate_groups"]
            rec[f"{mod}_parse_fail"] = stats["parse_fail"]
            if verbose:
                print(f"  rep {rep} {mod}: reward={stats['mean_reward']:.3f} "
                      f"degenerate={stats['degenerate_groups']}/{n} "
                      f"backwards={stats['backwards']}", flush=True)

        gv, gt = sinks["image"].vector(), sinks["text"].vector()
        rec.update({f"exact_{k}": v for k, v in alignment_from_vectors(gv, gt).items()})

        sv = sketch.project_vector(gv)
        st = sketch.project_vector(gt)
        lam_s = float((sv * st).sum())
        rec["sketched_lambda"] = lam_s
        rec["sketched_alpha"] = lam_s / (float(sv.norm()) * float(st.norm()) + 1e-30)
        rec["sketch_rel_err"] = abs(lam_s - rec["exact_lambda"]) / (abs(rec["exact_lambda"]) + 1e-30)
        rec["sketch_predicted_rel_err"] = 1.0 / (abs(rec["exact_alpha"]) * math.sqrt(sketch.m) + 1e-30)
        out.append(rec)
        if verbose:
            print(f"  rep {rep}: exact Lambda={rec['exact_lambda']:+.4e} "
                  f"alpha={rec['exact_alpha']:+.4f} | sketched={lam_s:+.4e} "
                  f"(rel err {rec['sketch_rel_err']:.1%}, predicted "
                  f"{rec['sketch_predicted_rel_err']:.1%})", flush=True)
        del sinks, gv, gt, sv, st
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    alphas = [abs(r["exact_alpha"]) for r in out]
    from .sketch import recommended_m

    return {
        "n": n, "G": G, "reps": reps, "records": out,
        "mean_abs_alpha": float(np.mean(alphas)),
        "mean_exact_lambda": float(np.mean([r["exact_lambda"] for r in out])),
        "mean_sketch_rel_err": float(np.mean([r["sketch_rel_err"] for r in out])),
        "recommended_m_for_5pct": int(recommended_m(float(np.mean(alphas)), 0.05)),
        "sketch_m": int(sketch.m),
    }


# ------------------------------------------------------------------ acid test


def _var_ratio_ci(s2_fresh: float, df_fresh: int, s2_boot: float, df_boot: int) -> tuple:
    """95% CI for the variance ratio sigma^2_fresh / sigma^2_boot."""
    F = s2_fresh / max(s2_boot, 1e-300)
    try:
        from scipy.stats import f as fdist

        lo = F / fdist.ppf(0.975, df_fresh, df_boot)
        hi = F / fdist.ppf(0.025, df_fresh, df_boot)
    except Exception:
        # chi2 approximation for large df_boot: F_{q,df,inf} = chi2_{q,df} / df
        table = {4: (11.14, 0.484), 9: (19.02, 2.700), 14: (26.12, 5.629),
                 19: (32.85, 8.907), 29: (45.72, 16.05), 49: (70.22, 31.55)}
        key = min(table, key=lambda k: abs(k - df_fresh))
        hi_chi, lo_chi = table[key]
        lo = F * key / hi_chi
        hi = F * key / lo_chi
    return float(F), float(lo), float(hi)


def run_acid_test(cfg, extractor, sketch, cache, items, *, grpo: dict,
                  verbose: bool = True) -> dict:
    """Does group-bootstrap variance equal fresh-generation variance?

    Same items on both sides, same G, same sketch, same convention. Only the source of the
    G answers differs: freshly generated vs. drawn with replacement from the cached pool
    of K. If the bootstrap is reproducing the right sampling noise, the two standard
    deviations must agree.
    """
    import torch

    from .gradient import SketchSink

    n = int(cfg.acid_test.n)
    G = int(cfg.acid_test.G)
    R = int(cfg.acid_test.n_fresh)

    rng = np.random.default_rng(int(cfg.study.seed) + 4242)
    pos = np.sort(rng.choice(cache.N, size=n, replace=False))
    sub = [items[int(i)] for i in pos]
    if verbose:
        print(f"acid test: {R} fresh re-generations vs cached bootstrap, "
              f"n={n} G={G}, items held fixed")

    fresh: list[float] = []
    fresh_diag: list[dict] = []
    for rep in range(R):
        vecs = {}
        diag = {}
        for mod in MODALITIES:
            sink = SketchSink(sketch)
            st = extractor.modality_gradient(sub, mod, G, grpo=grpo, sink=sink,
                                             seed=100_003 * (rep + 1) + (0 if mod == "image" else 5))
            vecs[mod] = sink.vector().clone()
            diag[f"{mod}_reward"] = st["mean_reward"]
            diag[f"{mod}_degenerate"] = st["degenerate_groups"]
        lam = float((vecs["image"] * vecs["text"]).sum())
        fresh.append(lam)
        fresh_diag.append(diag)
        if verbose:
            print(f"  fresh rep {rep+1}/{R}: Lambda={lam:+.4e}", flush=True)
        del vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Matched bootstrap: identical items, answers resampled from the cache.
    gen = torch.Generator(device=cache.device)
    gen.manual_seed(int(cfg.study.seed) + 777)
    item_pos = torch.as_tensor(pos, dtype=torch.long, device=cache.device)
    n_boot = max(int(cfg.study.n_boot), 200)
    boot: list[float] = []
    made = 0
    while made < n_boot:
        b = min(64, n_boot - made)
        draws = [make_draw_fixed_items(cache, item_pos, G, gen,
                                       group_resample=cfg.study.group_resample)
                 for _ in range(b)]
        boot.extend(estimate_batch(cache, draws, grpo)["lambda_paired"].tolist())
        made += b

    f = np.asarray(fresh, dtype=float)
    b_ = np.asarray(boot, dtype=float)
    F, lo, hi = _var_ratio_ci(float(f.var(ddof=1)), len(f) - 1,
                              float(b_.var(ddof=1)), len(b_) - 1)
    passed = bool(lo <= 1.0 <= hi)

    # A mean shift is a different failure from a variance mismatch; report it separately.
    mean_z = (float(f.mean()) - float(b_.mean())) / (f.std(ddof=1) / math.sqrt(len(f)) + 1e-300)

    out = {
        "n": n, "G": G, "n_fresh": len(f), "n_boot": len(b_),
        "fresh_lambda": f.tolist(),
        "fresh_mean": float(f.mean()), "fresh_std": float(f.std(ddof=1)),
        "boot_mean": float(b_.mean()), "boot_std": float(b_.std(ddof=1)),
        "var_ratio": F, "var_ratio_ci": [lo, hi], "variance_agrees": passed,
        "mean_shift_z": float(mean_z),
        "finite_population_factor": 1.0 - 1.0 / float(cache.K),
        "fresh_diagnostics": fresh_diag,
    }
    if verbose:
        print(f"  fresh std = {out['fresh_std']:.4e}   bootstrap std = {out['boot_std']:.4e}")
        print(f"  variance ratio = {F:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  -> "
              f"{'AGREE' if passed else 'DISAGREE'}")
        print(f"  mean shift z = {mean_z:+.2f} (|z| > 2 suggests a bias, not just variance)")
        if not passed:
            print("  ACTION: set study.group_resample=subsample_fpc and re-run the sweep; "
                  "the bootstrap is not reproducing answer-sampling noise here.")
    if not passed and cfg.get_path("acid_test.fail_action") == "raise":
        raise RuntimeError(f"acid test failed: variance ratio CI {[lo, hi]} excludes 1.0")
    return out


# ------------------------------------------------------------------ verdict


def verdict(sweep_rows: list[dict], cfg, *, exact: dict | None = None,
            acid: dict | None = None) -> dict:
    """Go / no-go against the acceptance criteria, plus the best passing (n, G)."""
    est = cfg.study.estimator
    key = f"lambda_{est}"
    conv = cfg.grpo.convention
    sign_min = float(cfg.acceptance.sign_stability_min)
    rel_max = float(cfg.acceptance.ci_halfwidth_frac_max)

    cells = []
    for r in sweep_rows:
        if r["convention"] != conv:
            continue
        ok_sign = r[f"{key}.sign_agreement"] >= sign_min
        ok_ci = bool(r[f"{key}.ci_excludes_zero"])
        ok_rel = r[f"{key}.rel_halfwidth"] < rel_max
        cells.append({
            "n": r["n"], "G": r["G"], "cost": r["n"] * r["G"],
            "lambda_median": r[f"{key}.median"],
            "lambda_std": r[f"{key}.std"],
            "rel_halfwidth": r[f"{key}.rel_halfwidth"],
            "sign_agreement": r[f"{key}.sign_agreement"],
            "alpha_shc_median": r["alpha_shc.median"],
            "control_v_frac_positive": r.get("sq_norm_v_unbiased.frac_positive"),
            "control_t_frac_positive": r.get("sq_norm_t_unbiased.frac_positive"),
            "passes": bool(ok_sign and ok_ci and ok_rel),
            "fails": [k for k, ok in (("sign_stability", ok_sign),
                                      ("ci_excludes_zero", ok_ci),
                                      ("rel_halfwidth", ok_rel)) if not ok],
        })

    passing = [c for c in cells if c["passes"]]
    passing.sort(key=lambda c: c["cost"])
    anchor = next((c for c in cells if c["n"] == cfg.study.anchor.n
                   and c["G"] == cfg.study.anchor.G), None)

    status = "PASS" if passing else "FAIL"
    if passing and acid is not None and not acid.get("variance_agrees", True):
        status = "PASS (QUALIFIED)"

    lines = [f"TASK 0: {status}"]
    if passing:
        c = passing[0]
        lines.append(f"  cheapest passing setting: n={c['n']}, G={c['G']} "
                     f"({c['cost']} generations per gradient)")
        lines.append(f"    Lambda = {c['lambda_median']:+.4e}  std = {c['lambda_std']:.3e}  "
                     f"CI half-width = {c['rel_halfwidth']:.1%} of |Lambda|")
        lines.append(f"    sign agreement = {c['sign_agreement']:.1%}  "
                     f"alpha (split-half) = {c['alpha_shc_median']:+.4f}")
    else:
        lines.append("  no (n, G) cell met all three criteria.")
        if cells:
            best = min(cells, key=lambda c: c["rel_halfwidth"])
            lines.append(f"  closest cell: n={best['n']} G={best['G']} "
                         f"rel CI={best['rel_halfwidth']:.1%} "
                         f"sign={best['sign_agreement']:.1%} "
                         f"(failed: {', '.join(best['fails'])})")

    # The interpretive fork. A failed Lambda means two completely different things
    # depending on whether the apparatus can resolve a quantity it KNOWS is positive.
    ctrl_cells = [c for c in cells if c.get("control_v_frac_positive") is not None]
    if ctrl_cells:
        best_ctrl = max(ctrl_cells, key=lambda c: min(c["control_v_frac_positive"],
                                                      c["control_t_frac_positive"]))
        cv, ct = best_ctrl["control_v_frac_positive"], best_ctrl["control_t_frac_positive"]
        lines.append(f"  positive control (within-modality <g^A, g^B>, true sign is +): "
                     f"best at n={best_ctrl['n']} G={best_ctrl['G']} — "
                     f"image {cv:.1%}, text {ct:.1%} positive")
        control_ok = min(cv, ct) >= sign_min
        if not passing and control_ok:
            lines.append("  READ: the estimator CAN resolve a known-positive quantity at "
                         "this cost, but cannot resolve Lambda. That points to alignment "
                         "genuinely near zero, not to a blind instrument. More samples "
                         "will not rescue a Lambda that is actually ~0.")
        elif not passing and not control_ok:
            lines.append("  READ: the estimator cannot resolve even the known-positive "
                         "control, so the Lambda result is uninformative — this is a noise "
                         "floor problem, not evidence about alignment. Raise G/n per the "
                         "variance decomposition below before concluding anything.")
        elif passing and not control_ok:
            lines.append("  WARNING: Lambda passes but the within-modality control does "
                         "not. That is internally inconsistent — the control is the easier "
                         "measurement. Suspect a bug before believing the Lambda.")
    vd = variance_decomposition(sweep_rows, key, conv)
    if vd.get("ok"):
        lines.append(f"  variance split: Var ~ (A + B/G)/n with A={vd['A_item']:.3e} "
                     f"B={vd['B_answer']:.3e} (R^2={vd['r2']:.2f}) — {vd['advice']}")

    if exact:
        lines.append(f"  exact full-backbone |alpha| = {exact['mean_abs_alpha']:.4f}; "
                     f"sketch m = {exact['sketch_m']:,} gives {exact['mean_sketch_rel_err']:.1%} "
                     f"relative error (m = {exact['recommended_m_for_5pct']:,} would give 5%)")
    if acid:
        lines.append(f"  acid test: fresh std {acid['fresh_std']:.3e} vs bootstrap "
                     f"{acid['boot_std']:.3e}, ratio CI "
                     f"[{acid['var_ratio_ci'][0]:.2f}, {acid['var_ratio_ci'][1]:.2f}] "
                     f"-> {'agree' if acid['variance_agrees'] else 'DISAGREE'}")

    return {"status": status, "cells": cells, "passing": passing, "anchor": anchor,
            "report": "\n".join(lines), "variance_decomposition": vd,
            "criteria": {"sign_stability_min": sign_min, "ci_halfwidth_frac_max": rel_max,
                         "estimator": est, "convention": conv}}


# ------------------------------------------------------------------ outputs


def write_results(cfg, sweep: dict, *, exact=None, acid=None, verdict_=None,
                  outdir: str | None = None) -> dict:
    """Write the CSVs, the JSON bundle, and the figures."""
    import csv

    outdir = outdir or rel("results")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    rows = sweep["rows"]
    est = cfg.study.estimator
    key = f"lambda_{est}"

    noise_path = os.path.join(outdir, "noise_floor.csv")
    with open(noise_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sign_path = os.path.join(outdir, "sign_stability.csv")
    with open(sign_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["convention", "n", "G", "estimator", "sign_agreement", "frac_positive",
                    "sign_agreement_R", "ci_excludes_zero", "lambda_median", "lambda_std",
                    "rel_halfwidth", "n_draws"])
        R = int(cfg.study.n_resamples)
        for r in rows:
            w.writerow([r["convention"], r["n"], r["G"], est,
                        f"{r[key + '.sign_agreement']:.4f}",
                        f"{r[key + '.frac_positive']:.4f}",
                        f"{r[key + f'.sign_agreement_R{R}']:.4f}",
                        int(bool(r[key + ".ci_excludes_zero"])),
                        f"{r[key + '.median']:.6e}", f"{r[key + '.std']:.6e}",
                        f"{r[key + '.rel_halfwidth']:.4f}", r["n_draws"]])

    bundle = {"stamp": stamp(cfg), "sweep_rows": rows, "exact_anchor": exact,
              "acid_test": acid, "verdict": verdict_}
    json_path = os.path.join(outdir, "task0_results.json")
    with open(json_path, "w") as f:
        json.dump(bundle, f, indent=1, default=str)

    # Every individual draw, not just the summaries. Without this you cannot re-plot,
    # take different quantiles, check for outliers or non-normality, or run a test nobody
    # thought of yet -- you would have to rebuild the cache. It is a few MB.
    raw_path = os.path.join(outdir, "raw_draws.npz")
    flat = {f"{cell}|{metric}": np.asarray(vals, dtype=np.float32)
            for cell, metrics in sweep.get("raw", {}).items()
            for metric, vals in metrics.items()}
    if flat:
        np.savez_compressed(raw_path, **flat)

    figs = make_figures(cfg, sweep, outdir=outdir)
    out = {"noise_floor_csv": noise_path, "sign_stability_csv": sign_path,
           "results_json": json_path, "figures": figs}
    if flat:
        out["raw_draws_npz"] = raw_path
    return out


def make_figures(cfg, sweep: dict, *, outdir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in sweep["rows"] if r["convention"] == cfg.grpo.convention]
    if not rows:
        return []
    est = cfg.study.estimator
    key = f"lambda_{est}"
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    Gs = sorted({r["G"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    paths: list[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for G in Gs:
        sel = sorted([r for r in rows if r["G"] == G], key=lambda r: r["n"])
        x = [r["n"] for r in sel]
        axes[0].plot(x, [r[key + ".std"] for r in sel], "o-", label=f"G={G}")
        axes[1].plot(x, [r[key + ".rel_halfwidth"] for r in sel], "o-", label=f"G={G}")
        axes[2].plot(x, [r[key + ".sign_agreement"] for r in sel], "o-", label=f"G={G}")

    axes[0].set(xscale="log", yscale="log", xlabel="calibration items n",
                ylabel=r"std($\hat\Lambda$)", title="noise floor")
    # 1/sqrt(n) reference through the first point of the largest G
    ref = sorted([r for r in rows if r["G"] == Gs[-1]], key=lambda r: r["n"])
    if ref and np.isfinite(ref[0][key + ".std"]):
        c = ref[0][key + ".std"] * math.sqrt(ref[0]["n"])
        axes[0].plot(ns, [c / math.sqrt(n) for n in ns], "k--", alpha=0.4,
                     label=r"$\propto n^{-1/2}$")
    axes[0].legend(fontsize=8)

    axes[1].axhline(float(cfg.acceptance.ci_halfwidth_frac_max), color="r", ls="--",
                    alpha=0.6, label="acceptance")
    axes[1].set(xscale="log", yscale="log", xlabel="calibration items n",
                ylabel=r"CI half-width / $|\hat\Lambda|$",
                title="relative precision of one measurement")
    axes[1].legend(fontsize=8)

    axes[2].axhline(float(cfg.acceptance.sign_stability_min), color="r", ls="--", alpha=0.6,
                    label="acceptance")
    axes[2].set(xscale="log", ylim=(0.4, 1.02), xlabel="calibration items n",
                ylabel=r"fraction agreeing on sign($\hat\Lambda$)", title="sign stability")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Task 0 noise floor — {cfg.model.name}, estimator={est}, "
                 f"convention={cfg.grpo.convention}", fontsize=11)
    fig.tight_layout()
    p = os.path.join(figdir, "noise_floor.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Distribution at the anchor, both estimators and both cosines.
    akey = f"{cfg.grpo.convention}|n{cfg.study.anchor.n}|G{cfg.study.anchor.G}"
    raw = sweep.get("raw", {}).get(akey)
    if raw:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
        for ax, k, title in (
            (axes[0], "lambda_paired", r"$\hat\Lambda$ paired (same items)"),
            (axes[1], "lambda_crossfit", r"$\hat\Lambda$ cross-fit (disjoint halves)"),
        ):
            v = np.asarray(raw[k], dtype=float)
            ax.hist(v, bins=60, color="#4676d0", alpha=0.85)
            ax.axvline(0, color="k", lw=1)
            ax.axvline(float(np.median(v)), color="crimson", lw=1.5)
            lo, hi = np.percentile(v, [2.5, 97.5])
            ax.axvspan(lo, hi, color="crimson", alpha=0.12)
            ax.set(title=title, xlabel=r"$\hat\Lambda$")
        for k, c in (("alpha_plugin", "#888888"), ("alpha_shc", "#d06a46")):
            v = np.asarray(raw[k], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                axes[2].hist(v, bins=60, alpha=0.65, color=c,
                             label=("plug-in (biased low)" if k == "alpha_plugin"
                                    else "split-half (debiased)"))
        axes[2].axvline(0, color="k", lw=1)
        axes[2].set(title=r"cosine $\hat\alpha$", xlabel=r"$\hat\alpha$")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.suptitle(f"anchor cell n={cfg.study.anchor.n}, G={cfg.study.anchor.G}", fontsize=11)
        fig.tight_layout()
        p = os.path.join(figdir, "anchor_distribution.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    return paths


__all__ = ["summarize", "run_sweep", "run_exact_anchor", "run_acid_test", "verdict",
           "write_results", "make_figures"]
