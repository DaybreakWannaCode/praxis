"""Task 0 entry point.

    python -m src.run_task0 all --smoke        # end-to-end in ~15 min, tiny everything
    python -m src.run_task0 all                # the real study
    python -m src.run_task0 sanity --fp32      # correctness gate, strictest form
    python -m src.run_task0 sweep              # re-run the study from an existing cache

Stage order and why
-------------------
  sanity   correctness gate. Nothing downstream means anything until this passes.
  data     download VIVA, verify images decode, build the paired calibration pool.
  anchor   exact full-backbone Lambda. Runs BEFORE the cache because it measures the true
           |alpha|, which is what determines whether the sketch dimension m is adequate
           (--auto-m acts on that measurement).
  cache    the one expensive pass: K generations and K sketched gradients per
           (item, modality). Resumable.
  acid     bootstrap-vs-fresh variance test on the cache.
  sweep    the (n, G) grid, CIs, sign stability, figures, verdict. Pure arithmetic.

Only `sweep` runs without the model, so it is the cheap one to iterate on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):  # allow `python src/run_task0.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"

from .config import load_config, project_root, rel, resolve_grpo, both_conventions

SMOKE = [
    "data.n_pool=24", "sampling.K=8", "sampling.gen_batch=8",
    "study.n_values=[8,16]", "study.G_values=[2,4]",
    "study.anchor={\"n\":16,\"G\":4}", "study.n_resamples=10", "study.n_boot=200",
    "exact_anchor.n=8", "exact_anchor.G=4", "exact_anchor.n_reps=1",
    "acid_test.n=8", "acid_test.G=4", "acid_test.n_fresh=3",
    "policy.max_new_tokens=64", "sketch.m_target=16384",
]


def _banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74, flush=True)


def setup_model(cfg, device: str):
    """Load the frozen checkpoint, isolate the backbone, build the sketch and extractor."""
    from .gradient import PolicyGradientExtractor
    from .load_model import cuda_report, load_model
    from .sketch import BlockSignSketch

    print(cuda_report())
    model, proc, spec = load_model(cfg, device=device)
    print(spec.summary())

    sk_device = cfg.get_path("sketch.device", "cuda")
    if sk_device == "cuda" and not _cuda_ok():
        sk_device = "cpu"
    sketch = BlockSignSketch(spec.numels, m_target=int(cfg.sketch.m_target),
                             seed=int(cfg.sketch.seed), device=sk_device)
    print(sketch.describe())
    print(cuda_report())

    ex = PolicyGradientExtractor(model, proc, spec, cfg, device=device)
    print(f"logits_to_keep support: {ex._ltk_kw or 'no (full logits will be sliced)'}")
    print(f"eos ids: {ex._eos_ids}   sampling: T={ex.temperature} "
          f"top_p={cfg.get_path('policy.top_p')} top_k={cfg.get_path('policy.top_k')}")
    return model, proc, spec, sketch, ex


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------------------------------------------ stages


def stage_sanity(cfg, args) -> dict:
    _banner("STAGE: sanity checks")
    from . import sanity_checks

    if args.tiny:
        return sanity_checks.run_all(cfg, device="cpu", include_image=args.images)

    if args.fp32:
        # float32 makes the finite-difference slope check strict: in bf16 the perturbation
        # has to be ~3% of a weight to survive rounding, which loosens the tolerance.
        cfg.set_path("model.dtype", "float32")
        print("loading in float32 for a strict finite-difference check")

    device = "cuda" if _cuda_ok() else "cpu"
    model, proc, spec, sketch, ex = setup_model(cfg, device)
    try:
        from .data_prep import load_calibration

        item = load_calibration(cfg, n=1)[0]
    except Exception as e:
        print(f"no calibration data yet ({e}); running text-only checks on a stub item")
        from .gradient import CalibItem

        item = CalibItem(index=-1, image_path="", situation="A cyclist has fallen on a wet road.",
                         action_list=["A. Keep walking.", "B. Stop and help them up."],
                         answer="B")
        args.images = False

    out = sanity_checks.run_all(cfg, extractor=ex, item=item, sketch=sketch,
                                device=device, include_image=args.images)
    _write_json("results/sanity_checks.json", out)
    from .load_model import free

    free(model, ex, sketch)
    return out


def stage_data(cfg, args) -> dict:
    _banner("STAGE: calibration data")
    from .data_prep import build_calibration

    path = build_calibration(cfg, download=not args.no_download, limit=args.image_limit)
    if cfg.get_path("data.text_channel") == "self_caption":
        from .data_prep import add_self_captions

        add_self_captions(cfg, path, device="cuda" if _cuda_ok() else "cpu")
    return {"calibration": path}


def stage_anchor(cfg, args, shared=None) -> dict:
    _banner("STAGE: exact full-backbone anchor")
    from .data_prep import load_calibration
    from .noise_study import run_exact_anchor

    items = load_calibration(cfg)
    made = shared is None
    if made:
        shared = setup_model(cfg, "cuda" if _cuda_ok() else "cpu")
    _model, _proc, _spec, sketch, ex = shared

    out = run_exact_anchor(cfg, ex, sketch, items, grpo=resolve_grpo(cfg))
    print(f"\nmean |alpha| = {out['mean_abs_alpha']:.4f}")
    print(f"sketch m = {out['sketch_m']:,} -> {out['mean_sketch_rel_err']:.1%} measured "
          f"relative error on Lambda")

    from .sketch import cache_plan, describe_cache_plan

    plan = cache_plan(out["mean_abs_alpha"], int(cfg.data.n_pool), int(cfg.sampling.K),
                      budget_gb=float(cfg.get_path("sketch.cache_budget_gb", 30.0)))
    print(describe_cache_plan(plan))
    out["cache_plan"] = plan

    if args.auto_m:
        m_max = int(cfg.get_path("sketch.m_max", 1 << 19))
        new_m = min(out["recommended_m_for_5pct"], m_max)
        if new_m > int(cfg.sketch.m_target):
            print(f"--auto-m: raising sketch.m_target {cfg.sketch.m_target:,} -> {new_m:,}"
                  + (f" (capped at m_max={m_max:,})" if new_m == m_max else ""))
            cfg.set_path("sketch.m_target", new_m)
        else:
            print(f"--auto-m: current m={cfg.sketch.m_target:,} is already sufficient")

    if made:
        from .load_model import free

        free(*shared)
    with open(rel("results/exact_anchor.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def stage_cache(cfg, args, shared=None) -> dict:
    _banner("STAGE: gradient cache")
    from .cache import build_cache, cache_dir
    from .data_prep import load_calibration

    items = load_calibration(cfg)
    made = shared is None
    if made:
        shared = setup_model(cfg, "cuda" if _cuda_ok() else "cpu")
    _model, _proc, _spec, sketch, ex = shared

    K = int(cfg.sampling.K)
    total = len(items) * 2 * K
    print(f"{len(items)} items x 2 modalities x K={K} = {total:,} generations and "
          f"{total:,} sketched backward passes")
    root = build_cache(cfg, ex, sketch, items, root=cache_dir(cfg))
    if made:
        from .load_model import free

        free(*shared)
    return {"cache": root}


def stage_acid(cfg, args, shared=None) -> dict:
    _banner("STAGE: acid test (bootstrap vs fresh)")
    from .cache import GradientCache, cache_dir
    from .data_prep import load_calibration
    from .noise_study import run_acid_test

    items = load_calibration(cfg)
    cache = GradientCache(cache_dir(cfg), device=cfg.get_path("study.device", "auto"))
    print(cache.describe())

    made = shared is None
    if made:
        shared = setup_model(cfg, "cuda" if _cuda_ok() else "cpu")
    _model, _proc, _spec, sketch, ex = shared

    out = run_acid_test(cfg, ex, sketch, cache, items, grpo=resolve_grpo(cfg))
    if made:
        from .load_model import free

        free(*shared)
    with open(rel("results/acid_test.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def stage_sweep(cfg, args, exact=None, acid=None) -> dict:
    _banner("STAGE: noise study and (n, G) sweep")
    from .cache import GradientCache, cache_dir
    from .noise_study import run_sweep, verdict, write_results

    cache = GradientCache(cache_dir(cfg), device=cfg.get_path("study.device", "auto"))
    print(cache.describe())

    convs = both_conventions(cfg) if args.both_conventions else {
        cfg.grpo.convention: resolve_grpo(cfg)}
    sweep = run_sweep(cache, cfg, conventions=convs)

    exact = exact or _load_json(rel("results/exact_anchor.json"))
    acid = acid or _load_json(rel("results/acid_test.json"))
    v = verdict(sweep["rows"], cfg, exact=exact, acid=acid)
    paths = write_results(cfg, sweep, exact=exact, acid=acid, verdict_=v)

    _banner(v["report"])
    print("\nwrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return {"sweep": sweep, "verdict": v, "paths": paths}


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _write_json(relpath: str, payload) -> None:
    path = rel(relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)


def _timed(name: str, fn, timings: dict):
    """Run a stage and record its wall-clock, so Task 1's budget can be planned from
    measured cost rather than from the estimates in the README."""
    import time

    t0 = time.time()
    try:
        return fn()
    finally:
        timings[name] = round(time.time() - t0, 1)
        _write_json("results/timings.json", timings)
        print(f"[stage '{name}' took {timings[name]/60:.1f} min]")


def stage_all(cfg, args) -> dict:
    out: dict = {}
    timings: dict = {}
    # Data first, even though sanity is the gate: without a real calibration item the
    # checks fall back to a text-only stub and the image path — pixel_values plumbing,
    # vision-token accounting, the modality that is actually more fragile — never runs.
    out["data"] = _timed("data", lambda: stage_data(cfg, args), timings)
    out["sanity"] = _timed("sanity", lambda: stage_sanity(cfg, args), timings)
    if not out["sanity"]["ok"] and not args.force:
        raise SystemExit("sanity checks failed — fix them before measuring anything "
                         "(pass --force to override, which you should not do)")

    shared = setup_model(cfg, "cuda" if _cuda_ok() else "cpu")
    try:
        if cfg.get_path("exact_anchor.enabled", True):
            out["anchor"] = _timed("anchor",
                                   lambda: stage_anchor(cfg, args, shared=shared), timings)
            if args.auto_m and int(cfg.sketch.m_target) != shared[3].m:
                from .sketch import BlockSignSketch

                shared = (*shared[:3],
                          BlockSignSketch(shared[2].numels, m_target=int(cfg.sketch.m_target),
                                          seed=int(cfg.sketch.seed),
                                          device=str(shared[3].device)),
                          shared[4])
                print(shared[3].describe())
        out["cache"] = _timed("cache", lambda: stage_cache(cfg, args, shared=shared), timings)
        if cfg.get_path("acid_test.enabled", True):
            out["acid"] = _timed("acid", lambda: stage_acid(cfg, args, shared=shared), timings)
    finally:
        from .load_model import free

        free(*shared)

    out["sweep"] = _timed(
        "sweep",
        lambda: stage_sweep(cfg, args, exact=out.get("anchor"), acid=out.get("acid")),
        timings)
    print(f"\nstage timings (min): "
          f"{ {k: round(v/60, 1) for k, v in timings.items()} }")
    return out


# ------------------------------------------------------------------ cli

STAGES = {"sanity": stage_sanity, "data": stage_data, "anchor": stage_anchor,
          "cache": stage_cache, "acid": stage_acid, "sweep": stage_sweep, "all": stage_all}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--config", default=os.path.join(project_root(), "config.yaml"))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="a.b.c=value", help="config override, value parsed as JSON")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny everything: full pipeline end-to-end in ~15 min")
    ap.add_argument("--fp32", action="store_true",
                    help="sanity: load float32 so the finite-difference slope check is strict")
    ap.add_argument("--tiny", action="store_true",
                    help="sanity: use a 2-layer random model instead of the real checkpoint")
    ap.add_argument("--no-images", dest="images", action="store_false",
                    help="sanity: skip the image-modality checks")
    ap.add_argument("--no-download", action="store_true", help="data: skip image download")
    ap.add_argument("--image-limit", type=int, default=None,
                    help="data: only attempt this many image downloads")
    ap.add_argument("--auto-m", action="store_true",
                    help="anchor: resize the sketch from the measured |alpha| before caching")
    ap.add_argument("--both-conventions", action="store_true",
                    help="sweep: report verl and spec GRPO conventions side by side")
    ap.add_argument("--force", action="store_true", help="all: continue past failed sanity checks")
    args = ap.parse_args(argv)

    overrides = (SMOKE if args.smoke else []) + list(args.overrides)
    cfg = load_config(args.config, overrides)
    os.makedirs(rel("results/figures"), exist_ok=True)
    if args.smoke:
        print("SMOKE MODE: results are for pipeline validation only, not for science.")

    STAGES[args.stage](cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
