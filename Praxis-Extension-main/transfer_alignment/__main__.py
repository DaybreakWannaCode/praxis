"""Bounded two-branch pilot; large study execution is intentionally not exposed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .core import isolated_rng, seed_all
from .data import load_manifest, synthetic_items, validate_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["synthetic", "qwen"], default="synthetic")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Validate inputs and environment without loading model")
    args = parser.parse_args()
    default = Path(__file__).parent / "configs" / ("qwen_smoke.json" if args.backend == "qwen" else "synthetic.json")
    cfg = json.loads((args.config or default).read_text())
    for key, low, high in [("candidates", 2, 8), ("batch_size", 1, 8), ("branch_steps", 1, 4),
                            ("warmup_steps", 0, 10), ("probe_group_size", 2, 32),
                            ("text_group_size", 2, 16), ("eval_samples", 1, 64)]:
        if not isinstance(cfg[key], int) or not low <= cfg[key] <= high:
            parser.error(f"{key} must be an integer in [{low}, {high}] for this engineering runner")
    if not 0 < cfg["lr"] <= 0.1 or not 0 < cfg["clip_ratio"] < 1 or cfg["kl_coef"] < 0:
        parser.error("Invalid optimizer settings")
    if args.backend == "qwen" and not args.manifest:
        parser.error("Qwen requires --manifest with local train/score/dev records")
    items = load_manifest(args.manifest) if args.manifest else synthetic_items()
    validate_items(items, real_images=args.backend == "qwen")
    train = [x for x in items if x.split == "train"]
    need = cfg["candidates"] * cfg["batch_size"]
    if len(train) < need:
        parser.error(f"Need {need} distinct training examples; found {len(train)}")
    if args.backend == "qwen" and (sum(i.split == "score" for i in items) > 32 or
                                    sum(i.split == "dev" for i in items) > 64):
        parser.error("Engineering cap: <=32 score and <=64 dev items; separate production configuration required")
    if args.output.exists():
        parser.error("Output directory exists; choose a new path to preserve previous evidence")
    if args.preflight:
        import importlib.util
        import torch
        print(json.dumps({"backend": args.backend, "cuda": torch.cuda.is_available(),
                          "splits": {s: sum(x.split == s for x in items) for s in ["train", "score", "dev", "test"]},
                          "packages": {p: importlib.util.find_spec(p) is not None
                                       for p in ["torch", "transformers", "peft", "PIL"]},
                          "note": "No model loaded, downloaded or trained"}, indent=2))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed_all(cfg["seed"])
    from .backends import TinyBackend, QwenBackend
    from .experiment import run
    backend = TinyBackend() if args.backend == "synthetic" else QwenBackend(cfg)
    import random
    with isolated_rng(cfg["seed"]):
        random.shuffle(train)
    candidates = [train[b*cfg["batch_size"]:(b+1)*cfg["batch_size"]] for b in range(cfg["candidates"])]
    outcomes = run(backend, items, candidates, cfg, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "backend": args.backend,
                      "branches": len(outcomes), "replay_exact": all(x["replay_exact"] for x in outcomes)}, indent=2))


if __name__ == "__main__":
    main()
