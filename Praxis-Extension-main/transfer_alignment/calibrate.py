"""Bounded inference-only calibration on development images; no optimizer updates."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .core import derived_seed, digest, weights
from .experiment import append_json, response_record, write_json


def summarize(rows):
    if not rows:
        raise ValueError("No responses to summarize")
    n = len(rows)
    parsed = sum(r["parsed"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    lengths = sorted(r["length"] for r in rows)
    return {"responses": n, "correctness": correct/n, "parse_rate": parsed/n,
            "unparsed": n-parsed, "wrong_parsed": sum(r["parsed"] and not r["correct"] for r in rows),
            "accuracy_given_parsed": correct/parsed if parsed else None,
            "truncated": sum(r["truncated"] for r in rows),
            "truncation_rate": sum(r["truncated"] for r in rows)/n,
            "mean_tokens": sum(lengths)/n, "max_tokens_observed": lengths[-1],
            "p95_tokens": lengths[max(0, (95*n+99)//100-1)]}


def run(backend, items, output, *, samples=8, replicates=1, seed=20260907):
    if not items or any(i.split != "dev" for i in items):
        raise ValueError("Calibration uses development images only")
    if len(items)>32 or not 2<=samples<=16 or not 1<=replicates<=3:
        raise ValueError("Calibration exceeds engineering bounds")
    output=Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start=time.monotonic()
    before=digest(weights(backend.model))
    meta={"status":"running", "backend":backend.metadata, "samples":samples,
          "replicates":replicates, "seed":seed, "items":[vars(i) for i in items],
          "parameter_digest_before":before, "training_updates":0}
    write_json(output/"manifest.json",meta)
    all_rows=[]
    summaries=[]
    for rep in range(replicates):
        rep_rows=[]
        for item in items:
            item_seed=derived_seed(seed,"calibration",rep,item.id)
            responses=backend.sample(item,"image",samples,item_seed)
            rows=[dict(response_record(s,item,item_seed,"calibration"),replicate=rep,sample=j)
                  for j,s in enumerate(responses)]
            for row in rows:
                append_json(output/"responses.jsonl",row)
            rep_rows.extend(rows)
            append_json(output/"items.jsonl",dict(item_id=item.id,replicate=rep,**summarize(rows)))
            print(f"Calibration repeat {rep}, {item.id}: {summarize(rows)}",flush=True)
        summaries.append(summarize(rep_rows))
        all_rows.extend(rep_rows)
    after=digest(weights(backend.model))
    if before!=after:
        raise AssertionError("Trainable parameters changed during inference calibration")
    result={"aggregate":summarize(all_rows),"replicates":summaries,
            "seconds":time.monotonic()-start,"trainable_parameters_unchanged":True,
            "note":"Development-only generation calibration; repeated draws share images, not independent task samples."}
    write_json(output/"summary.json",result)
    meta.update(status="complete",parameter_digest_after=after)
    write_json(output/"manifest.json",meta)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True,help="Trusted local full parent checkpoint")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=8)
    p.add_argument("--replicates",type=int,default=1)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    if cfg["max_new_tokens"] not in [512,1024]:
        p.error("Generation calibration supports 512 or 1024 token caps")
    if a.output.exists():
        p.error("Use a new output directory")
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    from .backends import QwenBackend
    from .data import load_manifest
    from .core import coordinate_manifest, seed_all
    from .experiment import load_trusted_checkpoint
    items=[i for i in load_manifest(a.manifest) if i.split=="dev"]
    if not items or len(items)>32 or not 2<=a.samples<=16 or not 1<=a.replicates<=3:
        p.error("Invalid calibration size")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    seed_all(cfg["seed"])
    backend=QwenBackend(cfg)
    parent=load_trusted_checkpoint(a.parent)
    if parent["compact"] or parent["coordinates"]!=coordinate_manifest(backend.model):
        raise ValueError("Calibration requires a matching full parent")
    backend.model.load_state_dict(parent["model"],strict=True)
    del parent
    backend.metadata.update(parent_checkpoint=str(a.parent),purpose="generation_calibration")
    result=run(backend,items,a.output,samples=a.samples,replicates=a.replicates,seed=cfg["seed"])
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
