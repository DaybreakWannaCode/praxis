"""Two independent visual probes and one no-update outcome repeat; no training."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .core import alignment, derived_seed, digest, weights
from .experiment import append_json, visual_gradient, write_json


def null_comparison(first, second):
    if not first or set(first)!=set(second):
        raise ValueError("Null comparison requires the same outcome images")
    diffs={k:second[k]-first[k] for k in first}
    n=len(diffs);mean=sum(diffs.values())/n
    se=(sum((d-mean)**2 for d in diffs.values())/(n*(n-1)))**.5 if n>1 else None
    return {"no_update_mean_change":mean,"per_image_changes":diffs,
            "descriptive_paired_image_se":se,
            "note":"One repeated evaluation on fixed images; not a validated population noise distribution."}


def run(backend, items, deltas, output, *, seed=20260908, group_size=8):
    import torch
    score=[i for i in items if i.split=="score"]
    if not 2<=len(score)<=16 or not 2<=group_size<=16 or not 2<=len(deltas)<=8:
        raise ValueError("Stability calibration exceeds bounds")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    before=digest(weights(backend.model));start=time.monotonic()
    write_json(output/"manifest.json",{"status":"running","backend":backend.metadata,
                "seed":seed,"group_size":group_size,"score_items":[vars(i) for i in score],
                "delta_digests":[digest(d) for d in deltas],"training_updates":0})
    gradients=[];records=[]
    for rep in range(2):
        g,rows,info=visual_gradient(backend,score,group_size,derived_seed(seed,"probe_repeat",rep))
        gradients.append(g)
        torch.save(g,output/f"gradient-{rep}.pt")
        for row in rows:append_json(output/"responses.jsonl",dict(row,replicate=rep))
        scores=[alignment(g,d) for d in deltas]
        values=[s["alignment"] for s in scores]
        ranking=sorted(range(len(scores)),key=lambda k:values[k],reverse=True) if len(set(values))==len(values) else None
        records.append({"replicate":rep,"probe":info,"scores":scores,
                        "ranking":ranking})
        write_json(output/f"probe-{rep}.json",records[-1])
        print(f"Visual probe {rep} complete: {records[-1]}",flush=True)
    unchanged=before==digest(weights(backend.model))
    if not unchanged:raise AssertionError("Probe changed model parameters")
    result={"replicates":records,"gradient_agreement":alignment(gradients[0],gradients[1]),
            "same_ranking":records[0]["ranking"]==records[1]["ranking"] if all(r["ranking"] is not None for r in records) else None,
            "seconds":time.monotonic()-start,"trainable_parameters_unchanged":unchanged,
            "note":"Two estimates and few candidate deltas are engineering stability evidence only."}
    write_json(output/"summary.json",result)
    manifest=json.loads((output/"manifest.json").read_text());manifest["status"]="complete"
    write_json(output/"manifest.json",manifest)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--delta",type=Path,action="append",required=True)
    p.add_argument("--previous-outcomes",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Use a new output directory")
    cfg=json.loads(a.config.read_text())
    if cfg["max_new_tokens"] not in [512,1024]:p.error("Calibrate a supported token cap first")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    from .backends import QwenBackend
    from .data import load_manifest
    from .core import coordinate_manifest,seed_all
    from .experiment import load_trusted_checkpoint
    from .calibrate import run as calibration_run
    items=load_manifest(a.manifest);dev=[i for i in items if i.split=="dev"]
    if not 2<=len(dev)<=32:p.error("Need 2-32 development images")
    prior=[json.loads(s) for s in a.previous_outcomes.read_text().splitlines()]
    if any(r["replicate"]!=0 for r in prior):p.error("Supply a single previous outcome replicate")
    means={r["item_id"]:r["correctness"] for r in prior}
    if len(means)!=len(prior) or set(means)!={i.id for i in dev}:p.error("Previous outcome IDs must uniquely match development images")
    previous=json.loads((a.previous_outcomes.parent/"manifest.json").read_text())
    if previous["status"]!="complete" or previous["samples"]!=8:p.error("Need completed eight-sample parent calibration")
    if {i["id"]:i for i in previous["items"]}!={i.id:vars(i) for i in dev}:p.error("Outcome image content changed")
    old_cfg=previous["backend"]["config"]
    if {k:v for k,v in old_cfg.items() if k!="seed"}!={k:v for k,v in cfg.items() if k!="seed"}:p.error("Null comparison requires the same generation configuration")
    if cfg["seed"]==previous["seed"]:p.error("Use an independent seed for the no-update repeat")
    if Path(previous["backend"]["parent_checkpoint"]).resolve()!=a.parent.resolve():p.error("Parent checkpoint differs")
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4);seed_all(cfg["seed"])
    backend=QwenBackend(cfg)
    parent=load_trusted_checkpoint(a.parent)
    if parent["compact"] or parent["coordinates"]!=coordinate_manifest(backend.model):
        raise ValueError("Need matching full parent")
    backend.model.load_state_dict(parent["model"],strict=True);del parent
    if digest(weights(backend.model))!=previous["parameter_digest_before"]:
        raise ValueError("Parent adapter parameters differ from the previous evaluation")
    backend.metadata.update(parent_checkpoint=str(a.parent),purpose="fixed_parent_stability")
    deltas=[torch.load(path,map_location="cpu",weights_only=True) for path in a.delta]
    run(backend,items,deltas,a.output,seed=cfg["seed"])
    calibration_run(backend,dev,a.output/"outcome-repeat",samples=8,seed=cfg["seed"])
    repeat=[json.loads(s) for s in (a.output/"outcome-repeat/items.jsonl").read_text().splitlines()]
    write_json(a.output/"null-outcome.json",null_comparison(means,{r["item_id"]:r["correctness"] for r in repeat}))
    write_json(a.output/"completed.json",{"status":"complete","training_updates":0})


if __name__=="__main__":main()
