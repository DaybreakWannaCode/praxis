"""Bounded repeated full-Praxis projections and fixed-panel visual outcomes.

This command consumes already measured candidate displacements. It does not
train, select candidates, access final-test images, or choose a horizon.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from .core import derived_seed, digest, dot, seed_all, trainables
from .data import load_manifest
from .experiment import append_json, response_record, visual_gradient, write_json
from .production_displacement import load_tensor
from .production_statistics import alignment_contrasts, outcome_contrast
from .production_visual import FullVisualBackend


def validate(cfg, source_manifest, gates):
    if cfg.get("frozen") is not True:
        raise ValueError("Draft precision configurations cannot run")
    if not 4 <= len(gates) <= 8:
        raise ValueError("Precision check requires four to eight fixed candidates")
    if not 3 <= cfg["probe_repeats"] <= 8 or not 2 <= cfg["probe_samples"] <= 16:
        raise ValueError("Invalid independent probe budget")
    if not 2 <= cfg["outcome_samples"] <= 32 or cfg["horizon"] not in (1, 4):
        raise ValueError("Invalid outcome budget or predeclared horizon")
    if not cfg.get("scene_audit_sha256") or not cfg.get("decision_rule"):
        raise ValueError("Freeze scene audit and decision rule before sampling")
    for path,key in ((cfg["scene_audit"],"scene_audit_sha256"),(source_manifest,"visual_manifest_sha256")):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=cfg[key]:
            raise ValueError("Frozen audit or visual manifest has changed")
    if [str(p) for p in gates]!=cfg["candidate_gates"]:
        raise ValueError("Candidate identities differ from frozen configuration")
    reports=[json.loads((p/"parity.json").read_text()) for p in gates]
    for r in reports:
        if r["status"]!="passed" or not all(r["equal"].values()) or "canonical_validation" not in r:
            raise ValueError("Candidate gate did not pass")
        for field in ("parameters", "buffers", "optimizer", "scheduler"):
            if r["parent_digests"][field]!=reports[0]["parent_digests"][field]:
                raise ValueError("Candidates must share the same complete parent")
    # Current gate exports one optimizer step only. Do not relabel it as H=4.
    if cfg["horizon"]!=1:
        raise ValueError("H=4 total-displacement gate is not implemented")
    mappings=[json.loads((p/"coordinates.json").read_text()) for p in gates]
    if any(m!=mappings[0] for m in mappings):
        raise ValueError("Canonical layouts differ")
    raw=json.loads(Path(source_manifest).read_text())
    if any(r.get("split")=="test" for r in raw):
        raise ValueError("Final-test data is forbidden in development")
    by_id={i.id:i for i in load_manifest(source_manifest)}
    score=[by_id[x] for x in cfg["score_ids"]]
    dev=[by_id[x] for x in cfg["dev_ids"]]
    ids=cfg["score_ids"]+cfg["dev_ids"]
    if not score or not dev or len(ids)!=len(set(ids)):
        raise ValueError("Empty, repeated, or overlapping panels")
    if any(i.split!="score" for i in score) or any(i.split!="dev" for i in dev):
        raise ValueError("Manifest split mismatch")
    if {i.group_id for i in score} & {i.group_id for i in dev}:
        raise ValueError("Scene families overlap")
    if set(cfg["image_sha256"])!=set(ids):
        raise ValueError("Frozen image hash inventory differs")
    for item in score+dev:
        if hashlib.sha256(Path(item.image_path).read_bytes()).hexdigest()!=cfg["image_sha256"][item.id]:
            raise ValueError("Frozen image bytes have changed")
    count=len(score)*cfg["probe_repeats"]*cfg["probe_samples"]
    count+=len(dev)*(len(gates)+2)*cfg["outcome_samples"]
    count+=cfg["outcome_samples"]  # exact parent replay on first dev image
    if count!=cfg["response_budget"] or count>4096:
        raise ValueError("Response budget differs from frozen configuration")
    return reports,mappings[0],score,dev


def run(cfg, source_manifest, gates, output):
    from verl.utils.reward_score.mcq import mcq_compute_score
    from .production_rewards import verify_contract
    contract=verify_contract(cfg["reward_contract"],mcq_compute_score)
    if contract["visual"]["profile"]!=cfg["answer_parser"]:
        raise ValueError("Configured parser differs from the reward contract")
    reports,mapping,score,dev=validate(cfg,source_manifest,gates)
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    write_json(output/"manifest.json",{"status":"running","config":cfg,"gates":reports})
    seed_all(cfg["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    backend=FullVisualBackend(cfg,cfg["parent_model"],mapping)
    write_json(output/"backend.json",backend.metadata)
    parameters=trainables(backend.model)
    before=digest(parameters)
    deltas=[]
    norms=[]
    for gate in gates:
        m=json.loads((gate/"delta/manifest.json").read_text())
        delta={r["name"]:load_tensor(gate/"delta",r) for r in m["parameters"]}
        if set(delta)!=set(parameters) or any(delta[n].shape!=p.shape for n,p in parameters.items()):
            raise ValueError("Displacement coordinate mismatch")
        norm=dot(delta,delta)**.5
        if abs(norm-m["update_norm"])>1e-12*max(1.,norm):
            raise ValueError("Displacement norm mismatch")
        deltas.append(delta)
        norms.append(norm)
    projected=[]
    replicate_scores=[]
    for repeat in range(cfg["probe_repeats"]):
        image_scores=[]
        def per_image(item,g):
            values=[dot(g,d) for d in deltas]
            image_scores.append(values)
            append_json(output/"image-projections.jsonl",{"repeat":repeat,"item_id":item.id,"alignment":values})
        g,rows,info=visual_gradient(backend,score,cfg["probe_samples"],
                                  derived_seed(cfg["seed"],"probe-repeat",repeat),on_item=per_image)
        for row in rows:append_json(output/"responses.jsonl",dict(row,repeat=repeat))
        if digest(parameters)!=before:raise AssertionError("Probe changed parent parameters")
        gn=dot(g,g)**.5
        values=[]
        for i,d in enumerate(deltas):
            a=dot(g,d)
            values.append({"candidate":i,"alignment":a,"cosine":a/(gn*norms[i]) if gn*norms[i] else None,
                           "update_norm":norms[i],"visual_gradient_norm":gn})
        replicate_scores.append({"repeat":repeat,"scores":values,"probe":info})
        projected.append(image_scores)
        del g
    pair_count=len(gates)*(len(gates)-1)//2
    sealed={"replicates":replicate_scores,"sealed_before_outcomes":True,
            "pairwise":alignment_contrasts(projected),
            "bonferroni_pairwise":alignment_contrasts(projected,alpha=.05/pair_count)}
    write_json(output/"scores.json",sealed)
    outcomes={}
    first_parent_tokens={}
    kinds=["parent","null_independent"]+[f"child_{i}" for i in range(len(gates))]+["null_replay"]
    for kind in kinds:
        backend.restore_parent()
        if kind.startswith("child_"):
            index=int(kind.split("_")[1])
            with torch.no_grad():
                for n,p in parameters.items():p.copy_(backend.parent_state[n]+deltas[index][n])
        panel=dev[:1] if kind=="null_replay" else dev
        matrix=[]
        parsed=correct=truncated=responses=0
        for item in panel:
            rewards=[]
            for j in range(cfg["outcome_samples"]):
                # Seed each completion separately: EOS length in one response
                # cannot shift the random stream of the next paired response.
                namespace="independent-null" if kind=="null_independent" else "paired-outcome"
                seed=derived_seed(cfg["seed"],namespace,item.id,j)
                sample=backend.sample(item,"image",1,seed)[0]
                row=dict(response_record(sample,item,seed,kind),sample=j)
                if kind=="parent" and item.id==dev[0].id:first_parent_tokens[j]=row["sequence_token_ids"]
                if kind=="null_replay" and row["sequence_token_ids"]!=first_parent_tokens[j]:
                    raise AssertionError("Parent did not replay exactly")
                append_json(output/"responses.jsonl",row)
                rewards.append(row["correct"])
                parsed+=int(row["parsed"])
                correct+=int(row["correct"])
                truncated+=int(row["truncated"])
                responses+=1
            matrix.append(rewards)
        outcomes[kind]=matrix
        write_json(output/f"outcome-{kind}.json",{"correctness":float(np.mean(matrix)),"matrix":matrix,
                   "parse_success":parsed/responses,"conditional_correctness":correct/parsed if parsed else None,
                   "truncations":truncated,"responses":responses})
    contrasts={}
    for kind in kinds[1:-1]:contrasts[kind]=outcome_contrast(outcomes[kind],outcomes["parent"])
    pairwise=[]
    for a,b in itertools.combinations(range(len(gates)),2):
        first,second=outcomes[f"child_{a}"],outcomes[f"child_{b}"]
        pairwise.append({"first":a,"second":b,"pointwise":outcome_contrast(first,second),
                         "simultaneous":outcome_contrast(first,second,alpha=.05/pair_count)})
    summary={"status":"complete","seconds":time.monotonic()-start,
             "max_torch_allocated_bytes":torch.cuda.max_memory_allocated(),
             "scores":sealed,"parent_contrasts":contrasts,"candidate_contrasts":pairwise,
             "parent_replay_exact":True,"scope":"Fixed-parent development precision; no population transfer claim"}
    write_json(output/"summary.json",summary)
    record=json.loads((output/"manifest.json").read_text())
    record["status"]="complete"
    write_json(output/"manifest.json",record)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--gate",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(json.loads(a.config.read_text()),a.manifest,a.gate,a.output),indent=2))


if __name__=="__main__":main()
