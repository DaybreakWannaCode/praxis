"""Bounded two-branch visual integration check, not a precision study or trainer."""
import argparse
import json
import time
from pathlib import Path

import torch

from .core import derived_seed, digest, dot, seed_all, trainables
from .data import load_manifest
from .experiment import append_json, response_record, visual_gradient, write_json
from .production_displacement import load_tensor
from .production_visual import FullVisualBackend


def run(cfg, source_manifest, gates, output):
    from verl.utils.reward_score.mcq import mcq_compute_score
    from .production_rewards import verify_contract
    verify_contract(cfg["reward_contract"],mcq_compute_score)
    if len(gates)!=2:raise ValueError("This integration check requires exactly two candidates")
    reports=[json.loads((p/"parity.json").read_text()) for p in gates]
    if any(r["status"]!="passed" or not all(r["equal"].values()) or "canonical_validation" not in r for r in reports):
        raise ValueError("Each candidate must pass full-coordinate and update parity gates")
    if reports[0]["parent_digests"]["parameters"]!=reports[1]["parent_digests"]["parameters"]:
        raise ValueError("Candidates have different parent weights")
    for field in ("optimizer","scheduler","buffers"):
        if reports[0]["parent_digests"][field]!=reports[1]["parent_digests"][field]:
            raise ValueError(f"Candidates have different parent {field}")
    # Worker RNG after sampling may differ across candidate text batches.
    # Each gate restores its captured fixed-input state; fresh-engine replay is
    # separately reported, not inferred from these state comparisons.
    mappings=[json.loads((p/"coordinates.json").read_text()) for p in gates]
    if mappings[0]!=mappings[1]:raise ValueError("Candidate canonical layouts differ")
    if any(r.get("split")=="test" for r in json.loads(Path(source_manifest).read_text())):
        raise ValueError("Do not supply final-test images to this development command")
    items=load_manifest(source_manifest)
    by_id={i.id:i for i in items}
    score=[by_id[x] for x in cfg["score_ids"]]
    dev=[by_id[x] for x in cfg["dev_ids"]]
    if not 2<=len(score)<=8 or not 2<=len(dev)<=8 or cfg["samples"]!=4:
        raise ValueError("Integration response budget exceeded")
    if any(x.split!="score" for x in score) or any(x.split!="dev" for x in dev):
        raise ValueError("Scoring and development splits differ from the locked manifest")
    if len(set(cfg["score_ids"]+cfg["dev_ids"]))!=len(score)+len(dev):
        raise ValueError("Repeated or overlapping visual IDs")
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    write_json(output/"manifest.json",{"status":"running","purpose":"production two-branch integration only",
               "config":cfg,"candidate_gate_reports":reports,"coordinates":mappings[0],
               "scene_family_audit":"pending; engineering panels only"})
    seed_all(cfg["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    backend=FullVisualBackend(cfg,cfg["parent_model"],mappings[0])
    write_json(output/"backend.json",backend.metadata)
    parameters=trainables(backend.model)
    before=digest(parameters)
    deltas=[]
    norms=[]
    for gate in gates:
        m=json.loads((gate/"delta/manifest.json").read_text())
        delta={r["name"]:load_tensor(gate/"delta",r) for r in m["parameters"]}
        if set(delta)!=set(parameters):raise ValueError("Visual/delta parameter names differ")
        norm=dot(delta,delta)**.5
        if abs(norm-m["update_norm"])>1e-12*max(1.,norm):raise ValueError("Saved delta norm differs")
        deltas.append(delta)
        norms.append(norm)
    def per_image(item,g):
        append_json(output/"image-projections.jsonl",{"item_id":item.id,
                    "alignment":[dot(g,d) for d in deltas]})
    g,rows,info=visual_gradient(backend,score,cfg["samples"],derived_seed(cfg["seed"],"score"),on_item=per_image)
    for row in rows:append_json(output/"responses.jsonl",row)
    if digest(parameters)!=before:raise AssertionError("Visual probe changed model parameters")
    gn=dot(g,g)**.5
    scores=[]
    for i,d in enumerate(deltas):
        a=dot(g,d)
        scores.append({"candidate":i,"alignment":a,"cosine":a/(gn*norms[i]) if gn*norms[i] else None,
                       "update_norm":norms[i],"visual_gradient_norm":gn})
    write_json(output/"scores.json",{"scores":scores,"probe":info,"sealed_before_outcomes":True})
    del g
    outcomes={}
    parent_sequences={}
    for kind in ("parent","null_replay","null_independent","child_0","child_1"):
        backend.restore_parent()
        if kind.startswith("child"):
            index=int(kind[-1])
            with torch.no_grad():
                for n,p in parameters.items():p.copy_(backend.parent_state[n]+deltas[index][n])
        results=[]
        for item in dev:
            key="parent" if kind=="null_replay" else kind
            seed=derived_seed(cfg["seed"],"outcome",key,item.id)
            samples=backend.sample(item,"image",cfg["samples"],seed)
            for j,s in enumerate(samples):
                row=dict(response_record(s,item,seed,kind),sample=j)
                seq=row["sequence_token_ids"]
                if kind=="parent":parent_sequences[(item.id,j)]=seq
                if kind=="null_replay" and seq!=parent_sequences[(item.id,j)]:
                    raise AssertionError("Restored parent visual generation did not replay")
                append_json(output/"responses.jsonl",row)
                results.append(row)
        outcomes[kind]={"correctness":sum(r["correct"] for r in results)/len(results),
                        "parse_success":sum(r["parsed"] for r in results)/len(results),
                        "conditional_correctness":(sum(r["correct"] for r in results)/sum(r["parsed"] for r in results)
                                                   if any(r["parsed"] for r in results) else None),
                        "truncations":sum(r["truncated"] for r in results),"responses":len(results)}
        write_json(output/f"outcome-{kind}.json",outcomes[kind])
    summary={"status":"complete","scores":scores,"outcomes":outcomes,"parent_replay_exact":True,
             "seconds":time.monotonic()-start,"max_torch_allocated_bytes":torch.cuda.max_memory_allocated(),
             "scope":"Small integration check; no reliable transfer or precision claim"}
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
