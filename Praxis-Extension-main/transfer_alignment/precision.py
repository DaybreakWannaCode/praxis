"""Fixed-budget probe uncertainty and paired-outcome diagnostic, never training."""
from __future__ import annotations
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
import torch
from .core import derived_seed, digest, dot, trainables, weights
from .experiment import append_json, response_record, visual_gradient, write_json


def mean_se(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Need at least two finite cluster values")
    return {"mean": statistics.mean(values), "se": statistics.stdev(values)/math.sqrt(len(values)), "n":len(values)}


def probe_summary(records):
    """Exactly four independent repeats; intervals are descriptive t approximations."""
    if len(records)!=4:
        raise ValueError("The prespecified diagnostic requires four repeats")
    ids=set(records[0])
    if len(ids)<2 or any(set(r)!=ids for r in records):
        raise ValueError("Probe image sets differ")
    if any(len(v)!=2 for r in records for v in r.values()):
        raise ValueError("This diagnostic compares two fixed candidates")
    gap=lambda r: statistics.mean(r[k][0]-r[k][1] for k in sorted(ids))
    mc=mean_se([gap(r) for r in records])
    image=mean_se([statistics.mean(r[k][0]-r[k][1] for r in records) for k in sorted(ids)])
    # df=3, two-sided 95%; few repeats => explicitly approximate, not certification.
    mc['approx_95_interval']=[mc['mean']-3.182446305*mc['se'],mc['mean']+3.182446305*mc['se']]
    lo,hi=mc['approx_95_interval']
    winner=0 if lo>0 else (1 if hi<0 else None)
    return {"candidate_mean_alignment":[statistics.mean(statistics.mean(r[k][j] for k in sorted(ids)) for r in records) for j in range(2)],
            "gap_candidate0_minus_candidate1":mc,"across_image_gap":image,
            "conditional_mc_ranking":([winner,1-winner] if winner is not None else None),
            "selection_authorized":False,
            "note":"MC interval is conditional on these fixed images. Image SE addresses a different source of variation. Four repeats do not certify reliability or authorize data selection."}


def set_weights(model, state):
    params=trainables(model)
    if set(params)!=set(state):raise ValueError("Trainable coordinate mismatch")
    with torch.no_grad():
        for n,p in params.items():
            if p.shape!=state[n].shape:raise ValueError("Coordinate shape mismatch")
            p.copy_(state[n])


def paired_outcomes(backend, items, child, output, *, count=4, seed=20260909):
    if not 2<=len(items)<=16 or any(i.split!='dev' for i in items) or not 2<=count<=8:
        raise ValueError("Bounded development-only paired diagnostic")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    parent=weights(backend.model);before=digest(parent)
    if digest(child)==before:raise ValueError("Need a nonzero actual candidate update")
    table={};start=time.monotonic()
    try:
        for label,state,stream in [('parent',parent,'paired'),('null',parent,'paired'),
                                    ('child_paired',child,'paired'),('child_independent',child,'independent')]:
            set_weights(backend.model,state)
            for item in items:
                s=derived_seed(seed,'precision_outcome',stream,item.id)
                draws=backend.sample(item,'image',count,s)
                if len(draws)!=count:raise ValueError("Unexpected sample count")
                key=(label,item.id)
                table[key]=[response_record(d,item,s,label) for d in draws]
                for j,row in enumerate(table[key]):append_json(output/'responses.jsonl',dict(row,sample=j))
            print('Outcome arm complete:',label,flush=True)
        identical=all(all(a['text']==b['text'] and a['correct']==b['correct'] and a['length']==b['length']
                          for a,b in zip(table['parent',i.id],table['null',i.id])) for i in items)
        if not identical:raise AssertionError("Identical-policy paired generation failed replay")
        changes={}
        for label in ('child_paired','child_independent'):
            changes[label]={i.id:statistics.mean(b['correct']-a['correct'] for a,b in zip(table['parent',i.id],table[label,i.id])) for i in items}
        result={"null_exact_replay":identical,"paired":mean_se(list(changes['child_paired'].values())),
                "independent":mean_se(list(changes['child_independent'].values())),"per_image_changes":changes,
                "sampled_correctness":{label:statistics.mean(r['correct'] for i in items for r in table[label,i.id]) for label in ('parent','null','child_paired','child_independent')},
                "paired_response_change_fraction":statistics.mean(a['text']!=b['text'] for i in items for a,b in zip(table['parent',i.id],table['child_paired',i.id])),
                "seconds":time.monotonic()-start,
                "note":"Same seed and batch shape couple policies; marginal sampling is unchanged. A single panel does not establish variance reduction. Zero observed differences do not prove equivalence or zero population variance."}
        write_json(output/'summary.json',result)
    finally:
        set_weights(backend.model,parent)
        if digest(weights(backend.model))!=before:raise AssertionError("Parent restoration failed")
    return result


def run(backend, items, deltas, child, output, *, seed=20260909):
    score=[i for i in items if i.split=='score'];dev=[i for i in items if i.split=='dev'][:8]
    if not 2<=len(score)<=16 or len(deltas)!=2:raise ValueError("Bounded two-candidate probe")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    before=digest(weights(backend.model));records=[];start=time.monotonic()
    write_json(output/'manifest.json',{'status':'running','backend':backend.metadata,'seed':seed,
        'protocol':{'repeats':4,'probe_samples_per_image':8,'outcome_samples_per_image':4,'no_optional_retries':True},
        'score_items':[vars(i) for i in score],'dev_items':[vars(i) for i in dev],
        'delta_digests':[digest(d) for d in deltas],'child_digest':digest(child),'parent_digest':before,
        'source_hashes':{p.name:digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}})
    for rep in range(4):
        values={}
        def capture_item(item,g):values[item.id]=[dot(g,d) for d in deltas]
        g,rows,info=visual_gradient(backend,score,8,derived_seed(seed,'precision_probe',rep),on_item=capture_item)
        scores=[dot(g,d) for d in deltas]
        for j in range(2):
            if not math.isclose(scores[j],statistics.mean(v[j] for v in values.values()),rel_tol=1e-4,abs_tol=1e-8):
                raise AssertionError('Item decomposition differs from global alignment')
        records.append(values)
        write_json(output/f'probe-{rep}.json',{'items':values,'alignment':scores,'probe':info})
        for row in rows:append_json(output/'probe-responses.jsonl',dict(row,replicate=rep))
        print('Probe complete:',rep,scores,flush=True)
    if digest(weights(backend.model))!=before:raise AssertionError('Probe modified parameters')
    summary=probe_summary(records);write_json(output/'probe-summary.json',summary)
    paired_outcomes(backend,dev,child,output/'outcomes',seed=seed)
    manifest=json.loads((output/'manifest.json').read_text());manifest['status']='complete'
    write_json(output/'manifest.json',manifest)
    write_json(output/'completed.json',{'status':'complete','seconds':time.monotonic()-start,'training_updates':0})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','manifest','parent','child','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--delta',type=Path,action='append',required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.output.exists():p.error('Use a new output directory')
    if cfg['max_new_tokens']!=512 or cfg.get('generation_batch_size')!=8:
        p.error('This follow-up fixes the calibrated 512-token cap and batch size 8')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from .core import coordinate_manifest,seed_all
    from .backends import QwenBackend
    from .data import load_manifest
    from .experiment import load_trusted_checkpoint
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4);seed_all(cfg['seed'])
    items=load_manifest(a.manifest);backend=QwenBackend(cfg)
    parent=load_trusted_checkpoint(a.parent);package=load_trusted_checkpoint(a.child)
    if package['parent_id']!=digest(parent):raise ValueError('Child belongs to a different full parent')
    child=package['state'];del package
    if parent['compact'] or not child['compact'] or parent['coordinates']!=coordinate_manifest(backend.model) or child['coordinates']!=parent['coordinates']:
        raise ValueError('Mismatched full parent / compact child')
    backend.model.load_state_dict(parent['model'],strict=True)
    child_weights={n:child['model'][n] for n in trainables(backend.model)}
    if any(not torch.equal(v,parent['model'][n]) for n,v in child['model'].items() if n not in child_weights):
        raise ValueError('This paired runner requires unchanged child buffers/frozen state')
    deltas=[torch.load(f,map_location='cpu',weights_only=True) for f in a.delta]
    actual={n:child_weights[n].float()-v for n,v in weights(backend.model).items()}
    if digest(actual)!=digest(deltas[0]):raise ValueError('Child is not candidate 0 from this parent')
    del parent,child
    backend.metadata.update(parent_checkpoint=str(a.parent),child_checkpoint=str(a.child))
    run(backend,items,deltas,child_weights,a.output,seed=cfg['seed'])


if __name__=='__main__':main()
