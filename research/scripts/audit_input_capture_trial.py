"""CPU audit of completed original-worker input capture, before visual scoring."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import torch
from transfer_alignment.candidate_inputs import input_digest
from transfer_alignment.core import digest

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
root=a.run
read=lambda p:json.loads(p.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
launcher=read(root/'launcher.json')
assert launcher['status']=='complete' and launcher['exit']==0 and not launcher['tagged_processes_remaining']
assert (root/'run.exit').read_text().strip()=='0'
plan=read(root/'plan.json');capture=read(root/'inputs/capture.json')
assert capture['status']=='updated' and capture['serialization_roundtrip_exact'] and capture['capture_state_unchanged']
assert capture['parent_receipt_sha256']==plan['parent_receipt_sha256']
path=root/'inputs/fixed-update-input.pt';assert sha(path)==capture['input_file_sha256']
payload=torch.load(path,map_location='cpu',weights_only=False)
data=payload['data']
assert input_digest(data)==capture['input_digest']
assert digest(payload['worker_state'])==capture['worker_state_digest']
assert int(data.batch.batch_size[0])==160
required={'input_ids','responses','attention_mask','position_ids','old_log_probs','advantages'}
assert required.issubset(data.batch.keys())
for key,value in data.batch.items():
    if value.is_floating_point():assert torch.isfinite(value).all(),key
before,after=capture['before_steps'],capture['after_steps'];assert len(before)==len(after)>0
changes=[x-y for x,y in zip(after,before)];assert min(before)>0 and max(changes)==1 and all(x in (0,1) for x in changes)
actor=Path(plan['scratch'])/'exports/global_step_1/actor'
cost=read(actor/'cost.json');manifest=read(actor/'delta/manifest.json')
assert cost['status']=='complete' and cost['child_reconstruction_exact']
assert cost['optimizer_state_entries']==len(before) and cost['incremented_entries']==sum(x==1 for x in changes)
assert cost['maximum_optimizer_step_increment']==1
rows=manifest['parameters'];assert len(rows)==len({r['name'] for r in rows})==824
assert manifest['canonical_numel']==sum(r['numel'] for r in rows)==3754622976
assert math.isfinite(manifest['update_norm']) and manifest['update_norm']==cost['update_norm']>0
for row in rows:
    assert Path(row['file']).name==row['file']
    assert (actor/'delta'/row['file']).stat().st_size==row['compressed_bytes']
size=sum(row['compressed_bytes'] for row in rows)
assert size==cost['export_bytes']<=plan['export_limit_bytes']
exact=root/'export-receipt/byte-exact';exact.mkdir(exist_ok=True)
variance={}
for relative,name in [('cost.json','cost.json'),('coordinates.json','coordinates.json'),('delta/manifest.json','delta-manifest.json')]:
    source=actor/relative;original=root/'export-receipt'/name
    assert read(source)==read(original),'Semantic receipt mismatch: '+name
    variance[name]=sha(source)!=sha(original)
    dest=exact/name
    if dest.exists():assert sha(dest)==sha(source)
    else:
        with source.open('rb') as f,dest.open('xb') as out:
            shutil.copyfileobj(f,out);out.flush();os.fsync(out.fileno())
    assert sha(dest)==sha(source)
report=dict(status='passed',input_bytes=path.stat().st_size,input_rows=160,
    tensor_shapes={k:list(v.shape) for k,v in data.batch.items()},
    populated_optimizer_states=len(before),incremented_states=sum(x==1 for x in changes),
    update_norm=cost['update_norm'],export_bytes=size,compressed_cap_bytes=plan['export_limit_bytes'],
    original_receipt_byte_variance=variance,byte_exact_receipts_verified=True,
    scope='Input serialization/digests and export metadata; raw displacement checksums still require scorer consumption',
    source_sha256={'input':sha(path),'plan':sha(root/'plan.json'),'manifest':sha(actor/'delta/manifest.json')})
(root/'capture-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('tensor_shapes','source_sha256')},indent=2))
