"""Frozen, resumable greedy before/after evaluation; no training or test access."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import torch
from .checkpoint_publication import atomic_json
from .data import Item
from .experiment import response_record
from .production_visual import FullVisualBackend


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(plan_path, endpoint, output, resume=False):
    plan=json.loads(plan_path.read_text());identity=sha(plan_path)
    if resume and not output.exists():raise ValueError('Cannot resume missing output')
    output.mkdir(parents=True,exist_ok=resume)
    with (output/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record_path=output/'manifest.json'
        if resume:
            record=json.loads(record_path.read_text())
            if record['plan_sha256']!=identity or record['endpoint']!=endpoint:raise ValueError('Resume identity differs')
            if record['status']=='complete':raise ValueError('Endpoint already complete; do not rerun')
        else:
            record=dict(status='running',plan_sha256=identity,endpoint=endpoint,attempts=[])
        started=time.monotonic()
        record['attempts'].append(dict(started_wall=time.time(),resumed=resume))
        atomic_json(record_path,record)
        required={plan['panel'],plan['coordinates'],plan['endpoints']['final']['path'],*plan['model_files']}
        if not required.issubset(plan['artifact_sha256']):raise ValueError('Required model/data identities missing')
        actual_files={str(p) for p in Path(plan['visual_config']['model']).rglob('*') if p.is_file()}
        if actual_files!=set(plan['model_files']):raise ValueError('Pinned model directory contents differ')
        for path,expected in plan['artifact_sha256'].items():
            if sha(path)!=expected:raise ValueError('Frozen artifact changed: '+path)
        panel=json.loads(Path(plan['panel']).read_text())
        rows=panel['items']+panel['shuffled_controls']
        if len(panel['items'])!=256 or len(panel['shuffled_controls'])!=32 or any(r['split']!='dev' for r in rows):
            raise ValueError('Unexpected development endpoint scope')
        for path,expected in panel['image_sha256'].items():
            if sha(path)!=expected:raise ValueError('Frozen image changed: '+path)
        cfg=plan['visual_config']
        if cfg['max_new_tokens']!=512 or cfg['answer_parser']!='explicit_final_v3':raise ValueError('Decoding contract differs')
        model=plan['endpoints'][endpoint]
        if endpoint=='initial' and model['kind']!='pinned_pretrained':raise ValueError('Initial model must be pretrained')
        if endpoint=='final' and model['kind']!='checkpoint':raise ValueError('Final model must be audited checkpoint')
        directory=output/'responses';directory.mkdir(exist_ok=resume)
        cached=[]
        for index,row in enumerate(rows):
            path=directory/f'{index:04d}.json'
            if not path.exists():break
            value=json.loads(path.read_text())
            if value['item_id']!=row['id'] or value['group_id']!=row['group_id'] or value['plan_sha256']!=identity:
                raise ValueError('Saved response identity differs')
            cached.append(value)
        if len(list(directory.glob('*.json')))!=len(cached):raise ValueError('Response sequence has gaps or unexpected files')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        mapping=json.loads(Path(plan['coordinates']).read_text())
        backend=FullVisualBackend(cfg,model.get('path'),mapping,pretrained_initial=endpoint=='initial')
        backend.model.eval()
        record['backend']=backend.metadata;record['attempts'][-1]['startup_seconds']=time.monotonic()-started
        atomic_json(record_path,record)
        if cached:
            replay=backend.sample(Item(**rows[0]),'image',1,plan['seed'],greedy=True)[0]
            if replay.payload[0].tolist()!=cached[0]['sequence_token_ids']:raise ValueError('Resume anchor replay differs')
            record['attempts'][-1]['resume_anchor_replay_exact']=True
            atomic_json(record_path,record)
        for index in range(len(cached),len(rows)):
            item=Item(**rows[index]);tick=time.monotonic()
            response=backend.sample(item,'image',1,plan['seed']+index,greedy=True)[0]
            value=response_record(response,item,plan['seed']+index,'correct_image' if index<256 else 'shuffled_image')
            value.update(plan_sha256=identity,seconds=time.monotonic()-tick)
            atomic_json(directory/f'{index:04d}.json',value);cached.append(value)
            atomic_json(output/'progress.json',dict(completed=len(cached),total=288,endpoint=endpoint))
        replay=backend.sample(Item(**rows[0]),'image',1,plan['seed']+10000,greedy=True)[0]
        if replay.payload[0].tolist()!=cached[0]['sequence_token_ids']:raise ValueError('Final greedy replay differs')
        summary=dict(status='complete',endpoint=endpoint,count=288,greedy_replay_exact=True,
                     plan_sha256=identity,final_attempt_seconds=time.monotonic()-started,
                     new_model_or_gradient_files=0,scope=plan['scope'])
        for kind in ['correct_image','shuffled_image']:
            subset=[r for r in cached if r['kind']==kind]
            summary[kind]=dict(count=len(subset),correct=sum(r['correct'] for r in subset),
                parsed=sum(r['parsed'] for r in subset),truncated=sum(r['truncated'] for r in subset))
        atomic_json(output/'summary.json',summary)
        record['status']='complete';record['attempts'][-1]['elapsed_seconds']=time.monotonic()-started
        atomic_json(record_path,record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--endpoint',choices=['initial','final'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();run(args.plan,args.endpoint,args.output,args.resume)
