"""Read-only checkpoint audit after ordinary baseline GPU work has ended."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def audit(directory):
    output=directory/'completion-audit'
    output.mkdir(exist_ok=False)
    started=time.monotonic()
    torch.set_num_threads(4)
    launcher=json.loads((directory/'launcher.json').read_text())
    if launcher.get('status')!='complete' or launcher.get('exit')!=0 or launcher.get('tagged_processes_remaining'):
        raise ValueError('Baseline not successfully complete and cleaned up')
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    training=[r for r in rows if 'timing_s/step' in r['metrics']]
    if [r['step'] for r in training]!=list(range(1,33)):
        raise ValueError('Expected exactly 32 completed training steps')
    if any(not torch.isfinite(torch.tensor(r['metrics']['actor/grad_norm'])) for r in training):
        raise ValueError('Nonfinite actor gradient norm in the run')
    result=dict(status='running',training_steps=32,checkpoints={},files={})
    save(output/'report.json',result)
    for step in (16,32):
        root=directory/f'checkpoints/global_step_{step}'
        for path in sorted(p for p in root.rglob('*') if p.is_file()):
            phase=time.monotonic(); h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
            result['files'][str(path.relative_to(directory))]=dict(
                bytes=path.stat().st_size,sha256=h.hexdigest(),hash_seconds=time.monotonic()-phase)
            save(output/'report.json',result)
        actor=root/'actor'
        phase=time.monotonic()
        model=torch.load(actor/'model_world_size_1_rank_0.pt',map_location='cpu',mmap=True,weights_only=False)
        if not model or any(not isinstance(t,torch.Tensor) or not torch.isfinite(t).all() for t in model.values()):
            raise ValueError('Invalid or nonfinite model checkpoint')
        model_summary=dict(tensors=len(model),tensor_numel_including_aliases=sum(t.numel() for t in model.values()))
        del model
        optimizer=torch.load(actor/'optim_world_size_1_rank_0.pt',map_location='cpu',mmap=True,weights_only=False)
        counters=[]
        for state in optimizer['state'].values():
            for value in state.values():
                if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite optimizer state')
            if 'step' in state: counters.append(int(state['step']))
        if not counters or set(counters)!={step}:
            raise ValueError('Optimizer counters do not match completed step count')
        lrs=[g['lr'] for g in optimizer['param_groups']]
        if any(lr<=0 for lr in lrs): raise ValueError('Parent has zero next learning rate')
        del optimizer
        extra=torch.load(actor/'extra_state_world_size_1_rank_0.pt',map_location='cpu',weights_only=False)
        if 'lr_scheduler' not in extra or 'rng' not in extra:
            raise ValueError('Missing scheduler or worker RNG state')
        dataloader=torch.load(root/'dataloader.pt',map_location='cpu',weights_only=False)
        result['checkpoints'][str(step)]=dict(
            **model_summary,optimizer_state_entries=len(counters),optimizer_step=step,
            learning_rates=lrs,extra_state_keys=sorted(extra),dataloader_state_keys=sorted(dataloader),
            finite_check_and_load_seconds=time.monotonic()-phase)
        save(output/'report.json',result)
    result.update(status='passed',elapsed_seconds=time.monotonic()-started,
                  scope='File hashes, CPU loadability, finite model/Adam tensors and step counters; '
                        'not a proof of fresh rollout replay or visual transfer')
    save(output/'report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    report=audit(args.directory)
    print(json.dumps({'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
                      'checkpoint_steps':list(report['checkpoints'])}))
