"""Bounded scoring-only check on the existing single cost candidate.

Keeps the probe gradient in CPU RAM, never writes a gradient or child checkpoint,
and never deletes the archived input. No training or final-test data access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from .choice_measure import evaluate, mean, save
from .choice_probe import SYSTEM_PROMPT, objective
from .production_precision import validate
from .production_visual import FullVisualBackend
from .streaming_projection import project_and_apply


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def run(plan, output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status=dict(status='running',scope='One archived candidate, scoring split only; no training, test access or deletion',plan=plan)
    save(output/'manifest.json',status)
    for source, expected in plan['artifact_sha256'].items():
        if file_hash(source)!=expected:raise ValueError('Frozen scoring input changed: '+source)
    cfg=json.loads(Path(plan['visual_config']).read_text())
    _,_,score,_=validate(cfg,Path(plan['visual_manifest']),[Path(p) for p in cfg['candidate_gates']])
    if len(score)!=16:raise ValueError('Exactly the existing 16-image scoring panel is allowed')
    root=Path(plan['candidate_dir'])
    cost=json.loads((root/'cost.json').read_text())
    delta=json.loads((root/'delta/manifest.json').read_text())
    mapping=json.loads((root/'coordinates.json').read_text())
    parent=Path(plan['parent_model'])
    if cost['status']!='complete' or Path(cost['parent_model']).resolve()!=parent.resolve():
        raise ValueError('Candidate parent identity differs')
    if not cost['child_reconstruction_exact'] or cost['maximum_optimizer_step_increment']!=1:
        raise ValueError('Candidate did not pass one-update export checks')
    if cost['update_norm']!=delta['update_norm'] or delta['canonical_numel']!=3754622976:
        raise ValueError('Candidate manifest differs from verified cost receipt')
    phase=time.monotonic()
    if file_hash(parent)!=plan['parent_sha256']:raise ValueError('Parent checksum mismatch')
    parent_hash_seconds=time.monotonic()-phase
    status.update(parent_hash_seconds=parent_hash_seconds,score_ids=[i.id for i in score],
                  objective='normalized_label_prefix_log_likelihood_v1',system_prompt=SYSTEM_PROMPT)
    save(output/'manifest.json',status)
    torch.set_num_threads(4)
    torch.manual_seed(20260915);torch.cuda.manual_seed_all(20260915)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.use_deterministic_algorithms(True)
    phase=time.monotonic()
    backend=FullVisualBackend(cfg,parent,mapping,attn_implementation='sdpa')
    backend.model.eval()
    params=dict(backend.model.named_parameters())
    startup_seconds=time.monotonic()-phase
    parent_rows=evaluate(backend,score)
    if evaluate(backend,[score[0]])[0]!=parent_rows[0]:raise ValueError('Parent replay failed')
    save(output/'parent-score.json',dict(rows=parent_rows))
    phase=time.monotonic()
    gradient={n:torch.zeros(p.shape,dtype=torch.float64) for n,p in params.items()}
    for index,item in enumerate(score):
        backend.model.zero_grad(set_to_none=True)
        value,_=objective(backend,item)
        if abs(float(value.detach())-parent_rows[index]['log_q'])>1e-7:raise ValueError('Forward/gradient objective differs')
        value.backward()
        for n,p in params.items():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite visual gradient')
                gradient[n].add_(p.grad.detach().cpu().double()/len(score))
        save(output/'progress.json',dict(stage='gradient',completed_images=index+1))
    backend.model.zero_grad(set_to_none=True)
    gradient_seconds=time.monotonic()-phase
    phase=time.monotonic()
    try:
        projection=project_and_apply(gradient,backend.parent_state,params,root/'delta',delta)
        projection_seconds=time.monotonic()-phase
        # Alignment is sealed before child visual scoring.
        save(output/'alignment.json',projection)
        phase=time.monotonic()
        child_rows=evaluate(backend,score)
        direct_seconds=time.monotonic()-phase
        save(output/'candidate-score.json',dict(alignment=projection['alignment'],cosine=projection['cosine'],
            direct_lookahead=mean(child_rows)-mean(parent_rows),rows=child_rows,
            warning='Same scoring panel, not independent transfer validation'))
    finally:
        backend.restore_parent()
    replay=evaluate(backend,[score[0]])[0]
    if replay!=parent_rows[0]:raise ValueError('Final parent restoration replay differs')
    summary=dict(status='complete',parent_hash_seconds=parent_hash_seconds,startup_seconds=startup_seconds,
        gradient_seconds=gradient_seconds,projection_seconds=projection_seconds,direct_seconds=direct_seconds,
        elapsed_seconds=time.monotonic()-started,gradient_ram_bytes=sum(g.numel()*g.element_size() for g in gradient.values()),
        new_model_or_gradient_files=0,source_delta_bytes=cost['export_bytes'],
        parent_replay_exact=True,validated_tensors=projection['validated_tensors'],
        alignment=projection['alignment'],direct_lookahead=mean(child_rows)-mean(parent_rows),
        limitations='One existing archived candidate. No export-time or parent-reuse speedup measured. Existing archive retained.')
    save(output/'summary.json',summary)
    status['status']='complete';save(output/'manifest.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(json.loads(a.plan.read_text()),a.output)
