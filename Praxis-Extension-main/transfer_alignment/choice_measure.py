"""Development-only likelihood check on the four archived H4 displacements.

Run with --config (the old frozen H4 config), --manifest, --output.
The old config is used ONLY for artifact/split validation. The new objective,
precision and prompt are explicit constants, not its sampling settings.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from .choice_probe import SYSTEM_PROMPT, objective
from .production_precision import validate
from .production_visual import FullVisualBackend
from .production_displacement import load_tensor


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def evaluate(backend, items):
    result = []
    with torch.no_grad():
        for item in items:
            value, tokens = objective(backend, item)
            result.append(dict(id=item.id, group_id=item.group_id, log_q=float(value), tokens=tokens))
    return result


def mean(rows):
    return sum(r['log_q'] for r in rows)/len(rows)


def run(cfg, manifest, output):
    gates = [Path(x) for x in cfg['candidate_gates']]
    if cfg['horizon'] != 4 or len(gates) != 4:
        raise ValueError('Only the four archived H4 candidates are authorized')
    reports, mapping, score, dev = validate(cfg, manifest, gates)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(status='running', objective='normalized_label_prefix_log_likelihood_v1',
                  units='natural log probability per image', system_prompt=SYSTEM_PROMPT,
                  precision='FP32, no autocast, no TF32, math SDPA',
                  source_config=cfg, score_ids=[i.id for i in score], dev_ids=[i.id for i in dev],
                  scope='development only; no generated answers, no text training',
                  hardware_caveat='archived text updates mix H100 and A100',
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__), Path(__file__).with_name('choice_probe.py')]})
    save(output/'manifest.json', record)
    torch.set_num_threads(4)
    torch.manual_seed(20260914)
    torch.cuda.manual_seed_all(20260914)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    backend = FullVisualBackend(cfg, cfg['parent_model'], mapping, attn_implementation='sdpa')
    backend.model.eval()
    parameters = dict(backend.model.named_parameters())
    torch.cuda.reset_peak_memory_stats()
    phase = time.monotonic()
    parent = dict(score=evaluate(backend, score), dev=evaluate(backend, dev))
    replay = evaluate(backend, [score[0]])[0]
    if replay != parent['score'][0]:
        raise ValueError('Deterministic parent replay failed')
    save(output/'parent.json', parent)
    forward_seconds = time.monotonic()-phase
    phase = time.monotonic()
    # One image at a time: only one computation graph lives on the GPU.
    # CPU FP64 accumulation avoids repeated FP32 accumulation loss.
    gradient = {n:torch.zeros(p.shape, dtype=torch.float64) for n,p in parameters.items()}
    for index, item in enumerate(score):
        backend.model.zero_grad(set_to_none=True)
        value, _ = objective(backend, item)
        if abs(float(value.detach())-parent['score'][index]['log_q']) > 1e-7:
            raise ValueError('Forward and gradient objective differ')
        value.backward()
        for n,p in parameters.items():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all(): raise ValueError('Nonfinite gradient')
                gradient[n].add_(p.grad.detach().cpu().double()/len(score))
        save(output/'progress.json', dict(stage='gradient', completed_images=index+1))
    backend.model.zero_grad(set_to_none=True)
    gnorm = math.sqrt(sum(float((g*g).sum()) for g in gradient.values()))
    if not math.isfinite(gnorm) or gnorm == 0: raise ValueError('Invalid gradient norm')
    gradient_seconds = time.monotonic()-phase
    results = []
    for index, gate in enumerate(gates):
        phase = time.monotonic()
        displacement = json.loads((gate/'delta/manifest.json').read_text())
        rows = {r['name']:r for r in displacement['parameters']}
        if set(rows) != set(parameters): raise ValueError('Displacement names differ')
        alignment = norm2 = 0.
        backend.restore_parent()
        with torch.no_grad():
            for n,p in parameters.items():
                d = load_tensor(gate/'delta', rows[n]).double()
                if d.shape != p.shape or not torch.isfinite(d).all():
                    raise ValueError('Invalid displacement')
                alignment += float((gradient[n]*d).sum())
                norm2 += float((d*d).sum())
                p.copy_(backend.parent_state[n].double()+d)
        norm = math.sqrt(norm2)
        if abs(norm-displacement['update_norm']) > 1e-12*max(1., norm):
            raise ValueError('Displacement norm mismatch')
        load_dot_seconds = time.monotonic()-phase
        phase = time.monotonic()
        child_score = evaluate(backend, score)
        score_seconds = time.monotonic()-phase
        # Seal candidate scores before its development likelihood is evaluated.
        scored = dict(candidate=index, alignment=alignment, gradient_norm=gnorm,
                      update_norm=norm, cosine=alignment/(gnorm*norm) if norm else None,
                      direct_lookahead=mean(child_score)-mean(parent['score']), score=child_score,
                      load_dot_seconds=load_dot_seconds, score_seconds=score_seconds)
        save(output/f'candidate-{index}-score.json', scored)
        phase = time.monotonic()
        child_dev = evaluate(backend, dev)
        scored.update(dev=child_dev, dev_change=mean(child_dev)-mean(parent['dev']),
                      dev_seconds=time.monotonic()-phase)
        save(output/f'candidate-{index}.json', scored)
        results.append(scored)
    backend.restore_parent()
    if evaluate(backend, [score[0]])[0] != replay:
        raise ValueError('Final parent restoration replay failed')
    summary = dict(results=results, parent_forward_seconds=forward_seconds,
                   gradient_seconds=gradient_seconds, elapsed_seconds=time.monotonic()-started,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                   interpretation='Surrogate development only; four mixed-hardware updates cannot establish transfer')
    save(output/'summary.json', summary)
    record['status'] = 'complete'
    save(output/'manifest.json', record)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(json.loads(a.config.read_text()), a.manifest, a.output)
