"""Bounded greedy-generation timing on the existing audited development panel.

This is a throughput profile of the warm parent, not a baseline transfer result
or a substitute for the proposed broader scene evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from .choice_measure import save
from .experiment import response_record
from .production_precision import validate
from .production_visual import FullVisualBackend


def run(cfg, manifest, output, *, endpoint_model=None, endpoint_sha256=None):
    gates=[Path(p) for p in cfg['candidate_gates']]
    _,mapping,_,dev=validate(cfg,manifest,gates)
    if len(dev)!=32: raise ValueError('Only the existing 32-image development timing panel is permitted')
    if (endpoint_model is None)!=(endpoint_sha256 is None):
        raise ValueError('Endpoint timing requires both a model file and its frozen SHA-256')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    model_path=Path(cfg['parent_model'])
    hash_seconds=0.
    if endpoint_model is not None:
        model_path=Path(endpoint_model)
        phase=time.monotonic()
        h=hashlib.sha256()
        with model_path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
        hash_seconds=time.monotonic()-phase
        if h.hexdigest()!=endpoint_sha256:
            raise ValueError('Endpoint checkpoint hash differs')
    scope=('greedy throughput only, ordinary baseline endpoint, existing dev panel, no test data or training'
           if endpoint_model is not None else 'greedy throughput only, warm parent, no test data or training')
    record=dict(status='running',source_config=cfg,scope=scope,
                actual_model=str(model_path),endpoint_sha256=endpoint_sha256,
                checkpoint_hash_seconds=hash_seconds,
                count=32,replay_count=1,batch_size=1,do_sample=False,precision='FP32 parameters, BF16 autocast, FlashAttention2',
                text_prompt=cfg['system_prompt'],max_new_tokens=cfg['max_new_tokens'],image_max_pixels=cfg['max_pixels'])
    save(output/'manifest.json',record)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    backend=FullVisualBackend(cfg,model_path,mapping)
    backend.model.eval()
    torch.cuda.synchronize()
    startup=time.monotonic()-started
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    for i,item in enumerate(dev):
        torch.cuda.synchronize()
        tick=time.monotonic()
        response=backend.sample(item,'image',1,20260914+i,greedy=True)[0]
        torch.cuda.synchronize()
        elapsed=time.monotonic()-tick
        row=dict(response_record(response,item,20260914+i,'greedy_timing'),seconds=elapsed)
        with (output/'responses.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        rows.append(row)
        save(output/'progress.json',dict(completed_images=i+1,elapsed_seconds=time.monotonic()-started))
        del response
    replay=backend.sample(dev[0],'image',1,20260915,greedy=True)[0]
    if replay.payload[0].tolist()!=rows[0]['sequence_token_ids']:
        raise ValueError('Greedy token replay differs across seeds')
    torch.cuda.synchronize()
    seconds=sum(r['seconds'] for r in rows)
    tokens=sum(r['length'] for r in rows)
    summary=dict(count=len(rows),startup_seconds=startup,decode_seconds=seconds,
                 checkpoint_hash_seconds=hash_seconds,
                 elapsed_seconds=time.monotonic()-started,greedy_replay_exact=True,
                 responses_per_second=len(rows)/seconds,response_tokens_per_second=tokens/seconds,
                 response_tokens=tokens,correct=sum(r['correct'] for r in rows),parsed=sum(r['parsed'] for r in rows),
                 truncated=sum(r['truncated'] for r in rows),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                 scope=record['scope'],warning='One model and availability-limited panel; future checkpoints may have different response lengths. Not vLLM throughput.')
    save(output/'summary.json',summary)
    record['status']='complete';save(output/'manifest.json',record)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint-model',type=Path)
    p.add_argument('--endpoint-sha256')
    a=p.parse_args();run(json.loads(a.config.read_text()),a.manifest,a.output,
                        endpoint_model=a.endpoint_model,endpoint_sha256=a.endpoint_sha256)
