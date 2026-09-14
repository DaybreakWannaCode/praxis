"""Compare real baseline loader continuation with its saved halfway state; CPU only."""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoProcessor
from verl.trainer.config import PPOConfig
from verl.trainer.ray_trainer import RayPPOTrainer


def digest(batch):
    h = hashlib.sha256()
    for key in sorted(batch):
        h.update(key.encode() + b'\0')
        value = batch[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            h.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            h.update(value.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            h.update(json.dumps(value.tolist(), sort_keys=True, allow_nan=False).encode())
        else:
            raise TypeError(f'Unexpected batch field type for {key}: {type(value)}')
    return h.hexdigest()


def main():
    root = Path('/workspace/praxis-dataloader-restore-20260915')
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    base = Path('/workspace/praxis/runs/ordinary-baseline-20260914')
    config = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load(base/'config.yaml'))
    audit = json.loads(Path('/workspace/praxis/data/ordinary-baseline-20260914/audit.json').read_text())
    train_path = Path(config.data.train_files)
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    if train_sha != audit['output_sha256']['train.parquet']:
        raise ValueError('Training input differs from original audit')
    model = config.worker.actor.model.model_path
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    def create():
        stub = SimpleNamespace(config=OmegaConf.create(OmegaConf.to_container(config)),
                               tokenizer=tokenizer, processor=processor)
        RayPPOTrainer._create_dataloader(stub)
        return stub.train_dataloader
    loader = create()
    uninterrupted = [digest(batch) for batch in loader]
    if len(uninterrupted) != 32:
        raise ValueError('Expected 32 baseline batches')
    state_file = base/'checkpoints/global_step_16/dataloader.pt'
    state = torch.load(state_file, map_location='cpu', weights_only=False)
    restored = create()
    restored.load_state_dict(state)
    continuation = [digest(batch) for batch in restored]
    if continuation != uninterrupted[16:]:
        raise ValueError('Restored continuation differs from uninterrupted final 16 batches')
    # A new loader without restoration must start at batch 1, not batch 17.
    if uninterrupted[:16] == continuation:
        raise ValueError('Uninformative repeated-prefix control')
    result = dict(status='passed', full_batches=len(uninterrupted), restored_batches=len(continuation),
        continuation_exact=True, fresh_prefix_differs=True, uninterrupted_sha256=uninterrupted,
        restored_sha256=continuation, train_sha256=train_sha,
        dataloader_state_sha256=hashlib.sha256(state_file.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__, elapsed_seconds=time.monotonic()-started,
        scope='Actual original loader construction and saved step16 state on original text data; '
              'batch tensor/metadata digests match uninterrupted continuation; no model/optimizer/rollout replay')
    with (root/'result.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)


if __name__ == '__main__': main()
