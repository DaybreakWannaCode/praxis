"""Tiny real-CUDA check using the original manager and saved worker RNG state."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

root = Path('/workspace/praxis-rng-restore-20260915')
root.mkdir(exist_ok=False)
started = time.monotonic()
source = Path('/workspace/praxis/runs/ordinary-baseline-20260914/checkpoints/global_step_32/actor/extra_state_world_size_1_rank_0.pt')
raw = source.read_bytes()
result = {'status': 'running', 'checkpoint_extra_sha256': hashlib.sha256(raw).hexdigest()}
(root/'started.json').write_text(json.dumps(result, indent=2)+'\n')
try:
    torch.cuda.set_device(0)
    saved = torch.load(source, map_location='cpu', weights_only=False)['rng']
    original = BaseCheckpointManager.get_rng_state()
    def draw():
        return (torch.rand(1024), torch.rand(1024, device='cuda').cpu(),
                np.random.random(1024), [random.random() for _ in range(64)])
    def equal(a, b):
        return [torch.equal(a[0], b[0]), torch.equal(a[1], b[1]),
                np.array_equal(a[2], b[2]), a[3] == b[3]]
    try:
        BaseCheckpointManager.load_rng_state(saved)
        first = draw()
        advanced = draw()
        BaseCheckpointManager.load_rng_state(saved)
        replay = draw()
        exact = equal(first, replay)
        changed = [not item for item in equal(first, advanced)]
        if not all(exact) or not all(changed):
            raise ValueError('RNG replay or advance control failed')
    finally:
        BaseCheckpointManager.load_rng_state(original)
    manager = Path(inspect.getfile(BaseCheckpointManager))
    result.update(status='passed', generators=['torch_cpu', 'torch_cuda', 'numpy', 'python'],
        replay_exact=exact, advance_control_changed=changed,
        original_manager_file=str(manager), manager_sha256=hashlib.sha256(manager.read_bytes()).hexdigest(),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Original manager RNG restoration on saved step32 state; not full model/optimizer/dataloader resume or rollout replay')
except Exception as exc:
    result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
result['elapsed_seconds'] = time.monotonic()-started
result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with (root/'result.json').open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['status']=='passed' else 1)
