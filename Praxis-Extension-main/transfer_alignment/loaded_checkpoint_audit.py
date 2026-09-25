"""Exact loaded-state comparison; no model or optimizer checkpoint output.

The manager audit must run immediately after the original load, before updates.
A passed comparison proves restoration equality, not subsequent rollout replay.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch


def compare_state(actual, expected, path='state'):
    """Compare nested states with at most one tensor chunk transferred to CPU."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f'{path}: tensor shape/type mismatch')
        if not actual.is_contiguous() or not expected.is_contiguous():
            raise ValueError(f'{path}: unexpected noncontiguous tensor')
        a, b = actual.detach().view(-1), expected.detach().view(-1)
        for offset in range(0, b.numel(), 1024*1024):
            left = a[offset:offset+1024*1024].cpu()
            right = b[offset:offset+1024*1024].cpu()
            if not torch.equal(left, right):
                raise ValueError(f'{path}: tensor values differ at chunk {offset}')
        return expected.numel()
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f'{path}: NumPy state differs')
        return 0
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'{path}: dictionary keys differ')
        return sum(compare_state(actual[k], v, f'{path}.{k}') for k,v in expected.items())
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f'{path}: sequence differs')
        return sum(compare_state(a,b,f'{path}[{i}]') for i,(a,b) in enumerate(zip(actual,expected)))
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f'{path}: scalar differs')
    return 0


def audit_manager(manager, actor_path, output):
    """Original single-rank FSDP manager, already loaded; output must be new."""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardedStateDictConfig, StateDictType
    if dist.get_world_size() != 1:
        raise ValueError('Loaded-state audit supports one rank only')
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    started = time.monotonic()
    actor = Path(actor_path)
    def load(kind):
        return torch.load(actor/f'{kind}_world_size_1_rank_0.pt', map_location='cpu', mmap=True, weights_only=False)
    # Compare RNG first, before any materialization work can affect it.
    extra = load('extra_state')
    compare_state(manager.get_rng_state(), extra['rng'], 'rng')
    compare_state(manager.lr_scheduler.state_dict(), extra['lr_scheduler'], 'scheduler')
    del extra
    with FSDP.state_dict_type(manager.model, StateDictType.SHARDED_STATE_DICT,
                             ShardedStateDictConfig(offload_to_cpu=True)):
        actual = manager.model.state_dict()
    expected = load('model')
    model_elements = compare_state(actual, expected, 'model')
    model_tensors = len(expected)
    del actual, expected
    expected = load('optim')
    optimizer_elements = compare_state(manager.optimizer.state_dict(), expected, 'optimizer')
    populated = sum(bool(s) for s in expected['state'].values())
    del expected
    result = dict(status='passed', model_tensors=model_tensors,
        model_elements_including_aliases=model_elements, optimizer_tensor_elements=optimizer_elements,
        populated_optimizer_states=populated, rng_exact=True, scheduler_exact=True,
        elapsed_seconds=time.monotonic()-started, actor_path=str(actor),
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All loaded model/Adam/scheduler/RNG values equal saved state; not an update or rollout replay')
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    return result
