"""Read-only single-rank Praxis checkpoint validation against a pinned schema.

Load only trusted project checkpoints: the original format uses Python pickle.
Schema capture records structure, not validity; validate against a separately
approved reference. A passing report is not a real worker resume/replay test.
"""
import json
import math
import random
from pathlib import Path
import numpy as np
import torch


def _load(path):
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def _tensor_spec(value):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError('Expected dense single-rank tensor')
    return {'shape': list(value.shape), 'dtype': str(value.dtype)}


def _finite(value):
    if isinstance(value, torch.Tensor):
        if not value.is_contiguous():
            raise ValueError('Unexpected noncontiguous saved tensor')
        flat = value.view(-1)
        for part in flat.split(1024 * 1024):
            if not torch.isfinite(part).all().item():
                raise ValueError('Nonfinite checkpoint tensor')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite checkpoint scalar')
    elif isinstance(value, dict):
        for item in value.values(): _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _finite(item)


def capture_schema(root):
    """Return structure for external review/pinning; does not certify a checkpoint."""
    actor = Path(root) / 'actor'
    model = _load(actor / 'model_world_size_1_rank_0.pt')
    model_spec = {key: _tensor_spec(value) for key, value in model.items()}
    del model
    optimizer = _load(actor / 'optim_world_size_1_rank_0.pt')
    state_spec = {str(key): {name: _tensor_spec(value) for name, value in state.items()}
                  for key, state in optimizer['state'].items()}
    groups = [{k: v for k, v in group.items() if k not in ('lr', 'initial_lr')}
              for group in optimizer['param_groups']]
    groups = json.loads(json.dumps(groups, allow_nan=False))
    return {'version': 1, 'model': model_spec, 'optimizer_state': state_spec,
            'optimizer_groups': groups}


def validate_checkpoint(root, schema, *, optimizer_step, scheduler_epoch, expected_lr,
                        tied_aliases=()):
    if schema.get('version') != 1:
        raise ValueError('Unknown checkpoint schema')
    if type(optimizer_step) is not int or optimizer_step < 1 or type(scheduler_epoch) is not int:
        raise ValueError('Explicit optimizer and scheduler counters required')
    if not math.isfinite(expected_lr) or expected_lr <= 0:
        raise ValueError('A positive next-update learning rate is required')
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symlink in checkpoint')
    actor = root / 'actor'
    expected_names = {f'{kind}_world_size_1_rank_0.pt' for kind in ('model', 'optim', 'extra_state')}
    if {p.name for p in actor.glob('*_world_size_*_rank_*.pt')} != expected_names:
        raise ValueError('Unexpected checkpoint rank topology')
    model = _load(actor / 'model_world_size_1_rank_0.pt')
    if {k: _tensor_spec(v) for k, v in model.items()} != schema['model']:
        raise ValueError('Model coordinates or dtype changed')
    for value in model.values(): _finite(value)
    for alias, canonical in tied_aliases:
        if alias not in model or canonical not in model or not torch.equal(model[alias], model[canonical]):
            raise ValueError('Tied model aliases differ')
    del model
    optimizer = _load(actor / 'optim_world_size_1_rank_0.pt')
    specs = {str(k): {n: _tensor_spec(v) for n, v in s.items()} for k, s in optimizer['state'].items()}
    if specs != schema['optimizer_state']:
        raise ValueError('Optimizer state layout changed')
    groups = [{k: v for k, v in group.items() if k not in ('lr', 'initial_lr')}
              for group in optimizer['param_groups']]
    groups = json.loads(json.dumps(groups, allow_nan=False))
    if groups != schema['optimizer_groups']:
        raise ValueError('Optimizer parameter mapping or hyperparameters changed')
    populated = 0
    for state in optimizer['state'].values():
        if not state: continue
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Expected Adam state')
        if state['step'].numel() != 1 or float(state['step']) != optimizer_step:
            raise ValueError('Wrong optimizer counter')
        if state['exp_avg'].shape != state['exp_avg_sq'].shape:
            raise ValueError('Adam moment shapes differ')
        _finite(state)
        for part in state['exp_avg_sq'].view(-1).split(1024 * 1024):
            if (part < 0).any().item(): raise ValueError('Negative Adam second moment')
        populated += 1
    if not populated: raise ValueError('No populated optimizer states')
    if any(g['lr'] != expected_lr for g in optimizer['param_groups']):
        raise ValueError('Next optimizer learning rate differs')
    del optimizer
    extra = _load(actor / 'extra_state_world_size_1_rank_0.pt')
    scheduler = extra['lr_scheduler']
    if scheduler['last_epoch'] != scheduler_epoch or scheduler['_step_count'] != scheduler_epoch + 1:
        raise ValueError('Wrong scheduler counter')
    if scheduler['_last_lr'] != [expected_lr] * len(groups):
        raise ValueError('Scheduler and optimizer learning rates differ')
    _finite(scheduler)
    rng = extra['rng']
    if set(rng) != {'cpu', 'cuda', 'numpy', 'random'}:
        raise ValueError('Incomplete worker RNG')
    torch.Generator(device='cpu').set_state(rng['cpu'])
    np.random.RandomState().set_state(rng['numpy'])
    random.Random().setstate(rng['random'])
    # CPU inspection can check CUDA state structure, not runtime compatibility.
    cuda = rng['cuda']
    if not isinstance(cuda, torch.Tensor) or cuda.dtype != torch.uint8 or cuda.ndim != 1 or not cuda.numel():
        raise ValueError('Invalid CUDA RNG structure')
    loader = _load(root / 'dataloader.pt')
    if set(loader) != {'_snapshot', '_steps_since_snapshot', '_iterator_finished'}:
        raise ValueError('Unexpected dataloader schema')
    if type(loader['_steps_since_snapshot']) is not int or loader['_steps_since_snapshot'] < 0:
        raise ValueError('Invalid dataloader counter')
    if type(loader['_iterator_finished']) is not bool or not isinstance(loader['_snapshot'], dict):
        raise ValueError('Invalid dataloader snapshot')
    needed = {'_snapshot_step', '_last_yielded_worker_id', '_main_snapshot', '_worker_snapshots'}
    if not needed.issubset(loader['_snapshot']):
        raise ValueError('Incomplete dataloader snapshot')
    return {'status': 'passed', 'model_tensors': len(schema['model']),
            'optimizer_state_entries': len(specs), 'populated_optimizer_states': populated,
            'optimizer_step': optimizer_step, 'scheduler_epoch': scheduler_epoch,
            'scope': 'CPU structural and finite-state validation against pinned schema; '
                     'CPU/NumPy/Python RNG loadability checked on isolated generators; '
                     'CUDA RNG and real worker/dataloader resume remain untested'}
