"""Bind real serialization, pinned validation and quota to an owned trainer.

Explicit caller entrypoint; importing this module neither patches nor launches
the original trainer. Counter expectations are sealed per save step, rather than
assuming a resumed driver's step is also its optimizer counter.
"""
import hashlib
import json
from pathlib import Path

import torch

from .checkpoint_quota import CheckpointQuota
from .original_checkpoint_adapter import OriginalCheckpointAdapter
from .original_checkpoint_validation import validate_checkpoint


def install_checkpointing(trainer, store, *, schema_path, schema_sha256,
                          expectations, volume, quota_bytes, replacement_bytes,
                          reserve_bytes, tied_aliases=()):
    config = trainer.config.trainer
    if config.n_gpus_per_node != 1 or config.nnodes != 1:
        raise ValueError('Pinned checkpoint schema supports one rank only')
    raw = Path(schema_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != schema_sha256:
        raise ValueError('Pinned checkpoint schema hash differs')
    schema = json.loads(raw)
    if schema.get('version') != 1:
        raise ValueError('Unknown checkpoint schema')
    # Freeze caller-owned mutable structures before installing callbacks.
    schedule = json.loads(json.dumps(expectations, allow_nan=False))
    aliases = tuple(tuple(pair) for pair in tied_aliases)
    if not schedule:
        raise ValueError('Explicit per-save expectations required')
    for step, values in schedule.items():
        if not step.isdecimal() or str(int(step)) != step:
            raise ValueError('Expected canonical nonnegative save-step keys')
        if set(values) != {'optimizer_step', 'scheduler_epoch', 'expected_lr'}:
            raise ValueError('Explicit optimizer, scheduler and learning-rate expectations required')
        if (type(values['optimizer_step']) is not int or values['optimizer_step'] < 1
                or type(values['scheduler_epoch']) is not int
                or type(values['expected_lr']) not in (int, float)
                or values['expected_lr'] <= 0):
            raise ValueError('Invalid checkpoint expectations')
    quota = CheckpointQuota(volume, quota_bytes=quota_bytes,
        replacement_bytes=replacement_bytes, reserve_bytes=reserve_bytes)

    def preflight(root, step):
        if str(step) not in schedule:
            raise ValueError('Save step absent from sealed expectations')
        return quota(root, step)

    def validate(root, step):
        result = validate_checkpoint(root, schema, **schedule[str(step)], tied_aliases=aliases)
        result['schema_sha256'] = schema_sha256
        return result

    adapter = OriginalCheckpointAdapter(trainer, store, validate=validate,
        preflight=preflight, save_state=torch.save)
    adapter.install()
    return adapter
