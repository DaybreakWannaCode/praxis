"""Opt-in save adapter for an owned, single-writer original Praxis GRPO run.

Does not install itself or change frozen trainers. Caller supplies a semantic
checkpoint validator and quota preflight; neither is replaced by file existence.
Real distributed-worker integration and crash recovery require separate validation.
"""
import fcntl
import os
from pathlib import Path
import types

from .checkpoint_publication import CheckpointStore, sync_directory


class OriginalCheckpointAdapter:
    def __init__(self, trainer, store, *, validate, preflight, save_state):
        if trainer.use_critic:
            raise ValueError('Only actor-only GRPO is supported')
        if not isinstance(store, CheckpointStore):
            raise TypeError('An explicitly owned CheckpointStore is required')
        if Path(trainer.config.trainer.save_checkpoint_path).resolve() != store.root.resolve():
            raise ValueError('Trainer save path differs from owned store')
        if not all(callable(f) for f in (validate, preflight, save_state)):
            raise TypeError('Validation, quota preflight and serialization are required')
        self.trainer, self.store = trainer, store
        self.validate, self.preflight, self.save_state = validate, preflight, save_state

    def save(self):
        t = self.trainer
        if t.use_critic or Path(t.config.trainer.save_checkpoint_path).resolve() != self.store.root.resolve():
            raise ValueError('Trainer checkpoint contract changed')
        # Lock covers quota inspection, worker saves, publication and retirement.
        with (self.store.root / 'writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.preflight(self.store.root, t.global_step)

            def write(pending):
                t.actor_rollout_wg.save_checkpoint(str(pending / 'actor'))
                self.save_state(t.train_dataloader.state_dict(), pending / 'dataloader.pt')

            def validate(pending):
                if not (pending / 'actor').is_dir() or not (pending / 'dataloader.pt').is_file():
                    raise ValueError('Original continuation files missing')
                return self.validate(pending, t.global_step)

            def tracker(destination):
                pointer = self.store.root / 'latest_global_step.txt'
                pending = pointer.with_name(pointer.name + '.pending')
                with pending.open('w') as stream:
                    stream.write(str(t.global_step))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(pending, pointer)
                sync_directory(self.store.root)

            return self.store.publish(t.global_step, write, validate, before_retire=tracker)

    def install(self):
        """Explicit per-instance replacement; the original class is untouched."""
        if '_alignment_checkpoint_adapter' in vars(self.trainer):
            raise ValueError('Checkpoint adapter already installed')
        original = self.trainer._save_checkpoint
        self.trainer._alignment_checkpoint_adapter = self
        self.trainer._save_checkpoint = types.MethodType(lambda instance: self.save(), self.trainer)
        return original
