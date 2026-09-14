"""Opt-in hooks for a separately copied original trainer: restore, audit, exit."""
import json
import os
from pathlib import Path


def context():
    if os.environ.get('PRAXIS_RESTORE_ONLY') != '1':
        raise RuntimeError('Explicit restore-only opt-in required')
    return Path(os.environ['PRAXIS_RESTORE_RUN']), Path(os.environ['PRAXIS_RESTORE_PARENT'])


def install_manager(cls):
    context()
    original = cls.load_checkpoint
    def load(self, path=None):
        root, parent = context()
        if path is None or Path(path).resolve() != (parent/'actor').resolve():
            raise ValueError('Unexpected restore parent')
        original(self, path)
        from .loaded_checkpoint_audit import audit_manager
        audit_manager(self, path, root/'loaded-state.json')
    def forbidden(*args, **kwargs):
        raise RuntimeError('Restore-only check prohibits checkpoint saves')
    cls.load_checkpoint = load
    cls.save_checkpoint = forbidden


def install_trainer(cls):
    context()
    def fit(self):
        root, parent = context()
        if self.use_critic or self.config.trainer.n_gpus_per_node != 1 or self.config.trainer.nnodes != 1:
            raise ValueError('Only one-rank actor-only restoration is supported')
        if Path(self.config.trainer.load_checkpoint_path).resolve() != parent.resolve():
            raise ValueError('Trainer parent differs')
        self._load_checkpoint()
        receipt = json.loads((root/'loaded-state.json').read_text())
        if receipt['status'] != 'passed':
            raise ValueError('Loaded-state audit did not pass')
        with (root/'driver-complete.json').open('x') as stream:
            json.dump(dict(status='passed', restored_driver_step=self.global_step,
                scope='Original driver load and loaded-value audit; no training or generation'),stream,indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    cls.fit = fit
