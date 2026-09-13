"""Opt-in, single-candidate cost hooks for a separate original-Praxis copy.

Training, rollout, reward and optimizer implementations stay unchanged. Only
driver checkpoint loading (fresh candidate data rather than old dataloader
position) and terminal export (displacement, not a resumable checkpoint) differ.
Never install these hooks into an ordinary training run.
"""
import json
import os
from pathlib import Path
import time


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_config(config):
    """Reject broader training or missing parent before any candidate work."""
    if config.trainer.max_steps != 1 or config.trainer.total_episodes != 1:
        raise ValueError('Cost check permits exactly one outer training step')
    if config.data.rollout_batch_size != 32 or config.worker.rollout.n != 5:
        raise ValueError('Expected 32 prompts and five responses per prompt')
    if config.worker.actor.global_batch_size != 32 or config.worker.actor.ppo_epochs != 1:
        raise ValueError('Expected exactly one actor minibatch and PPO epoch')
    if not config.trainer.load_checkpoint_path:
        raise ValueError('A saved, warm parent is required')


def install_trainer(trainer_class):
    if os.environ.get('PRAXIS_SINGLE_CANDIDATE_COST') != '1':
        raise RuntimeError('Candidate cost hooks need explicit opt-in')

    def load_parent(self):
        validate_config(self.config)
        if len(self.train_dataset) != 32 or len(self.train_dataloader) != 1:
            raise ValueError('Cost check requires exactly one fresh 32-prompt batch')
        parent = Path(self.config.trainer.load_checkpoint_path)
        expected = Path(os.environ['PRAXIS_CANDIDATE_PARENT']).resolve()
        if parent.resolve() != expected:
            raise ValueError('Driver and export parent paths differ')
        root = Path(os.environ['PRAXIS_CANDIDATE_COST_DIR'])
        started = time.monotonic()
        # Original worker restores model, Adam, scheduler and saved worker RNG.
        # Deliberately retain the NEW candidate dataloader, not the parent's cursor.
        self.actor_rollout_wg.load_checkpoint(str(parent/'actor'))
        self.global_step = 0
        save_json(root/'parent-restore.json', dict(
            seconds=time.monotonic()-started, parent=str(parent),
            driver_step_reset=0, dataloader='fresh fixed candidate batch',
            scope='Cost benchmark; not a validation of fresh rollout replay'))

    trainer_class._load_checkpoint = load_parent


def optimizer_steps(optimizer):
    # CPU scalar copies only; no second copy of Adam moments.
    return [int(state['step'].item() if hasattr(state['step'], 'item') else state['step'])
            for state in optimizer.state.values() if 'step' in state]


def install_checkpoint_manager(manager_class):
    if os.environ.get('PRAXIS_SINGLE_CANDIDATE_COST') != '1':
        raise RuntimeError('Candidate cost hooks need explicit opt-in')
    original_load = manager_class.load_checkpoint

    def load(self, path=None):
        expected = Path(os.environ['PRAXIS_CANDIDATE_PARENT'])/'actor'
        if path is None or Path(path).resolve() != expected.resolve():
            raise ValueError('Only the declared parent may be restored')
        original_load(self, path)
        self._cost_parent_steps = optimizer_steps(self.optimizer)
        if not self._cost_parent_steps or max(self._cost_parent_steps) <= 0:
            raise ValueError('Expected nonempty warm Adam state')
        if any(group['lr'] <= 0 for group in self.optimizer.param_groups):
            raise ValueError('Candidate must have nonzero next learning rate')

    def export(self, path):
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, ShardedStateDictConfig, StateDictType)
        from .production_displacement import save_displacement

        if dist.get_world_size() != 1:
            raise ValueError('Only a one-rank cost benchmark is supported')
        if not hasattr(self, '_cost_parent_steps') or getattr(self, '_cost_exported', False):
            raise ValueError('Export requires one parent restore and permits one child only')
        self._cost_exported = True
        root = Path(path)
        root_resolved = root.resolve()
        allowed = Path(os.environ['PRAXIS_CANDIDATE_COST_DIR']).resolve()
        if allowed not in root_resolved.parents:
            raise ValueError('Export is outside the private cost run directory')
        root.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        before_steps = self._cost_parent_steps
        after_steps = optimizer_steps(self.optimizer)
        if len(before_steps) != len(after_steps):
            raise ValueError('Adam state coverage changed; inspect before extrapolating')
        changes = [a-b for a,b in zip(after_steps, before_steps)]
        if not changes or max(changes) != 1 or any(d not in (0,1) for d in changes):
            raise ValueError('Expected one actual optimizer application')
        torch.cuda.synchronize()
        phase = time.monotonic()
        config = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, config):
            child = self.model.state_dict()
        torch.cuda.synchronize()
        materialize_seconds = time.monotonic()-phase
        phase = time.monotonic()
        parent_file = (Path(os.environ['PRAXIS_CANDIDATE_PARENT'])/'actor'/
                       'model_world_size_1_rank_0.pt')
        parent = torch.load(parent_file, map_location='cpu', mmap=True, weights_only=False)
        parent_open_seconds = time.monotonic()-phase
        # mmap page reads are charged to export, not the nearly free open above.
        phase = time.monotonic()
        report = save_displacement(parent, child, root/'delta', allow_float64=True)
        export_seconds = time.monotonic()-phase
        if report['update_norm'] <= 0:
            raise ValueError('Zero update is not a useful candidate cost check')
        save_json(root/'cost.json', dict(
            status='complete', parent_model=str(parent_file),
            materialize_seconds=materialize_seconds,
            parent_mmap_open_seconds=parent_open_seconds,
            lossless_export_seconds=export_seconds,
            total_seconds=time.monotonic()-started,
            export_bytes=sum(row['compressed_bytes'] for row in report['parameters']),
            canonical_numel=report['canonical_numel'], update_norm=report['update_norm'],
            child_reconstruction_exact=report['child_reconstruction_exact'],
            optimizer_state_entries=len(changes),
            incremented_entries=sum(d == 1 for d in changes),
            maximum_optimizer_step_increment=max(changes),
            artifact='Lossless model-state displacement only; NOT a resumable checkpoint',
            limitations='One fresh candidate; no visual scoring or selection claim. '
                        'Includes parent mmap page reads, precision checks and compression.'))

    manager_class.load_checkpoint = load
    manager_class.save_checkpoint = export
