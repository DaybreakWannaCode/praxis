"""Fresh selected-data arm from a warm actor, distinct from interrupted resume.

Restore model/optimizer/scheduler/worker RNG using the original worker interface.
Keep the newly constructed selected dataloader at its beginning; arm-local
driver counters start at zero. This is not a replacement for resume logic.
"""
import json
import os
from pathlib import Path
import types

from .selection_manifest import validate_training_handoff


def install_selection_warm_start(trainer, *, parent, manifest, selector, seed,
                                 ordered_prompt_ids, receipt_path):
    if trainer.use_critic:
        raise ValueError('Only actor-only selection supported')
    if '_selection_warm_start' in vars(trainer):
        raise ValueError('Fresh-arm restoration already installed')
    parent = Path(parent).resolve(strict=True)
    if not (parent / 'actor').is_dir():
        raise ValueError('Warm actor directory missing')
    if Path(trainer.config.trainer.load_checkpoint_path).resolve() != parent:
        raise ValueError('Configured parent differs')
    audit = validate_training_handoff(manifest, selector=selector, seed=seed,
        ordered_prompt_ids=ordered_prompt_ids, config=trainer.config)
    if trainer.training_steps != audit['batches']:
        raise ValueError('Effective arm training budget differs')
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        raise FileExistsError('Fresh-arm receipt exists; use explicit interrupted-run recovery')
    called = False

    def load(instance):
        nonlocal called
        if called or instance.global_step != 0:
            raise ValueError('Fresh-arm load may run only once at driver step zero')
        if (instance.use_critic or instance.training_steps != audit['batches']
                or Path(instance.config.trainer.load_checkpoint_path).resolve() != parent):
            raise ValueError('Warm-start contract changed')
        # Fail closed on an exception; do not retry a partly restored worker.
        called = True
        instance.actor_rollout_wg.load_checkpoint(str(parent / 'actor'))
        instance.global_step = 0
        # Deliberately never load the parent's dataloader.pt. The fresh dataset
        # must have been constructed from the independently sealed arm manifest.
        receipt = dict(status='loaded', parent=str(parent), arm_driver_step=0,
            selected_batches=audit['batches'], selector=selector, seed=seed,
            parent_dataloader_restored=False,
            scope='Original actor load returned; no loaded-value or rollout replay audit implied')
        with receipt_path.open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

    trainer._selection_warm_start = True
    trainer._load_checkpoint = types.MethodType(load, trainer)
