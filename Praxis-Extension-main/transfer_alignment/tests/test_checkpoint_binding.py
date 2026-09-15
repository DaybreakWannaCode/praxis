import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from transfer_alignment.checkpoint_binding import install_checkpointing
from transfer_alignment.checkpoint_publication import CheckpointStore


class BindingTests(unittest.TestCase):
    def test_bound_serialization_and_sealed_counter_expectations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema = root / 'schema.json'
            schema.write_text('{"version": 1}')
            store = CheckpointStore(root / 'checkpoints', create=True)
            def save_actor(path):
                Path(path).mkdir()
                (Path(path) / 'fixture.pt').write_bytes(b'fixture')
            trainer = NS(use_critic=False, global_step=20,
                config=NS(trainer=NS(n_gpus_per_node=1, nnodes=1,
                    save_checkpoint_path=str(store.root))),
                actor_rollout_wg=NS(save_checkpoint=save_actor),
                train_dataloader=NS(state_dict=lambda: {'position': torch.tensor([4])}),
                _save_checkpoint=lambda: None)
            expectations = {'20': dict(optimizer_step=19, scheduler_epoch=19, expected_lr=1e-6)}
            args = dict(schema_path=schema, schema_sha256=hashlib.sha256(schema.read_bytes()).hexdigest(),
                expectations=expectations, volume=root, quota_bytes=1000000,
                replacement_bytes=100000, reserve_bytes=100000)
            with self.assertRaises(ValueError):
                install_checkpointing(trainer, store, **{**args, 'schema_sha256': '0' * 64})
            self.assertFalse(hasattr(trainer, '_alignment_checkpoint_adapter'))
            install_checkpointing(trainer, store, **args)
            expectations['20']['optimizer_step'] = 999
            with patch('transfer_alignment.checkpoint_binding.validate_checkpoint',
                       return_value={'status': 'passed', 'fixture_only': True}) as validator:
                trainer._save_checkpoint()
                self.assertEqual(validator.call_args.kwargs['optimizer_step'], 19)
            state = torch.load(store.root / 'global_step_20/dataloader.pt', weights_only=True)
            self.assertEqual(state['position'].tolist(), [4])
            trainer.global_step = 21
            with self.assertRaisesRegex(ValueError, 'absent'):
                trainer._save_checkpoint()
            self.assertTrue((store.root / 'global_step_20').exists())
            self.assertFalse((store.root / 'global_step_21').exists())


if __name__ == '__main__':
    unittest.main()
