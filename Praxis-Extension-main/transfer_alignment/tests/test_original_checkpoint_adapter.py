import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from transfer_alignment.checkpoint_publication import CheckpointStore
from transfer_alignment.original_checkpoint_adapter import OriginalCheckpointAdapter


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = CheckpointStore(Path(self.temp.name) / 'checkpoints', create=True)
        self.calls = []
        def actor(path):
            self.calls.append('actor')
            p = Path(path); p.mkdir()
            (p / 'model.pt').write_bytes(b'model')
            (p / 'optimizer.pt').write_bytes(b'optimizer')
        self.trainer = SimpleNamespace(use_critic=False, global_step=1,
            config=SimpleNamespace(trainer=SimpleNamespace(save_checkpoint_path=str(self.store.root))),
            actor_rollout_wg=SimpleNamespace(save_checkpoint=actor),
            train_dataloader=SimpleNamespace(state_dict=lambda: {'position': 32}),
            _save_checkpoint=lambda: self.fail('Old delete-before-save path called'))
        def validate(path, step):
            self.calls.append('validate')
            self.assertEqual(json.loads((path / 'dataloader.pt').read_text()), {'position': 32})
            return {'status': 'passed', 'test_fixture_only': True}
        self.adapter = OriginalCheckpointAdapter(self.trainer, self.store,
            validate=validate, preflight=lambda *args: self.calls.append('preflight'),
            save_state=lambda state, path: path.write_text(json.dumps(state)))
        self.adapter.install()

    def tearDown(self): self.temp.cleanup()

    def test_actual_save_entrypoint_publishes_compatible_layout(self):
        self.trainer._save_checkpoint()
        self.assertEqual(self.calls, ['preflight', 'actor', 'validate'])
        self.trainer.global_step = 2
        self.trainer._save_checkpoint()
        self.assertEqual((self.store.root / 'latest_global_step.txt').read_text(), '2')
        self.assertTrue((self.store.root / 'global_step_2/actor/optimizer.pt').exists())
        self.assertFalse((self.store.root / 'global_step_1').exists())

    def test_failed_validation_keeps_previous_resume_target(self):
        self.trainer._save_checkpoint(); self.trainer.global_step = 2
        self.adapter.validate = lambda *args: {'status': 'failed'}
        with self.assertRaises(ValueError): self.trainer._save_checkpoint()
        self.assertEqual((self.store.root / 'latest_global_step.txt').read_text(), '1')
        self.assertTrue((self.store.root / 'global_step_1/actor/model.pt').exists())

    def test_tracker_failure_keeps_both_complete_checkpoints(self):
        self.trainer._save_checkpoint(); self.trainer.global_step = 2
        import transfer_alignment.original_checkpoint_adapter as module
        replace = module.os.replace
        def fail(source, dest):
            if Path(dest).name == 'latest_global_step.txt': raise OSError('disk failure')
            return replace(source, dest)
        with patch.object(module.os, 'replace', fail), self.assertRaises(OSError):
            self.trainer._save_checkpoint()
        self.assertEqual((self.store.root / 'latest_global_step.txt').read_text(), '1')
        for step in (1, 2):
            self.assertTrue((self.store.root / f'global_step_{step}/actor/model.pt').exists())

    def test_quota_failure_prevents_worker_call(self):
        def fail(*args): raise OSError('quota')
        self.adapter.preflight = fail
        with self.assertRaises(OSError): self.trainer._save_checkpoint()
        self.assertEqual(self.calls, [])

    def test_concurrent_writer_refused(self):
        with (self.store.root / 'writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): self.trainer._save_checkpoint()
        self.assertEqual(self.calls, [])

    def test_contract_change_and_duplicate_install_refused(self):
        with self.assertRaises(ValueError): self.adapter.install()
        self.trainer.use_critic = True
        with self.assertRaises(ValueError): self.trainer._save_checkpoint()


if __name__ == '__main__': unittest.main()
