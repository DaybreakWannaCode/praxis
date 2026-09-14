import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from transfer_alignment.checkpoint_publication import CheckpointStore
from transfer_alignment.checkpoint_recovery import inspect_recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=CheckpointStore(Path(self.tmp.name)/'checkpoints',create=True)
        (self.store.root/'writer.lock').touch()
        self.store.publish(1,lambda p:(p/'state').write_bytes(b'one'),lambda p:{'status':'passed'})
        (self.store.root/'latest_global_step.txt').write_text('1')
    def tearDown(self):self.tmp.cleanup()

    def test_consistent_and_corrupt_file(self):
        self.assertTrue(inspect_recovery(self.store.root)['automatic_resume_allowed'])
        (self.store.root/'global_step_1/state').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'hash'):inspect_recovery(self.store.root)

    def test_tracker_failure_requires_reconciliation_and_keeps_both(self):
        def fail(p):raise OSError('tracker')
        with self.assertRaises(OSError):
            self.store.publish(2,lambda p:(p/'state').write_bytes(b'two'),lambda p:{'status':'passed'},before_retire=fail)
        result=inspect_recovery(self.store.root)
        self.assertEqual(result['status'],'tracker_reconciliation_required')
        self.assertFalse(result['automatic_resume_allowed'])
        self.assertTrue((self.store.root/'global_step_1/state').exists())
        self.assertEqual((self.store.root/'latest_global_step.txt').read_text(),'1')

    def test_active_writer_and_extra_file_rejected(self):
        with (self.store.root/'writer.lock').open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):inspect_recovery(self.store.root)
        (self.store.root/'global_step_1/extra').write_text('unexpected')
        with self.assertRaisesRegex(ValueError,'coverage'):inspect_recovery(self.store.root)

    def test_tracker_ahead_rejected(self):
        (self.store.root/'latest_global_step.txt').write_text('2')
        with self.assertRaisesRegex(ValueError,'ahead'):inspect_recovery(self.store.root)


if __name__=='__main__':unittest.main()
