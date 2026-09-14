import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from transfer_alignment.checkpoint_publication import CheckpointStore


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=CheckpointStore(Path(self.temp.name)/'checkpoints',create=True)
        self.write=lambda p:(p/'state').write_bytes(b'checkpoint')
        self.valid=lambda p:{'status':'passed'}
        self.store.publish(1,self.write,self.valid)

    def tearDown(self):self.temp.cleanup()

    def test_success_retires_only_after_validation(self):
        old=self.store.root/'global_step_1'
        def validate(p):
            self.assertTrue(old.exists())
            self.assertEqual(json.loads((self.store.root/'latest.json').read_text())['step'],1)
            return {'status':'passed'}
        result=self.store.publish(2,self.write,validate)
        self.assertEqual(result['step'],2)
        self.assertFalse(old.exists())
        self.assertTrue((self.store.root/'global_step_2/state').exists())

    def test_failed_validation_preserves_previous(self):
        with self.assertRaises(ValueError):self.store.publish(2,self.write,lambda p:{'status':'failed'})
        self.assertTrue((self.store.root/'global_step_1/state').exists())
        self.assertEqual(json.loads((self.store.root/'latest.json').read_text())['step'],1)

    def test_writer_failure_preserves_previous(self):
        def fail(p):raise OSError('disk full')
        with self.assertRaises(OSError):self.store.publish(2,fail,self.valid)
        self.assertTrue((self.store.root/'global_step_1/state').exists())

    def test_pointer_failure_preserves_both_complete_copies(self):
        import transfer_alignment.checkpoint_publication as module
        original=module.atomic_json
        def fail_pointer(path,value):
            if path.name=='latest.json':raise OSError('pointer write failed')
            return original(path,value)
        with patch.object(module,'atomic_json',fail_pointer),self.assertRaises(OSError):
            self.store.publish(2,self.write,self.valid)
        self.assertTrue((self.store.root/'global_step_1/state').exists())
        self.assertTrue((self.store.root/'global_step_2/state').exists())

    def test_cannot_adopt_existing_directory(self):
        with self.assertRaises(FileExistsError):CheckpointStore(self.store.root,create=True)

    def test_refuses_symlink_input_and_preserves_external_file(self):
        external=Path(self.temp.name)/'external';external.write_text('keep')
        def write(p):(p/'state').symlink_to(external)
        with self.assertRaises(ValueError):self.store.publish(2,write,self.valid)
        self.assertEqual(external.read_text(),'keep')


if __name__=='__main__':unittest.main()
