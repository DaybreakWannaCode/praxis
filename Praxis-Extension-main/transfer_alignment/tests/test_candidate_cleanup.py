import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from transfer_alignment.candidate_cleanup import release, sha

class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.run=self.base/'persistent';self.scratch=self.base/'scratch'
        self.run.mkdir();self.scratch.mkdir()
        self.actor=self.scratch/'exports/global_step_1/actor'
        self.actor.mkdir(parents=True)
        def put(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))
        self.put=put
        put(self.run/'plan.json',dict(scratch=str(self.scratch),parent_receipt_sha256='a'*64))
        put(self.scratch/'owner.json',dict(persistent=str(self.run),plan_sha256=sha(self.run/'plan.json')))
        put(self.run/'launcher.json',dict(status='complete',exit=0,tagged_processes_remaining=[]))
        (self.run/'run.exit').write_text('0\n')
        (self.run/'inputs').mkdir();(self.run/'inputs/fixed-update-input.pt').write_bytes(b'captured input')
        put(self.run/'inputs/capture.json',dict(status='updated',serialization_roundtrip_exact=True,capture_state_unchanged=True,
            before_steps=[16],after_steps=[17],parent_receipt_sha256='a'*64,input_file_sha256=sha(self.run/'inputs/fixed-update-input.pt')))
        cost=dict(status='complete',child_reconstruction_exact=True,parent_model='/parent/model.pt',optimizer_state_entries=1,incremented_entries=1,maximum_optimizer_step_increment=1)
        for rel,name,value in [('cost.json','cost.json',cost),('coordinates.json','coordinates.json',{}),('delta/manifest.json','delta-manifest.json',{'fixture':True})]:
            put(self.actor/rel,value);put(self.run/'export-receipt/byte-exact'/name,value)
        put(self.run/'scoring/summary.json',dict(status='complete',parent_replay_exact=True,validated_tensors=824,alignment=.1,direct_lookahead=.09))
        put(self.run/'scoring/local-audit.json',dict(status='passed'))
        put(self.run/'scoring/manifest.json',dict(status='complete',plan=dict(candidate_dir=str(self.actor),parent_model='/parent/model.pt',
            artifact_sha256={str(self.actor/'delta/manifest.json'):sha(self.actor/'delta/manifest.json')})))
    def tearDown(self):self.tmp.cleanup()
    def clean(self,live=False):return release(self.run,temporary_base=self.base,no_live_workers=lambda:not live)
    def test_release_retains_inputs_and_scores_and_is_idempotent(self):
        result=self.clean();self.assertEqual(result['status'],'complete')
        self.assertFalse(self.scratch.exists());self.assertTrue((self.run/'inputs/fixed-update-input.pt').exists())
        self.assertEqual(self.clean()['status'],'complete')
    def test_live_worker_prevents_release(self):
        with self.assertRaises(ValueError):self.clean(live=True)
        self.assertTrue(self.scratch.exists())
    def test_changed_input_prevents_release(self):
        (self.run/'inputs/fixed-update-input.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.clean()
        self.assertTrue(self.scratch.exists())
    def test_missing_score_audit_prevents_release(self):
        (self.run/'scoring/local-audit.json').unlink()
        with self.assertRaises(ValueError):self.clean()
    def test_other_candidate_score_prevents_release(self):
        p=self.run/'scoring/manifest.json';x=json.loads(p.read_text());x['plan']['candidate_dir']='/other/candidate';self.put(p,x)
        with self.assertRaises(ValueError):self.clean()
    def test_symlink_prevents_release(self):
        (self.scratch/'link').symlink_to(self.run/'plan.json')
        with self.assertRaises(ValueError):self.clean()
        self.assertTrue((self.run/'plan.json').exists())
    def test_interrupted_delete_can_resume_same_directory(self):
        with patch('transfer_alignment.candidate_cleanup.shutil.rmtree',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):self.clean()
        self.assertEqual(json.loads((self.run/'cleanup.json').read_text())['status'],'deleting')
        self.assertEqual(self.clean()['status'],'complete')
    def test_recreated_scratch_is_not_deleted(self):
        self.clean();self.scratch.mkdir();(self.scratch/'new-file').write_text('keep')
        with self.assertRaises(ValueError):self.clean()
        self.assertTrue((self.scratch/'new-file').exists())

if __name__=='__main__':unittest.main()
