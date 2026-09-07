import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transfer_alignment.production_precision import validate


class PrecisionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.manifest=root/'visual.json'
        self.manifest.write_text('[{"split":"score"},{"split":"dev"}]')
        audit=root/'audit.json'
        audit.write_text('{}')
        self.gates=[]
        report={'status':'passed','equal':{'parameters':True},'canonical_validation':{},
                'parent_digests':dict.fromkeys(('parameters','buffers','optimizer','scheduler'),'same')}
        for index in range(4):
            gate=root/str(index)
            gate.mkdir()
            (gate/'parity.json').write_text(json.dumps(report))
            (gate/'coordinates.json').write_text('{}')
            self.gates.append(gate)
        self.cfg={'probe_repeats':4,'probe_samples':4,'outcome_samples':8,'horizon':1,
                  'scene_audit':str(audit),'scene_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),
                  'visual_manifest_sha256':hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                  'candidate_gates':list(map(str,self.gates)),'decision_rule':'fixed precision threshold',
                  'score_ids':['s'],'dev_ids':['d'],'response_budget':72}
        self.items=[SimpleNamespace(id='s',split='score',group_id='scene-s'),
                    SimpleNamespace(id='d',split='dev',group_id='scene-d')]
        self.cfg['image_sha256']={}
        for item in self.items:
            path=root/(item.id+'.jpg')
            path.write_bytes(item.id.encode())
            item.image_path=str(path)
            self.cfg['image_sha256'][item.id]=hashlib.sha256(path.read_bytes()).hexdigest()

    def check(self):
        with patch('transfer_alignment.production_precision.load_manifest',return_value=self.items):
            return validate(self.cfg,self.manifest,self.gates)

    def test_locked_valid_budget(self):
        self.assertEqual(len(self.check()[0]),4)

    def test_changed_budget_rejected(self):
        self.cfg['response_budget']+=1
        with self.assertRaisesRegex(ValueError,'budget'):self.check()

    def test_changed_manifest_rejected_before_loading(self):
        self.manifest.write_text('[]')
        with self.assertRaisesRegex(ValueError,'changed'):self.check()

    def test_scene_overlap_rejected(self):
        self.items[1].group_id='scene-s'
        with self.assertRaisesRegex(ValueError,'families'):self.check()

    def test_changed_image_rejected(self):
        Path(self.items[0].image_path).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'bytes'):self.check()

    def test_horizon_cannot_be_relabelled(self):
        self.cfg['horizon']=4
        with self.assertRaisesRegex(ValueError,'not implemented'):self.check()

    def test_different_optimizer_parent_rejected(self):
        path=self.gates[-1]/'parity.json'
        report=json.loads(path.read_text())
        report['parent_digests']['optimizer']='different'
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'complete parent'):self.check()


if __name__=='__main__':unittest.main()
