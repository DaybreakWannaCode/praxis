import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from transfer_alignment.baseline_endpoints import run, sha


class EndpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        model=self.root/'model';model.mkdir();weight=model/'weights';weight.write_bytes(b'pinned initial')
        final=self.root/'final.pt';final.write_bytes(b'final')
        coordinates=self.root/'coordinates.json';coordinates.write_text('{}')
        item=lambda i:dict(id=f'image-{i}',group_id=f'scene-{i}',split='dev',question='Q?',action_list=['A. yes','B. no'],answer='A')
        panel=self.root/'panel.json';panel.write_text(json.dumps(dict(items=[item(i) for i in range(256)],shuffled_controls=[item(i+256) for i in range(32)],image_sha256={})))
        self.plan=self.root/'plan.json'
        self.plan.write_text(json.dumps(dict(panel=str(panel),coordinates=str(coordinates),model_files=[str(weight)],
            endpoints={'initial':{'kind':'pinned_pretrained'},'final':{'kind':'checkpoint','path':str(final)}},
            artifact_sha256={str(p):sha(p) for p in [panel,coordinates,weight,final]},seed=42,scope='test fixture',
            visual_config=dict(model=str(model),max_new_tokens=512,answer_parser='explicit_final_v3'))))
        self.output=self.root/'output'
    def tearDown(self):self.tmp.cleanup()
    def backend(self, fail_on=None):
        calls=[];initializations=[]
        class FakeBackend:
            def __init__(inner,cfg,parent,mapping,**kw):
                initializations.append((parent,kw));inner.model=SimpleNamespace(eval=lambda:None);inner.metadata={}
            def sample(inner,item,modality,count,seed,greedy=False):
                calls.append(item.id)
                if len(calls)==fail_on:raise RuntimeError('simulated interruption')
                return [SimpleNamespace(payload=(torch.tensor([1,2]),1,{}),answer='A',reward=1.,parsed=True,length=1,truncated=False,text='A')]
        return FakeBackend,calls,initializations
    def test_resume_keeps_saved_prefix_and_replays_anchor(self):
        cls,_,_=self.backend(fail_on=2)
        with patch('transfer_alignment.baseline_endpoints.FullVisualBackend',cls):
            with self.assertRaisesRegex(RuntimeError,'interruption'):run(self.plan,'initial',self.output)
        before=(self.output/'responses/0000.json').read_bytes()
        cls,calls,inits=self.backend()
        with patch('transfer_alignment.baseline_endpoints.FullVisualBackend',cls):run(self.plan,'initial',self.output,True)
        self.assertEqual(before,(self.output/'responses/0000.json').read_bytes())
        self.assertEqual(calls[0],'image-0');self.assertEqual(calls[1],'image-1');self.assertEqual(calls[-1],'image-0')
        self.assertEqual(len(list((self.output/'responses').glob('*.json'))),288)
        self.assertIsNone(inits[0][0]);self.assertTrue(inits[0][1]['pretrained_initial'])
        summary=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(summary['correct_image']['count'],256);self.assertEqual(summary['shuffled_image']['count'],32)
    def test_changed_model_blocks_resume_before_backend_load(self):
        cls,_,_=self.backend(fail_on=2)
        with patch('transfer_alignment.baseline_endpoints.FullVisualBackend',cls):
            with self.assertRaises(RuntimeError):run(self.plan,'initial',self.output)
        (self.root/'model/weights').write_bytes(b'changed')
        cls,_,inits=self.backend()
        with patch('transfer_alignment.baseline_endpoints.FullVisualBackend',cls):
            with self.assertRaisesRegex(ValueError,'Frozen artifact changed'):run(self.plan,'initial',self.output,True)
        self.assertEqual(inits,[])
    def test_wrong_saved_item_blocks_resume(self):
        cls,_,_=self.backend(fail_on=2)
        with patch('transfer_alignment.baseline_endpoints.FullVisualBackend',cls):
            with self.assertRaises(RuntimeError):run(self.plan,'initial',self.output)
        path=self.output/'responses/0000.json';row=json.loads(path.read_text());row['item_id']='wrong';path.write_text(json.dumps(row))
        cls,_,inits=self.backend()
        with patch('transfer_alignment.baseline_endpoints.FullVisualBackend',cls):
            with self.assertRaisesRegex(ValueError,'Saved response identity'):run(self.plan,'initial',self.output,True)
        self.assertEqual(inits,[])


if __name__=='__main__':unittest.main()
