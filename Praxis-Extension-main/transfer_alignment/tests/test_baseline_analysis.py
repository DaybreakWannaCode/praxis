import json
from pathlib import Path
import tempfile
import unittest
from transfer_alignment.baseline_analysis import analyze,paired,sha


class BaselineAnalysisTests(unittest.TestCase):
    def test_paired_transitions_and_zero_interval(self):
        r=paired([0,1,0,1],[1,0,0,1],draws=100)
        self.assertEqual(r['transitions'],{'0_to_0':1,'0_to_1':1,'1_to_0':1,'1_to_1':1})
        self.assertEqual(r['accuracy_change'],0)
        self.assertEqual(paired([1,0],[1,0],draws=100)['paired_image_bootstrap_95'],[0,0])

    def test_complete_audit_rejects_changed_label_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
            items=[dict(id=str(i),group_id=str(i),answer='A') for i in range(256)]
            controls=[dict(items[i],id=str(i)+'-shuffled') for i in range(32)]
            panel=root/'panel.json';write(panel,dict(items=items,shuffled_controls=controls,shuffle_mapping={str(i):str((i+1)%32) for i in range(32)}))
            plan=root/'plan.json';write(plan,dict(panel='/frozen/panel',artifact_sha256={'/frozen/panel':sha(panel)},scope='test'))
            identity=sha(plan)
            for name in ['initial','final']:
                write(root/name/'manifest.json',dict(status='complete',endpoint=name,plan_sha256=identity))
                (root/(name+'.exit')).write_text('0\n')
                summary=dict(status='complete',endpoint=name,plan_sha256=identity,greedy_replay_exact=True,count=288)
                for i,item in enumerate(items+controls):
                    correct=name=='final' and i<256
                    kind='correct_image' if i<256 else 'shuffled_image'
                    write(root/name/'responses'/f'{i:04d}.json',dict(item_id=item['id'],group_id=item['group_id'],split='dev',kind=kind,plan_sha256=identity,answer='A' if correct else 'B',correct=float(correct),parsed=True,truncated=False,sequence_token_ids=[1,2]))
                for kind,n in [('correct_image',256),('shuffled_image',32)]:
                    summary[kind]=dict(count=n,correct=n if name=='final' and kind=='correct_image' else 0,parsed=n,truncated=0)
                write(root/name/'summary.json',summary)
            r=analyze(root,panel)
            self.assertEqual(r['primary']['accuracy_change'],1)
            self.assertEqual(r['image_controls']['image_gap_change'],1)
            self.assertEqual(len(r['response_sha256']),576)
            p=root/'final/responses/0000.json';v=json.loads(p.read_text());v['answer']='B';write(p,v)
            with self.assertRaisesRegex(ValueError,'correctness'):analyze(root,panel)

if __name__=='__main__':unittest.main()
