import tempfile
import unittest
from pathlib import Path
import torch
from transfer_alignment.backends import TinyBackend
from transfer_alignment.core import digest,weights
from transfer_alignment.data import synthetic_items
from transfer_alignment.experiment import visual_gradient
from transfer_alignment.precision import probe_summary,paired_outcomes,run

class PrecisionTests(unittest.TestCase):
    def test_item_decomposition_preserves_original_estimate(self):
        b=TinyBackend();items=[i for i in synthetic_items() if i.split=='score'];parts=[]
        g,_,_=visual_gradient(b,items,8,19)
        g2,_,_=visual_gradient(b,items,8,19,on_item=lambda i,v:parts.append(v))
        self.assertEqual(digest(g),digest(g2))
        for k in g:
            torch.testing.assert_close(g[k],torch.stack([v[k] for v in parts]).mean(0))

    def test_unresolved_ranking_and_fixed_repeat_count(self):
        records=[{'a':[v,0.],'b':[v,0.]} for v in [1.,-1.,1.,-1.]]
        self.assertIsNone(probe_summary(records)['conditional_mc_ranking'])
        records=[{'a':[v,0.],'b':[v,0.]} for v in [1.,1.1,1.,1.1]]
        self.assertEqual(probe_summary(records)['conditional_mc_ranking'],[0,1])
        self.assertFalse(probe_summary(records)['selection_authorized'])
        with self.assertRaises(ValueError):probe_summary(records[:2])

    def test_paired_actual_change_null_replay_and_restore(self):
        b=TinyBackend();before=weights(b.model);child={k:v.clone() for k,v in before.items()}
        child['weight'][0]+=10;child['weight'][1]-=10
        dev=[i for i in synthetic_items() if i.split=='dev']
        with tempfile.TemporaryDirectory() as tmp:
            r=paired_outcomes(b,dev,child,Path(tmp)/'out',count=8,seed=8)
            self.assertTrue(r['null_exact_replay'])
            self.assertGreater(r['paired_response_change_fraction'],0)
            self.assertEqual(digest(before),digest(weights(b.model)))

    def test_bounded_end_to_end_completion(self):
        import json
        b=TinyBackend();parent=weights(b.model)
        deltas=[{k:torch.tensor([[.02,0.],[0.,-.02]]) for k in parent},
                {k:torch.tensor([[-.02,0.],[0.,.02]]) for k in parent}]
        child={k:v+deltas[0][k] for k,v in parent.items()}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            run(b,synthetic_items(),deltas,child,out)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'complete')
            self.assertTrue((out/'completed.json').exists())
            self.assertEqual(digest(weights(b.model)),digest(parent))
