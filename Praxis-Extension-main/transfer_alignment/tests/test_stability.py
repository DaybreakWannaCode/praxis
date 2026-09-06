import tempfile
import unittest
from pathlib import Path
import torch
from transfer_alignment.backends import TinyBackend
from transfer_alignment.data import synthetic_items
from transfer_alignment.core import weights
from transfer_alignment.stability import null_comparison,run


class StabilityTests(unittest.TestCase):
    def test_no_update_comparison_requires_matched_images(self):
        r=null_comparison({'a':.5,'b':1.},{'a':.75,'b':.75})
        self.assertEqual(r['no_update_mean_change'],0)
        self.assertEqual(r['descriptive_paired_image_se'],.25)
        with self.assertRaises(ValueError):null_comparison({'a':1},{'b':1})

    def test_two_probes_without_training(self):
        backend=TinyBackend();items=synthetic_items()
        deltas=[{k:torch.ones_like(v)*sign*.01 for k,v in weights(backend.model).items()} for sign in [-1,1]]
        with tempfile.TemporaryDirectory() as temp:
            result=run(backend,items,deltas,Path(temp)/'run',group_size=4)
            self.assertEqual(len(result['replicates']),2)
            self.assertTrue(result['trainable_parameters_unchanged'])
            self.assertIsNone(result['same_ranking'])  # Ties are not evidence of stable ranking.
