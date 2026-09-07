"""Regression checks for exact endpoint recovery across cancellation."""
import tempfile
import unittest
from pathlib import Path

import torch

from transfer_alignment.production_displacement import load_tensor, save_displacement


class DisplacementPrecisionTests(unittest.TestCase):
    def test_selective_promotion_and_legacy_loading(self):
        before = {'crossing': torch.tensor([1., -1., 0.]), 'ordinary': torch.ones(4)}
        after = {'crossing': torch.tensor([1e-8, -1e-8, 1e-8]), 'ordinary': torch.ones(4)+1e-5}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Child reconstruction failed'):
                save_displacement(before, after, Path(folder)/'legacy-failure')
            root = Path(folder)/'mixed'
            report = save_displacement(before, after, root, allow_float64=True)
            self.assertEqual(report['version'], 2)
            self.assertEqual([r['dtype'] for r in report['parameters']], ['float64','float32'])
            for row in report['parameters']:
                delta = load_tensor(root, row)
                child = (before[row['name']] + delta).float()
                self.assertTrue(torch.equal(child, after[row['name']]))
            old = Path(folder)/'legacy'
            legacy = save_displacement({'p': before['ordinary']}, {'p': after['ordinary']}, old)
            self.assertNotIn('dtype', legacy['parameters'][0])
            self.assertEqual(load_tensor(old, legacy['parameters'][0]).dtype, torch.float32)

    def test_nonfinite_and_unrepresentable_remain_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            for i, value in enumerate([float('nan'), float('inf')]):
                with self.assertRaisesRegex(ValueError, 'Nonfinite endpoint'):
                    save_displacement({'p': torch.ones(1)}, {'p': torch.tensor([value])},
                                      Path(folder)/str(i), allow_float64=True)
            # Even FP64 is insufficient for arbitrary FP32 exponent separation.
            # The exporter must fail explicitly, never pretend recovery is exact.
            with self.assertRaisesRegex(ValueError, 'Child reconstruction failed'):
                save_displacement({'p': torch.ones(1)}, {'p': torch.tensor([1e-30])},
                                  Path(folder)/'wide', allow_float64=True)

    def test_promoted_bytes_are_verified_on_replay_and_load(self):
        before, after = {'p':torch.ones(1)}, {'p':torch.tensor([1e-8])}
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder)/'old', Path(folder)/'new'
            report = save_displacement(before, after, old, allow_float64=True)
            replay = save_displacement(before, after, new, resume_from=old, allow_float64=True)
            self.assertTrue((new/replay['parameters'][0]['file']).is_symlink())
            bad = dict(report['parameters'][0], raw_sha256='bad')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_tensor(old, bad)


if __name__ == '__main__':
    unittest.main()
