import tempfile
from pathlib import Path
import unittest
import torch
from transfer_alignment.production_displacement import save_displacement, load_tensor, ExportBudgetExceeded

class BudgetTests(unittest.TestCase):
    def test_bounded_export_is_lossless(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'delta';before={'p':torch.randn(100)};after={'p':before['p']+.001}
            m=save_displacement(before,after,root,allow_float64=True,max_output_bytes=10000)
            self.assertTrue(torch.equal((before['p'].double()+load_tensor(root,m['parameters'][0]).double()).float(),after['p']))
    def test_cumulative_limit_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'delta';before={str(i):torch.zeros(1000) for i in range(3)}
            after={k:torch.randn_like(v) for k,v in before.items()}
            with self.assertRaises(ExportBudgetExceeded):
                save_displacement(before,after,root,allow_float64=True,max_output_bytes=5000)
            self.assertLessEqual(sum(f.stat().st_size for f in root.iterdir()),5000)
            self.assertFalse((root/'manifest.json').exists())
    def test_limit_includes_header_and_footer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'delta'
            with self.assertRaises(ExportBudgetExceeded):
                save_displacement({'p':torch.zeros(2)},{'p':torch.zeros(2)},root,max_output_bytes=1)
            self.assertLessEqual(sum(f.stat().st_size for f in root.iterdir()),1)
    def test_reuse_cannot_bypass_budget(self):
        with self.assertRaises(ValueError):save_displacement({}, {}, '/unused',max_output_bytes=1,resume_from='/unused')

if __name__=='__main__':unittest.main()
