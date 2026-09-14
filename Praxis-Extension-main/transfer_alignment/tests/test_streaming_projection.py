import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transfer_alignment.production_displacement import save_displacement, load_tensor
from transfer_alignment.streaming_projection import project_and_apply


class StreamingProjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'delta'
        torch.manual_seed(72)
        self.parent = {'a':torch.randn(4,9), 'b':torch.randn(13)}
        self.child = {n:p+torch.randn_like(p)*.001 for n,p in self.parent.items()}
        self.gradient = {n:torch.randn(p.shape, dtype=torch.float64) for n,p in self.parent.items()}
        self.manifest = save_displacement(self.parent,self.child,self.root,allow_float64=True)
        self.params = {n:p.clone() for n,p in self.parent.items()}

    def tearDown(self):
        self.tmp.cleanup()

    def score(self):
        return project_and_apply(self.gradient,self.parent,self.params,self.root,self.manifest,chunk_elements=7)

    def test_matches_dense_projection_and_exact_child(self):
        result = self.score()
        expected = sum(float((self.gradient[r['name']]*load_tensor(self.root,r).double()).sum()) for r in self.manifest['parameters'])
        self.assertAlmostEqual(result['alignment'],expected,places=13)
        for name in self.params:
            self.assertTrue(torch.equal(self.params[name],self.child[name]))
        self.assertEqual(result['canonical_numel'],49)

    def test_checksums_fail_closed_and_restore_after_partial_apply(self):
        self.manifest['parameters'][-1]['raw_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'checksum'): self.score()
        for n in self.params:self.assertTrue(torch.equal(self.params[n],self.parent[n]))

    def test_rejects_duplicate_canonical_coordinates(self):
        self.manifest['parameters'].append(self.manifest['parameters'][0])
        with self.assertRaisesRegex(ValueError,'coverage'):self.score()

    def test_norm_mismatch_restores(self):
        self.manifest['update_norm']*=2
        with self.assertRaisesRegex(ValueError,'norm differs'):self.score()
        for n in self.params:self.assertTrue(torch.equal(self.params[n],self.parent[n]))

    def test_nonfinite_gradient_restores(self):
        self.gradient['b'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):self.score()
        for n in self.params:self.assertTrue(torch.equal(self.params[n],self.parent[n]))

    def test_float64_archive(self):
        row=self.manifest['parameters'][0]
        tensor=load_tensor(self.root,row).double()
        raw=tensor.numpy().tobytes()
        with gzip.open(self.root/row['file'],'wb') as f:f.write(raw)
        row['dtype']='float64';row['raw_sha256']=hashlib.sha256(raw).hexdigest()
        self.score()
        self.assertTrue(torch.equal(self.params['a'],self.child['a']))

    def test_trailing_bytes_fail_closed(self):
        row=self.manifest['parameters'][-1]
        with gzip.open(self.root/row['file'],'ab') as f:f.write(b'garbage')
        with self.assertRaisesRegex(ValueError,'Trailing'):self.score()
        for n in self.params:self.assertTrue(torch.equal(self.params[n],self.parent[n]))


if __name__=='__main__':unittest.main()
