import os
from pathlib import Path
import tempfile
import unittest
from transfer_alignment.checkpoint_quota import CheckpointQuota, accounted_bytes


class QuotaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = self.root / 'checkpoints'; self.store.mkdir()

    def tearDown(self): self.temp.cleanup()

    def test_sparse_file_counts_logical_size_and_blocks_new_checkpoint(self):
        with (self.root / 'sparse').open('wb') as f: f.truncate(10_000_000)
        self.assertGreaterEqual(accounted_bytes(self.root), 10_000_000)
        guard = CheckpointQuota(self.root, quota_bytes=12_000_000,
                                replacement_bytes=3_000_000, reserve_bytes=1_000_000)
        with self.assertRaises(OSError): guard(self.store, 2)

    def test_hardlinks_not_counted_as_second_full_copy(self):
        p = self.root / 'model'; p.write_bytes(b'x' * 100_000)
        os.link(p, self.store / 'model')
        self.assertLess(accounted_bytes(self.root), 200_000)

    def test_replacement_and_reserve_both_required(self):
        used = accounted_bytes(self.root)
        fits = CheckpointQuota(self.root, quota_bytes=used+30_000,
                              replacement_bytes=20_000, reserve_bytes=10_000)
        self.assertEqual(fits(self.store, 2)['accounted_bytes'], used)
        fails = CheckpointQuota(self.root, quota_bytes=used+29_999,
                               replacement_bytes=20_000, reserve_bytes=10_000)
        with self.assertRaises(OSError): fails(self.store, 2)

    def test_cannot_guard_store_on_different_volume(self):
        with tempfile.TemporaryDirectory() as outside:
            guard = CheckpointQuota(self.root, quota_bytes=10**9, replacement_bytes=10**6, reserve_bytes=10**6)
            with self.assertRaises(ValueError): guard(outside, 2)


if __name__ == '__main__': unittest.main()
