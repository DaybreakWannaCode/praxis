import json
from pathlib import Path
import tempfile
import unittest

from transfer_alignment.production_horizon import validate_four_steps


class HorizonTests(unittest.TestCase):
    def make_chain(self, folder):
        roots = []
        fields = ('parameters', 'buffers', 'optimizer', 'scheduler', 'worker')
        for i in range(4):
            root = Path(folder) / str(i)
            observer = root / 'observer/rank-00000'
            observer.mkdir(parents=True)
            parent = {k: f'{k}-{i}' for k in fields}
            child = {k: f'{k}-{i+1}' for k in fields}
            report = dict(status='passed', equal={k: True for k in fields},
                          parent_digests=parent, control_digests=child,
                          observed_digests=child)
            (root/'parity.json').write_text(json.dumps(report))
            (observer/'steps.jsonl').write_text(json.dumps(dict(status='applied', learning_rates_at_step=[1e-6]))+'\n')
            (observer/'boundaries.jsonl').write_text(json.dumps(dict(optimizer_calls=1))+'\n')
            roots.append(root)
        return roots

    def test_chain_and_incomplete_horizon(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = self.make_chain(folder)
            self.assertEqual(len(validate_four_steps(roots)), 4)
            for bad in (roots[:3], roots+roots[:1]):
                with self.assertRaisesRegex(ValueError, 'exactly four'):
                    validate_four_steps(bad)

    def test_reject_reset_adam_even_when_each_step_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = self.make_chain(folder)
            p = roots[2]/'parity.json'
            r = json.loads(p.read_text())
            r['parent_digests']['optimizer'] = 'reset-adam'
            p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, 'Discontinuous.*optimizer'):
                validate_four_steps(roots)

    def test_reject_multiple_optimizer_steps_and_zero_lr(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = self.make_chain(folder)
            p = roots[0]/'observer/rank-00000/steps.jsonl'
            original = p.read_text()
            p.write_text(original*2)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                validate_four_steps(roots)
            p.write_text(json.dumps(dict(status='applied', learning_rates_at_step=[0]))+'\n')
            with self.assertRaisesRegex(ValueError, 'positive learning'):
                validate_four_steps(roots)

if __name__ == '__main__':
    unittest.main()
