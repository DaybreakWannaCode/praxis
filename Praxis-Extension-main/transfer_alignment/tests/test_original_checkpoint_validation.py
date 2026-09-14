import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from transfer_alignment.original_checkpoint_validation import capture_schema, validate_checkpoint


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'actor').mkdir()
        self.model = {'weight': torch.ones(3), 'alias': torch.ones(3)}
        self.optimizer = {'state': {0: {'step': torch.tensor(2.), 'exp_avg': torch.ones(3),
            'exp_avg_sq': torch.ones(3)}, 1: {}}, 'param_groups': [{'params': [0, 1],
            'lr': 1e-6, 'initial_lr': 1e-6, 'betas': (.9, .999)}]}
        self.extra = {'lr_scheduler': {'last_epoch': 2, '_step_count': 3, '_last_lr': [1e-6]},
            'rng': {'cpu': torch.get_rng_state(), 'cuda': torch.zeros(16, dtype=torch.uint8),
            'numpy': np.random.get_state(), 'random': random.getstate()}}
        self.loader = {'_snapshot': dict.fromkeys(('_snapshot_step', '_last_yielded_worker_id',
            '_main_snapshot', '_worker_snapshots')), '_steps_since_snapshot': 0, '_iterator_finished': False}
        self.save()
        self.schema = json.loads(json.dumps(capture_schema(self.root)))

    def tearDown(self): self.temp.cleanup()

    def save(self):
        for kind, obj in [('model', self.model), ('optim', self.optimizer), ('extra_state', self.extra)]:
            torch.save(obj, self.root / 'actor' / f'{kind}_world_size_1_rank_0.pt')
        torch.save(self.loader, self.root / 'dataloader.pt')

    def check(self):
        return validate_checkpoint(self.root, self.schema, optimizer_step=2,
            scheduler_epoch=2, expected_lr=1e-6, tied_aliases=[('alias', 'weight')])

    def test_valid_layout_empty_states_and_rng_unchanged(self):
        before = torch.get_rng_state().clone()
        py = random.getstate(); np_before = np.random.get_state()
        self.assertEqual(self.check()['populated_optimizer_states'], 1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(py, random.getstate())
        after = np.random.get_state()
        self.assertTrue(np.array_equal(np_before[1], after[1]))
        self.assertEqual(np_before[2:], after[2:])

    def test_corruptions_rejected(self):
        mutations = [
            lambda: self.model['weight'].fill_(float('nan')),
            lambda: self.model['alias'].fill_(2),
            lambda: self.optimizer['state'][0]['step'].fill_(2.5),
            lambda: self.optimizer['state'][0]['exp_avg_sq'].fill_(-1),
            lambda: self.optimizer['state'][0]['exp_avg'].fill_(float('inf')),
            lambda: self.optimizer['param_groups'][0].update(params=[1,0]),
            lambda: self.extra['lr_scheduler'].update(last_epoch=3),
            lambda: self.extra['rng'].pop('cuda'),
            lambda: self.loader['_snapshot'].pop('_worker_snapshots'),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                # Restore independent fixture objects between corruption cases.
                original = [self.model, self.optimizer, self.extra, self.loader]
                import copy
                self.model, self.optimizer, self.extra, self.loader = copy.deepcopy(original)
                mutate(); self.save()
                with self.assertRaises((ValueError, KeyError)): self.check()
                self.model, self.optimizer, self.extra, self.loader = original
        self.save()
        self.check()

    def test_extra_rank_rejected(self):
        torch.save({}, self.root / 'actor/model_world_size_2_rank_1.pt')
        with self.assertRaises(ValueError): self.check()


if __name__ == '__main__': unittest.main()
