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

class HorizonDispatchTests(unittest.TestCase):
    def test_inventory_ignores_order_but_preserves_text_gold_and_counts(self):
        from types import SimpleNamespace
        from transfer_alignment.production_horizon import prompt_inventory
        def score(p,g):
            return prompt_inventory(SimpleNamespace(non_tensor_batch=dict(problem=p,ground_truth=g)))
        self.assertEqual(score(['a','b'],['A','B']),score(['b','a'],['B','A']))
        self.assertNotEqual(score(['a','b'],['A','B']),score(['a','b'],['B','A']))
        self.assertNotEqual(score(['a'],['A']),score(['a','a'],['A','A']))

    def test_h4_unresolved_is_terminal_budget_result(self):
        from transfer_alignment.production_statistics import precision_decision
        scores = [dict(first=0,second=1,approximate_t_interval=[-.1,.1])]
        outcomes = [dict(first=0,second=1,pointwise=dict(fixed_panel_interval=[-.1,.1]))]
        result = precision_decision(scores,outcomes,horizon=4,minimum_pairs=1)
        self.assertEqual(result['h4_decision'], 'bounded_budget_insufficient')
        self.assertNotIn('h1_decision', result)
        self.assertFalse(result['main_sweep_authorized'])

    def test_default_explicit_four_and_invalid_dispatch(self):
        import os
        from unittest.mock import patch
        from transfer_alignment.production_parity import dispatch_parity_gate
        with patch.dict(os.environ, {}, clear=True), \
                patch('transfer_alignment.production_parity.run_fixed_rollout_gate', return_value=1) as one, \
                patch('transfer_alignment.production_horizon.run_four_step_gate', return_value=4) as four:
            self.assertEqual(dispatch_parity_gate(None, None), 1)
            one.assert_called_once()
            os.environ['PRAXIS_ALIGNMENT_HORIZON'] = '4'
            self.assertEqual(dispatch_parity_gate(None, None), 4)
            four.assert_called_once()
            os.environ['PRAXIS_ALIGNMENT_HORIZON'] = '5'
            with self.assertRaisesRegex(ValueError, 'Only H=1 and H=4'):
                dispatch_parity_gate(None, None)
            self.assertEqual(one.call_count, 1)
            self.assertEqual(four.call_count, 1)


class HorizonWorkerTests(unittest.TestCase):
    def test_four_real_adam_updates_export_total_and_reject_fifth(self):
        import copy
        import os
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from transfer_alignment.tests.test_production_gate import Worker
        from transfer_alignment.production_parity import parameters
        from transfer_alignment.production_horizon import run_four_step_gate
        from transfer_alignment.production_displacement import load_tensor
        worker = Worker()
        worker.update_actor(None)
        initial = {k:p.detach().clone() for k,p in parameters(worker).items()}
        reference = Worker()
        reference.fsdp_module.load_state_dict(copy.deepcopy(worker.fsdp_module.state_dict()))
        reference.optimizer.load_state_dict(copy.deepcopy(worker.optimizer.state_dict()))
        reference.lr_scheduler.load_state_dict(copy.deepcopy(worker.lr_scheduler.state_dict()))
        data = SimpleNamespace(batch={'x':torch.ones(1)}, meta_info={}, non_tensor_batch={'problem':['fixed'], 'ground_truth':['A']})
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                'PRAXIS_PARITY_DIR':str(Path(folder)/'gate'),
                'PRAXIS_PARENT_MODEL':'unused', 'PRAXIS_REWARD_CONTRACT':'unused'}), \
                patch('torch.cuda.synchronize'), \
                patch('torch.distributed.get_world_size', return_value=1), \
                patch('transfer_alignment.production_coordinates.manifest', return_value={}), \
                patch('transfer_alignment.production_coordinates.verify_values', return_value={'passed':True}), \
                patch('transfer_alignment.production_coordinates.canonical_views', side_effect=lambda p,m:p), \
                patch('transfer_alignment.production_parity.audit_text_batch'), \
                patch('torch.load', return_value={}):
            for i in range(4):
                run_four_step_gate(worker, data, scorer=lambda *_:None)
                reference.update_actor(data)
                for k,p in parameters(worker).items():
                    self.assertTrue(torch.equal(p, parameters(reference)[k]))
                self.assertEqual(os.environ['PRAXIS_PARENT_MODEL'], 'unused')
                if i<3:
                    self.assertFalse((Path(folder)/'gate/delta').exists())
            root = Path(folder)/'gate'
            exported = json.loads((root/'delta/manifest.json').read_text())
            for row in exported['parameters']:
                self.assertTrue(torch.equal(initial[row['name']]+load_tensor(root/'delta',row),
                                            parameters(worker)[row['name']]))
            self.assertEqual(json.loads((root/'parity.json').read_text())['completed_steps'],4)
            from transfer_alignment.production_horizon import validate_horizon_export
            report = json.loads((root/'parity.json').read_text())
            validate_horizon_export(root, report, 4)
            wrong = copy.deepcopy(report)
            wrong['parent_digests']['optimizer'] = 'wrong-initial-adam'
            with self.assertRaisesRegex(ValueError, 'initial parent'):
                validate_horizon_export(root, wrong, 4)
            with self.assertRaisesRegex(ValueError, 'Export horizon'):
                validate_horizon_export(root, report, 1)
            with self.assertRaisesRegex(RuntimeError, 'four updates'):
                run_four_step_gate(worker,data,scorer=lambda *_:None)

if __name__ == '__main__':
    unittest.main()
