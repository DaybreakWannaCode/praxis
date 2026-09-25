import copy
import unittest
import numpy as np
import torch
from transfer_alignment.loaded_checkpoint_audit import compare_state


class LoadedStateTests(unittest.TestCase):
    def test_nested_optimizer_rng_state_and_no_mutation(self):
        state = {'state': {0: {'step': torch.tensor(16.), 'exp_avg': torch.arange(12.)}, 1: {}},
                 'param_groups': [{'params': [0,1], 'lr': 1e-6}], 'numpy': ('MT',np.arange(10,dtype=np.uint32))}
        actual = copy.deepcopy(state)
        self.assertEqual(compare_state(actual,state),13)
        self.assertEqual(compare_state(actual,state),13)

    def test_late_chunk_difference_detected(self):
        expected = torch.zeros(1024*1024+7)
        actual = expected.clone(); actual[-1] = 1
        with self.assertRaisesRegex(ValueError,'chunk 1048576'):
            compare_state(actual,expected)

    def test_mapping_dtype_and_scheduler_counter_mismatches(self):
        cases = [(torch.ones(2,dtype=torch.float64),torch.ones(2)),
                 ({'last_epoch':17},{'last_epoch':16}),
                 ({'params':[1,0]},{'params':[0,1]}),
                 ({0:{}},{0:{},1:{}}), ([1],(1,))]
        for actual,expected in cases:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                compare_state(actual,expected)


if __name__ == '__main__': unittest.main()
