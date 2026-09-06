import random
import unittest
from types import SimpleNamespace
import torch
from transfer_alignment.praxis_state import capture_worker_state,restore_worker_state


class PraxisStateTests(unittest.TestCase):
    def test_rollout_rng_and_mixed_modes_restore(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout())
        model.train();model[1].eval()
        manager=SimpleNamespace(torch_random_states=torch.tensor([1,2],dtype=torch.uint8),gen_random_states=torch.tensor([3,4],dtype=torch.uint8),freed_bytes=17)
        worker=SimpleNamespace(fsdp_module=model,rollout_sharding_manager=manager)
        state=capture_worker_state(worker,rank=0,world_size=1)
        expected=(random.random(),torch.rand(2))
        model.eval();manager.gen_random_states.zero_();manager.freed_bytes=0
        restore_worker_state(worker,state,rank=0,world_size=1)
        self.assertTrue(model.training);self.assertFalse(model[1].training)
        self.assertTrue(torch.equal(manager.gen_random_states,torch.tensor([3,4],dtype=torch.uint8)))
        self.assertEqual(manager.freed_bytes,17)
        self.assertEqual(random.random(),expected[0]);self.assertTrue(torch.equal(torch.rand(2),expected[1]))
        with self.assertRaises(ValueError):restore_worker_state(worker,state,rank=1,world_size=2)
