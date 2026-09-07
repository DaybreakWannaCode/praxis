import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from transfer_alignment.production_parity import run_fixed_rollout_gate, snapshot
from transfer_alignment.production_rewards import group_diagnostics
from transfer_alignment.core import digest


class Worker:
    def __init__(self):
        self.fsdp_module = torch.nn.Linear(2,1)
        self.fsdp_module.register_buffer("counter", torch.zeros(1))
        self.optimizer = torch.optim.AdamW(self.fsdp_module.parameters(), lr=.01)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _:1.)
        self.tokenizer = None
        self.actor = SimpleNamespace(actor_module=self.fsdp_module, actor_optimizer=self.optimizer,
                                     _optimizer_step=self.step)

    def step(self):
        norm = torch.nn.utils.clip_grad_norm_(self.fsdp_module.parameters(),1.)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return norm

    def update_actor(self, data):
        for p in self.fsdp_module.parameters():
            p.grad = torch.ones_like(p)
        self.actor._optimizer_step()
        self.fsdp_module.counter.add_(1)
        self.lr_scheduler.step()
        return {"completed":True}


class ProductionGateTests(unittest.TestCase):
    def test_warm_fixed_rollout_parity_and_no_delta_file(self):
        worker=Worker()
        worker.update_actor(None)
        data=SimpleNamespace(batch={"x":torch.ones(1)},meta_info={},non_tensor_batch={})
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{
                "PRAXIS_PARITY_DIR":str(Path(folder)/"gate"),"PRAXIS_REWARD_CONTRACT":"unused"}), \
                patch("torch.cuda.synchronize"), patch("torch.distributed.get_world_size",return_value=1), \
                patch("transfer_alignment.production_parity.audit_text_batch"):
            result=run_fixed_rollout_gate(worker,data,scorer=lambda *_:None)
            self.assertEqual(result,{"completed":True})
            report=json.loads((Path(folder)/"gate/parity.json").read_text())
            self.assertEqual(report["status"],"passed")
            self.assertTrue(all(report["equal"].values()))
            self.assertEqual(worker.fsdp_module.counter.item(),2)
            self.assertFalse(list((Path(folder)/"gate/observer").rglob("delta*.pt")))

    def test_reward_variation_can_exist_with_constant_correctness(self):
        rows=[{"group_id":"a","components":{"accuracy":1.,"format":f,"tag_count":0.,"length":0.,"overall":1.+.8*f}} for f in (0.,1.)]
        item=group_diagnostics(rows)[0]
        self.assertTrue(item["correctness_constant"])
        self.assertFalse(item["composite_constant"])
        self.assertAlmostEqual(item["composite_population_variance"],.16)
        self.assertAlmostEqual(sum(map(sum,item["weighted_component_population_covariance"])),.16)


if __name__ == "__main__":
    unittest.main()
