import json
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from transfer_alignment.production_parity import run_fixed_rollout_gate, snapshot, state_digests
from transfer_alignment.production_rewards import group_diagnostics
from transfer_alignment.production_rewards import audit_text_batch
from transfer_alignment.core import digest
from transfer_alignment.praxis_bridge import PraxisStepRecorder
from transfer_alignment.production_coordinates import flat_segments, canonical_views, verify_values
from transfer_alignment.production_displacement import save_displacement, load_tensor
from transfer_alignment.production_visual import FullVisualBackend


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
    def test_visual_child_load_always_starts_from_parent(self):
        backend=FullVisualBackend.__new__(FullVisualBackend)
        backend.model=torch.nn.Linear(2,1)
        backend.parent_state=copy.deepcopy(backend.model.state_dict())
        backend.buffers={}
        child={n:v+1e-5 for n,v in backend.parent_state.items()}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"delta"
            manifest=save_displacement(backend.parent_state,child,root)
            for _ in range(2):
                backend.apply_displacement(root,manifest)
                for n,p in backend.model.named_parameters():self.assertTrue(torch.equal(p,child[n]))
            backend.restore_parent()
            for n,p in backend.model.named_parameters():self.assertTrue(torch.equal(p,backend.parent_state[n]))

    def test_lossless_displacement_roundtrip_and_corruption(self):
        before={"a":torch.tensor([1.,-2.,0.]),"b":torch.zeros(2,3)}
        after={"a":before["a"]+torch.tensor([1e-6,-1e-6,1e-8]),"b":before["b"].clone()}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"delta"
            report=save_displacement(before,after,root)
            for row in report["parameters"]:
                self.assertTrue(torch.equal(before[row["name"]]+load_tensor(root,row),after[row["name"]]))
            row=dict(report["parameters"][0],raw_sha256="bad")
            with self.assertRaisesRegex(ValueError,"checksum"):
                load_tensor(root,row)

    def test_streamed_observer_matches_full_delta_record(self):
        worker=Worker()
        worker.update_actor(None)
        state=copy.deepcopy(worker.fsdp_module.state_dict())
        opt=copy.deepcopy(worker.optimizer.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            records=[]
            for save in (True,False):
                worker.fsdp_module.load_state_dict(state)
                worker.optimizer.load_state_dict(copy.deepcopy(opt))
                path=Path(folder)/str(save)
                with PraxisStepRecorder(worker.actor,path,save_delta=save):
                    worker.update_actor(None)
                records.append(json.loads((path/"rank-00000/steps.jsonl").read_text()))
            self.assertEqual(records[0],records[1])

    def test_canonical_padding_and_tied_alias_validation(self):
        flat=torch.tensor([1.,2.,999.,3.,4.])
        flat._fqns=("embed.weight","layer.weight")
        flat._shapes=((2,),(1,2))
        flat._numels=(2,2)
        flat._numels_with_padding=(2,1,2)
        flat._is_padding_mask=(False,True,False)
        segments=flat_segments(flat,"_fsdp_wrapped_module","p")
        mapping={"segments":segments,"aliases":{"head.weight":"embed.weight"}}
        canonical=canonical_views({"p":flat},mapping)
        self.assertEqual(sum(v.numel() for v in canonical.values()),4)
        full={k:v.clone() for k,v in canonical.items()}
        full["head.weight"]=full["embed.weight"].clone()
        self.assertEqual(verify_values({"p":flat},mapping,full)["canonical_numel"],4)
        full["head.weight"][0]=0
        with self.assertRaisesRegex(ValueError,"Tied checkpoint"):
            verify_values({"p":flat},mapping,full)
        flat._numels_with_padding=(2,1,1)
        with self.assertRaisesRegex(ValueError,"segment size"):
            flat_segments(flat,"","p")

    def test_live_hashes_match_immutable_snapshot(self):
        worker=Worker()
        worker.update_actor(None)
        with patch("torch.cuda.synchronize"):
            expected={k:digest(v) for k,v in snapshot(worker).items()}
            self.assertEqual(state_digests(worker),expected)
            worker.update_actor(None)
            self.assertNotEqual(state_digests(worker)["optimizer"],expected["optimizer"])

    def test_parity_failure_restores_parent(self):
        worker=Worker()
        worker.update_actor(None)
        data=SimpleNamespace(batch={"x":torch.ones(1)},meta_info={},non_tensor_batch={})
        original=PraxisStepRecorder._after_step
        def corrupt(recorder,*args):
            original(recorder,*args)
            with torch.no_grad():
                next(iter(recorder.params.values())).add_(1.)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{
                "PRAXIS_PARITY_DIR":str(Path(folder)/"gate"),"PRAXIS_REWARD_CONTRACT":"unused"}), \
                patch("torch.cuda.synchronize"), patch("torch.distributed.get_world_size",return_value=1), \
                patch("transfer_alignment.production_parity.audit_text_batch"):
            before=digest(snapshot(worker))
            with patch.object(PraxisStepRecorder,"_after_step",corrupt):
                with self.assertRaisesRegex(AssertionError,"post-state parity"):
                    run_fixed_rollout_gate(worker,data,scorer=lambda *_:None)
            self.assertEqual(before,digest(snapshot(worker)))
            self.assertEqual(json.loads((Path(folder)/"gate/parity.json").read_text())["status"],"failed")

    def test_audit_checks_actual_terminal_rewards(self):
        scores={"overall":1.8,"accuracy":1.,"format":1.,"tag_count":0.,"length":0.}
        row=SimpleNamespace(batch={"responses":torch.tensor([7,0]),
            "response_mask":torch.tensor([1,0]),"token_level_scores":torch.tensor([1.8,0.])},
            non_tensor_batch={"ground_truth":"A","uid":"fixed-prompt"})
        class Batch:
            def __len__(self):return 1
            def __getitem__(self,i):return row
        tokenizer=SimpleNamespace(decode=lambda *a,**k:"<answer>A</answer>")
        with tempfile.TemporaryDirectory() as folder, patch("transfer_alignment.production_rewards.verify_contract",return_value={}):
            audit_text_batch(Batch(),tokenizer,lambda *_:scores,folder,"unused")
            saved=json.loads((Path(folder)/"text-rewards.jsonl").read_text())
            self.assertEqual(saved["components"],scores)
            row.batch["token_level_scores"][0]=0.
            with self.assertRaisesRegex(ValueError,"reward tensor"):
                audit_text_batch(Batch(),tokenizer,lambda *_:scores,folder,"unused")

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
