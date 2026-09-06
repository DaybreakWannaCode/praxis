import ast
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, MethodType

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transfer_alignment.praxis_bridge import PraxisStepRecorder

ROOT=Path(os.environ.get("PRAXIS_ROOT",Path(__file__).resolve().parents[3]/"Praxis-VLM-main"))
SOURCE=ROOT/"verl/workers/actor/dp_actor.py"


@unittest.skipUnless(SOURCE.exists(),"Set PRAXIS_ROOT to the original source checkout")
class PraxisBoundaryTests(unittest.TestCase):
    def actor(self):
        # Execute the exact source method, avoiding Ray/vLLM imports. This checks
        # its optimizer boundary, not its distributed forward or rollout pipeline.
        tree=ast.parse(SOURCE.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DataParallelPPOActor")
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_optimizer_step"))
        env={"torch":torch,"nn":nn,"FSDP":FSDP}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),"exec"),env)
        model=nn.Linear(3,2)
        actor=SimpleNamespace(actor_module=model,actor_optimizer=torch.optim.AdamW(model.parameters(),lr=.01),config=SimpleNamespace(max_grad_norm=.5))
        actor._optimizer_step=MethodType(env["_optimizer_step"],actor)
        return actor

    def test_original_boundary_unchanged_with_warm_adam(self):
        actor=self.actor()
        for p in actor.actor_module.parameters():p.grad=torch.ones_like(p)
        actor._optimizer_step()
        control=self.actor();control.actor_module.load_state_dict(actor.actor_module.state_dict())
        control.actor_optimizer.load_state_dict(copy.deepcopy(actor.actor_optimizer.state_dict()))
        before={k:p.detach().clone() for k,p in actor.actor_module.named_parameters()}
        for obj in [actor,control]:
            for p in obj.actor_module.parameters():p.grad=torch.full_like(p,2.)
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            with PraxisStepRecorder(actor,Path(temp)/"record"):
                actor._optimizer_step()
            control._optimizer_step()
            for a,b in zip(actor.actor_module.parameters(),control.actor_module.parameters()):self.assertTrue(torch.equal(a,b))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            delta=torch.load(Path(temp)/"record/rank-00000/delta-000000.pt",weights_only=True)
            for i,(name,p) in enumerate(actor.actor_module.named_parameters()):
                self.assertTrue(torch.equal(delta[f"group0.param{i}"],p-before[name]))
            self.assertEqual(json.loads((Path(temp)/"record/rank-00000/steps.jsonl").read_text())["status"],"applied")

    def test_nonfinite_skip_is_not_a_fake_update(self):
        actor=self.actor()
        before=[p.detach().clone() for p in actor.actor_module.parameters()]
        for p in actor.actor_module.parameters():p.grad=torch.full_like(p,float('nan'))
        with tempfile.TemporaryDirectory() as temp:
            with PraxisStepRecorder(actor,temp): actor._optimizer_step()
            record=json.loads((Path(temp)/"rank-00000/steps.jsonl").read_text())
            self.assertEqual(record["status"],"skipped")
            self.assertFalse(list(Path(temp).rglob("delta*.pt")))
        for p,b in zip(actor.actor_module.parameters(),before):self.assertTrue(torch.equal(p,b))

    def test_rank_local_rejects_global_gradient(self):
        actor=self.actor()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):PraxisStepRecorder(actor,temp,scope="rank_local",visual_gradient={})
