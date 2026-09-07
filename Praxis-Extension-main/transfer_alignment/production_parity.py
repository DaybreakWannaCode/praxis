"""One-rank fixed-rollout parity gate in the actual original FSDP worker.

This is not complete rollout-engine replay, a visual probe, or global FSDP
alignment. It restores the mutable state used by a fixed optimizer input and
requires exact control/observed post-state agreement before returning to training.
"""
import copy
import gc
import os
from pathlib import Path
import time

import torch

from .core import cpu_copy, digest
from .experiment import write_json
from .praxis_bridge import PraxisStepRecorder
from .praxis_state import capture_worker_state, restore_worker_state
from .production_rewards import audit_text_batch


def parameters(worker):
    return {f"group{gi}.param{pi}": p
            for gi, group in enumerate(worker.optimizer.param_groups)
            for pi, p in enumerate(group["params"])}


def snapshot(worker):
    torch.cuda.synchronize()
    return {"parameters": {k: p.detach().cpu().clone() for k,p in parameters(worker).items()},
            "buffers": {k: b.detach().cpu().clone() for k,b in worker.fsdp_module.named_buffers()},
            "optimizer": cpu_copy(worker.optimizer.state_dict()),
            "scheduler": cpu_copy(worker.lr_scheduler.state_dict()),
            "worker": capture_worker_state(worker, rank=0, world_size=1)}


def restore(worker, state):
    current = parameters(worker)
    if current.keys() != state["parameters"].keys():
        raise ValueError("Optimizer coordinates changed")
    with torch.no_grad():
        for k,p in current.items():
            if p.shape != state["parameters"][k].shape or p.dtype != state["parameters"][k].dtype:
                raise ValueError("Parameter shape or dtype changed")
            p.copy_(state["parameters"][k].to(p.device))
        buffers = dict(worker.fsdp_module.named_buffers())
        if buffers.keys() != state["buffers"].keys():
            raise ValueError("Model buffer layout changed")
        for k,b in buffers.items():
            b.copy_(state["buffers"][k].to(b.device))
    worker.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    worker.lr_scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    worker.optimizer.zero_grad(set_to_none=True)
    restore_worker_state(worker, state["worker"], rank=0, world_size=1)


def run_fixed_rollout_gate(worker, data, *, scorer=None):
    if scorer is None:
        from verl.utils.reward_score.mcq import mcq_compute_score
        scorer = mcq_compute_score
    if torch.distributed.get_world_size() != 1:
        raise ValueError("This gate is validated for one rank only")
    if getattr(worker, "_parity_gate_done", False):
        raise RuntimeError("Bounded parity gate permits one real input batch")
    root = Path(os.environ["PRAXIS_PARITY_DIR"])
    root.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "scope": "one-rank fixed-rollout worker update",
              "not_validated": ["fresh rollout replay", "driver replay", "visual alignment"],
              "deterministic_flash_attention": os.environ.get("FLASH_ATTENTION_DETERMINISTIC")}
    start = time.monotonic()
    worker._parity_gate_active = True
    parent = None
    try:
        if any(float(g["lr"]) <= 0 for g in worker.optimizer.param_groups):
            raise ValueError("Parity requires a warm parent with nonzero next LR")
        audit_text_batch(data, worker.tokenizer, scorer, root,
                         os.environ["PRAXIS_REWARD_CONTRACT"])
        # Small compared with model state; preserve the exact original optimizer input.
        torch.save(data, root / "fixed-update-input.pt")
        report["input_digest"] = digest({"tensors": dict(data.batch.items()),
                                         "metadata": data.meta_info,
                                         "non_tensor": {k: v.tolist() if hasattr(v,"tolist") else v
                                                        for k,v in data.non_tensor_batch.items()}})
        parent = snapshot(worker)
        report["parent_digests"] = {k: digest(v) for k,v in parent.items()}
        control_result = worker.update_actor(copy.deepcopy(data))
        control = snapshot(worker)
        control_hashes = {k: digest(v) for k,v in control.items()}
        del control
        gc.collect()
        restore(worker, parent)
        restored = snapshot(worker)
        restored_hashes = {k: digest(v) for k,v in restored.items()}
        del restored
        if restored_hashes != report["parent_digests"]:
            raise AssertionError("Warm parent restoration differs before replay")
        with PraxisStepRecorder(worker.actor, root / "observer", scope="rank_local", save_delta=False):
            worker.update_actor(copy.deepcopy(data))
        observed = snapshot(worker)
        observed_hashes = {k: digest(v) for k,v in observed.items()}
        del observed
        report["control_digests"] = control_hashes
        report["observed_digests"] = observed_hashes
        report["equal"] = {k: control_hashes[k] == observed_hashes[k] for k in control_hashes}
        if not all(report["equal"].values()):
            raise AssertionError("Observer/control post-state parity failed")
        report["status"] = "passed"
        worker._parity_gate_done = True
        return control_result
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        if parent is not None:
            restore(worker, parent)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-start
        write_json(root / "parity.json", report)
        worker._parity_gate_active = False
