"""Supplement original worker checkpoints with exposed rollout-manager state.

This is not complete vLLM-engine or trainer/dataloader replay. It captures the
specific persistent RNG tensors present in the supplied Praxis sharding manager.
"""
from pathlib import Path
import torch
from .core import cpu_copy, rng_state, restore_rng


def capture_worker_state(worker, *, rank, world_size):
    manager=getattr(worker,"rollout_sharding_manager",None)
    rollout=None
    if manager is not None:
        required=("torch_random_states","gen_random_states","freed_bytes")
        if not all(hasattr(manager,k) for k in required):
            raise ValueError("Unrecognized Praxis rollout-manager state")
        rollout={k:cpu_copy(getattr(manager,k)) for k in required}
    return {"version":1,"rank":rank,"world_size":world_size,"rng":rng_state(),
            "modes":{n:m.training for n,m in worker.fsdp_module.named_modules()},
            "rollout_manager":rollout,
            "limitation":"Does not serialize vLLM engine/request state or trainer/dataloader state"}


def restore_worker_state(worker, state, *, rank, world_size):
    if (state["version"],state["rank"],state["world_size"])!=(1,rank,world_size):
        raise ValueError("Worker checkpoint topology differs")
    modules=dict(worker.fsdp_module.named_modules())
    if set(modules)!=set(state["modes"]):raise ValueError("Worker module layout changed")
    manager=getattr(worker,"rollout_sharding_manager",None)
    if (manager is None)!=(state["rollout_manager"] is None):raise ValueError("Rollout manager mismatch")
    if manager is not None:
        for k,v in state["rollout_manager"].items():
            if not hasattr(manager,k):raise ValueError("Rollout manager schema changed")
            setattr(manager,k,cpu_copy(v))
    for n,m in modules.items():m.training=state["modes"][n]
    restore_rng(state["rng"])


def save_sidecar(worker, path):
    import torch.distributed as dist
    rank,world=dist.get_rank(),dist.get_world_size()
    destination=Path(path)/f"alignment_extra_rank_{rank}.pt"
    temporary=destination.with_suffix('.tmp')
    torch.save(capture_worker_state(worker,rank=rank,world_size=world),temporary)
    temporary.replace(destination)


def load_sidecar(worker, path):
    import torch.distributed as dist
    rank,world=dist.get_rank(),dist.get_world_size()
    # This checkpoint must be produced locally by this project, not downloaded.
    state=torch.load(Path(path)/f"alignment_extra_rank_{rank}.pt",map_location="cpu",weights_only=False)
    restore_worker_state(worker,state,rank=rank,world_size=world)
