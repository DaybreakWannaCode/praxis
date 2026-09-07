"""Observe actual Praxis optimizer calls without replacing its update algorithm.

Single-process coordinates support alignment. Distributed/FSDP capture is explicitly
rank-local only: padding, replication and global gradient mapping need separate
validation before these records can be used as a global transfer score.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import torch

from .core import alignment, digest, dot
from .experiment import append_json, write_json


class PraxisStepRecorder:
    def __init__(self, actor, output, *, scope="single_process", visual_gradient=None, save_delta=True):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if getattr(actor,"_transfer_alignment_recorder",None) is not None:
            raise ValueError("Actor already has an optimizer recorder")
        self.actor=actor
        self.optimizer=actor.actor_optimizer
        if self.optimizer is None or not hasattr(actor,"_optimizer_step"):
            raise ValueError("An initialized trainable Praxis actor is required")
        dist=torch.distributed
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.world_size=dist.get_world_size() if dist.is_initialized() else 1
        sharded=any(isinstance(m,FSDP) for m in actor.actor_module.modules())
        if scope not in {"single_process","rank_local"}:
            raise ValueError("Unknown coordinate scope")
        if (sharded or self.world_size>1) and scope!="rank_local":
            raise ValueError("Distributed/FSDP capture requires explicit rank_local scope")
        if scope=="rank_local" and visual_gradient is not None:
            raise ValueError("Global visual-gradient alignment is not validated for local shards")
        self.scope=scope
        self.save_delta=save_delta
        self.output=Path(output)/f"rank-{self.rank:05d}"
        self.output.mkdir(parents=True,exist_ok=False)
        self.params={}
        seen=set()
        for gi,group in enumerate(self.optimizer.param_groups):
            for pi,param in enumerate(group["params"]):
                if not param.requires_grad:
                    continue
                if id(param) in seen:
                    raise ValueError("Duplicate optimizer parameter")
                if param.dtype not in {torch.float32,torch.float16,torch.bfloat16}:
                    raise ValueError("Capture supports FP32/FP16/BF16 optimizer coordinates")
                seen.add(id(param));self.params[f"group{gi}.param{pi}"]=param
        if not self.params:
            raise ValueError("Optimizer has no trainable coordinates")
        self.identities={k:id(v) for k,v in self.params.items()}
        self.gradient=None
        if visual_gradient is not None:
            if set(visual_gradient)!=set(self.params):
                raise ValueError("Visual gradient must use optimizer-coordinate keys")
            self.gradient={k:v.detach().float().cpu().clone() for k,v in visual_gradient.items()}
            for k,p in self.params.items():
                if self.gradient[k].shape!=p.shape or not torch.isfinite(self.gradient[k]).all():
                    raise ValueError("Visual gradient shape or values invalid")
        self.original=actor._optimizer_step
        try:
            source=inspect.getsource(self.original).encode()
        except (OSError,TypeError):
            source=None
        names={id(p):n for n,p in actor.actor_module.named_parameters()}
        write_json(self.output/"manifest.json",{
            "scope":scope,"rank":self.rank,"world_size":self.world_size,"fsdp":sharded,
            "source_sha256":hashlib.sha256(source).hexdigest() if source else None,
            "optimizer":type(self.optimizer).__qualname__,"torch":torch.__version__,
            "save_delta":save_delta,
            "visual_gradient_digest":digest(self.gradient) if self.gradient is not None else None,
            "visual_gradient_reference":"fixed_at_attachment; stale after updates" if self.gradient is not None else None,
            "coordinates":[{"key":k,"name":names.get(id(p)),"shape":list(p.shape),"dtype":str(p.dtype)} for k,p in self.params.items()],
            "limitation":"rank_local records are not global alignment; this observer does not implement full branch restore",
        })
        self.attempt=0;self.active=False;self.calls=0;self.before=None
        self.pre=self.optimizer.register_step_pre_hook(self._before_step)
        self.post=self.optimizer.register_step_post_hook(self._after_step)
        actor._optimizer_step=self._boundary
        actor._transfer_alignment_recorder=self

    def _snapshot(self):
        current={f"group{gi}.param{pi}":p for gi,g in enumerate(self.optimizer.param_groups)
                 for pi,p in enumerate(g["params"]) if p.requires_grad}
        if {k:id(v) for k,v in current.items()}!=self.identities:
            raise ValueError("Optimizer coordinate identities changed")
        return {k:p.detach().float().cpu().clone() for k,p in self.params.items()}

    def _before_step(self, optimizer, args, kwargs):
        if not self.active:
            raise RuntimeError("Optimizer invoked outside the instrumented Praxis boundary")
        if self.calls:
            raise RuntimeError("Expected one optimizer call per Praxis boundary")
        self.calls+=1
        self.learning_rates_at_step=[float(group["lr"]) for group in optimizer.param_groups]
        self.before=self._snapshot()

    def _after_step(self, optimizer, args, kwargs):
        if not self.save_delta and self.gradient is None:
            # Parity-only runs need an exact norm/digest, not three simultaneous
            # full-model copies (before, after, delta) on the host.
            squared=0.0
            for k in sorted(self.params):
                after=self.params[k].detach().float().cpu().reshape(-1)
                before=self.before[k].reshape(-1)
                for start in range(0,before.numel(),262144):
                    d=after[start:start+262144]-before[start:start+262144]
                    if not torch.isfinite(d).all():
                        raise FloatingPointError("Nonfinite realized Praxis displacement")
                    squared+=(d.double()*d.double()).sum().item()
            if not math.isfinite(squared):
                raise FloatingPointError("Nonfinite realized Praxis displacement norm")
            record={"attempt":self.attempt,"status":"applied","scope":self.scope,
                    "before_digest":digest(self.before),
                    "after_digest":digest({k:p.detach().float() for k,p in self.params.items()}),
                    "update_norm":squared**0.5,"learning_rates_at_step":self.learning_rates_at_step,
                    "zero_displacement":squared==0.0}
            append_json(self.output/"steps.jsonl",record)
            self.before=None
            return
        after=self._snapshot()
        delta={k:after[k]-v for k,v in self.before.items()}
        if not all(torch.isfinite(d).all() for d in delta.values()):
            raise FloatingPointError("Nonfinite realized Praxis displacement")
        record={"attempt":self.attempt,"status":"applied","scope":self.scope,
                "before_digest":digest(self.before),"after_digest":digest(after),
                "update_norm":dot(delta,delta)**0.5,
                "learning_rates_at_step":self.learning_rates_at_step}
        record["zero_displacement"]=record["update_norm"]==0.0
        if self.gradient is not None:
            record.update(alignment(self.gradient,delta))
        if self.save_delta:
            torch.save(delta,self.output/f"delta-{self.attempt:06d}.pt")
        append_json(self.output/"steps.jsonl",record)
        self.before=None

    def _boundary(self,*args,**kwargs):
        if self.active:
            raise RuntimeError("Reentrant optimizer boundary")
        self.active=True;self.calls=0
        try:
            norm=self.original(*args,**kwargs)
            scalar=float(norm)
            if not self.calls:
                append_json(self.output/"steps.jsonl",{"attempt":self.attempt,"status":"skipped",
                            "reason":"nonfinite_gradient" if not math.isfinite(scalar) else "no_optimizer_call",
                            "scope":self.scope})
            append_json(self.output/"boundaries.jsonl",{"attempt":self.attempt,"optimizer_calls":self.calls,
                        "grad_norm":scalar if math.isfinite(scalar) else None})
            return norm
        except BaseException as error:
            append_json(self.output/"boundaries.jsonl",{"attempt":self.attempt,"status":"failed",
                        "optimizer_calls":self.calls,"error":type(error).__name__})
            raise
        finally:
            self.active=False;self.before=None;self.attempt+=1

    def close(self):
        if self.active:
            raise RuntimeError("Cannot detach inside optimizer step")
        self.pre.remove();self.post.remove()
        self.actor._optimizer_step=self.original
        self.actor._transfer_alignment_recorder=None

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
