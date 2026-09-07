"""Fail-closed mapping of one-rank FSDP flat parameters to canonical names.

Uses version-sensitive FSDP metadata. A manifest alone is not validation: compare
every mapped value with the full checkpoint before using it for visual alignment.
Multi-rank, original-parameter mode and unrecognized padding are rejected.
"""
import math
import torch


def clean(name):
    return ".".join(p for p in name.split(".") if p and p != "_fsdp_wrapped_module")


def flat_segments(flat, prefix, key):
    required=("_fqns", "_shapes", "_numels", "_numels_with_padding", "_is_padding_mask")
    if not all(hasattr(flat,k) for k in required):
        raise ValueError("Unknown FSDP flat-parameter metadata")
    if not len(flat._fqns)==len(flat._shapes)==len(flat._numels):
        raise ValueError("FSDP name/shape metadata differs")
    if len(flat._numels_with_padding)!=len(flat._is_padding_mask):
        raise ValueError("FSDP padding metadata differs")
    offset=0
    index=0
    result=[]
    for size,padding in zip(flat._numels_with_padding,flat._is_padding_mask):
        size=int(size)
        row={"key":key,"offset":offset,"numel":size,"padding":bool(padding)}
        if not padding:
            if index>=len(flat._fqns) or size!=flat._numels[index] or size!=math.prod(flat._shapes[index]):
                raise ValueError("FSDP segment size differs")
            row.update(name=clean(prefix+"."+flat._fqns[index]),shape=list(flat._shapes[index]))
            index+=1
        result.append(row)
        offset+=size
    if offset!=flat.numel() or index!=len(flat._fqns):
        raise ValueError("Full FSDP flat parameter is not available on this rank")
    return result


def manifest(worker):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from .production_parity import parameters
    if torch.distributed.get_world_size()!=1:
        raise ValueError("Canonical mapping currently requires one rank")
    params=parameters(worker)
    ids={id(p):k for k,p in params.items()}
    if len(ids)!=len(params):raise ValueError("Duplicate optimizer coordinates")
    seen=set()
    segments=[]
    aliases={}
    for prefix,module in worker.fsdp_module.named_modules():
        if not isinstance(module,FSDP) or module._handle is None:continue
        if module.sharding_strategy!=ShardingStrategy.NO_SHARD or module._use_orig_params:
            raise ValueError("Only one-rank NO_SHARD flat parameters are supported")
        flat=module._handle.flat_param
        key=ids.get(id(flat))
        if key is None or key in seen:raise ValueError("Optimizer and FSDP handles differ")
        seen.add(key)
        segments.extend(flat_segments(flat,prefix,key))
        for info in flat._shared_param_infos:
            alias=clean(".".join((prefix,info.module_name,info.param_name)))
            primary=clean(".".join((prefix,info.prim_module_name,info.prim_param_name)))
            if alias in aliases:raise ValueError("Repeated shared alias")
            aliases[alias]=primary
    names=[r["name"] for r in segments if not r["padding"]]
    if seen!=set(params) or len(set(names))!=len(names) or set(names)&set(aliases):
        raise ValueError("Canonical mapping does not cover optimizer exactly once")
    if not set(aliases.values())<=set(names):raise ValueError("Unknown tied primary")
    return {"version":1,"torch":torch.__version__,"world_size":1,
            "segments":segments,"aliases":aliases,
            "canonical_numel":sum(r["numel"] for r in segments if not r["padding"])}


def canonical_views(flat_values, mapping):
    """Zero-copy views; padding and tied aliases are excluded from dot products."""
    result={}
    for row in mapping["segments"]:
        if row["padding"]:continue
        x=flat_values[row["key"]].reshape(-1)
        view=x.narrow(0,row["offset"],row["numel"]).view(row["shape"])
        if row["name"] in result:raise ValueError("Duplicate canonical coordinate")
        result[row["name"]]=view
    return result


def verify_values(flat_values, mapping, full_state):
    """Require bitwise value/shape/dtype equality, including every tied alias."""
    views=canonical_views(flat_values,mapping)
    for name,value in views.items():
        expected=full_state[name]
        if value.shape!=expected.shape or value.dtype!=expected.dtype:
            raise ValueError(f"Canonical shape/dtype mismatch: {name}")
        if not torch.equal(value.detach().cpu(),expected.detach().cpu()):
            raise ValueError(f"Canonical value mismatch: {name}")
    for alias,primary in mapping["aliases"].items():
        a,b=full_state[alias],full_state[primary]
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
            raise ValueError(f"Tied checkpoint values differ: {alias}")
    return {"validated_parameters":len(views),"validated_aliases":len(mapping["aliases"]),
            "canonical_numel":sum(v.numel() for v in views.values())}
