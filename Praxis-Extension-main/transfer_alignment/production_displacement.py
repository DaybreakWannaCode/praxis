"""Lossless, chunked FP32 canonical displacements for bounded production runs."""
import gzip
import hashlib
import sys
from pathlib import Path

import torch

from .experiment import write_json


def save_displacement(before, after, output):
    if sys.byteorder!="little":raise ValueError("Little-endian host required")
    if set(before)!=set(after):raise ValueError("Displacement coordinate keys differ")
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    rows=[]
    squared=0.
    for index,name in enumerate(sorted(before)):
        b,a=before[name].detach(),after[name].detach()
        if b.shape!=a.shape or b.dtype!=torch.float32 or a.dtype!=torch.float32:
            raise ValueError("Expected matching FP32 master coordinates")
        b,a=b.reshape(-1),a.reshape(-1)
        sha=hashlib.sha256()
        filename=f"{index:05d}.f32.gz"
        with gzip.open(root/filename,"wb",compresslevel=1) as dest:
            for start in range(0,b.numel(),262144):
                bc=b[start:start+262144].cpu()
                ac=a[start:start+262144].cpu()
                delta=ac-bc
                if not torch.isfinite(delta).all() or not torch.equal(bc+delta,ac):
                    raise ValueError("Delta is nonfinite or cannot exactly reconstruct child")
                raw=memoryview(delta.contiguous().numpy())
                sha.update(raw)
                dest.write(raw)
                squared+=(delta.double()*delta.double()).sum().item()
        rows.append({"name":name,"shape":list(before[name].shape),"file":filename,
                     "numel":b.numel(),"raw_sha256":sha.hexdigest(),
                     "compressed_bytes":(root/filename).stat().st_size})
    report={"version":1,"format":"gzip little-endian float32",
            "parameters":rows,"canonical_numel":sum(x["numel"] for x in rows),
            "update_norm":squared**.5,"child_reconstruction_exact":True}
    write_json(root/"manifest.json",report)
    return report


def load_tensor(root, row):
    import numpy as np
    with gzip.open(Path(root)/row["file"],"rb") as source:
        raw=source.read()
    if len(raw)!=row["numel"]*4 or hashlib.sha256(raw).hexdigest()!=row["raw_sha256"]:
        raise ValueError("Displacement checksum or length differs")
    return torch.from_numpy(np.frombuffer(raw,dtype="<f4").copy()).reshape(row["shape"])
