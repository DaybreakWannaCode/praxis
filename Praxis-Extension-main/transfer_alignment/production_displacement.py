"""Lossless, chunked FP32 canonical displacements for bounded production runs."""
import gzip
from contextlib import contextmanager
import hashlib
import sys
import zlib
from pathlib import Path

import torch

from .experiment import write_json


class ExportBudgetExceeded(OSError):
    pass


@contextmanager
def bounded_gzip(path, budget):
    class LimitedWriter:
        def __init__(self, stream): self.stream=stream
        def write(self, value):
            if budget['used']+len(value)>budget['limit']:
                raise ExportBudgetExceeded('Compressed displacement exceeds its byte budget')
            count=self.stream.write(value)
            budget['used']+=count
            return count
        def __getattr__(self, name): return getattr(self.stream,name)
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=LimitedWriter(raw),compresslevel=1,mtime=0) as stream:
            yield stream


def save_displacement(before, after, output, *, resume_from=None, allow_float64=False, max_output_bytes=None):
    if sys.byteorder!="little":raise ValueError("Little-endian host required")
    if set(before)!=set(after):raise ValueError("Displacement coordinate keys differ")
    if max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes<=0):
        raise ValueError('Positive integer output budget required')
    if max_output_bytes is not None and resume_from is not None:
        raise ValueError('Bounded export cannot reuse external files')
    budget={'limit':max_output_bytes,'used':0}
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
        dtype = torch.float32
        if allow_float64:
            # Some finite FP32 endpoints cannot round-trip through an FP32
            # difference (notably values crossing or approaching zero).
            # Promote only tensors that need it; never weaken reconstruction.
            for start in range(0, b.numel(), 262144):
                bc = b[start:start+262144].cpu()
                ac = a[start:start+262144].cpu()
                if not torch.isfinite(bc).all() or not torch.isfinite(ac).all():
                    raise ValueError(f"Nonfinite endpoint: {name}, offset {start}")
                if not torch.equal(bc + (ac-bc), ac):
                    dtype = torch.float64
                    break
        storage_dtype = "float64" if dtype == torch.float64 else "float32"
        filename=f"{index:05d}.{'f64' if dtype == torch.float64 else 'f32'}.gz"
        def chunks():
            for start in range(0,b.numel(),262144):
                bc=b[start:start+262144].cpu()
                ac=a[start:start+262144].cpu()
                if not torch.isfinite(bc).all() or not torch.isfinite(ac).all():
                    raise ValueError(f"Nonfinite endpoint: {name}, offset {start}")
                delta=ac.to(dtype)-bc.to(dtype)
                if not torch.isfinite(delta).all():
                    raise ValueError(f"Nonfinite delta: {name}, offset {start}, {storage_dtype}")
                if not torch.equal((bc.to(dtype)+delta).float(),ac):
                    raise ValueError(f"Child reconstruction failed: {name}, offset {start}, {storage_dtype}")
                yield delta
        reused=None
        old=Path(resume_from)/filename if resume_from else None
        if old is not None and old.exists():
            partial_squared=0.
            try:
                with gzip.open(old,"rb") as source:
                    for delta in chunks():
                        raw=memoryview(delta.contiguous().numpy())
                        found=source.read(raw.nbytes)
                        if found!=raw.tobytes():raise ValueError("Existing displacement does not match exact replay")
                        sha.update(raw)
                        partial_squared+=(delta.double()*delta.double()).sum().item()
                    if source.read(1):raise ValueError("Existing displacement has trailing values")
                (root/filename).symlink_to(old.resolve())
                squared+=partial_squared
                reused=str(old.resolve())
            except (EOFError,gzip.BadGzipFile,zlib.error):
                # An interrupted gzip stream is retained as evidence in its old
                # directory. Only this new export gets a freshly written copy.
                sha=hashlib.sha256()
        if reused is None:
            with (gzip.open(root/filename,"wb",compresslevel=1) if max_output_bytes is None else bounded_gzip(root/filename,budget)) as dest:
                for delta in chunks():
                    raw=memoryview(delta.contiguous().numpy())
                    sha.update(raw)
                    dest.write(raw)
                    squared+=(delta.double()*delta.double()).sum().item()
        rows.append({"name":name,"shape":list(before[name].shape),"file":filename,
                     "numel":b.numel(),"raw_sha256":sha.hexdigest(),
                     "compressed_bytes":(root/filename).stat().st_size,"reused_from":reused})
        if allow_float64:
            rows[-1]["dtype"] = storage_dtype
    report={"version":2 if allow_float64 else 1,
            "format":"gzip little-endian per-tensor dtype" if allow_float64 else "gzip little-endian float32",
            "parameters":rows,"canonical_numel":sum(x["numel"] for x in rows),
            "update_norm":squared**.5,"child_reconstruction_exact":True}
    write_json(root/"manifest.json",report)
    return report


def load_tensor(root, row):
    import numpy as np
    with gzip.open(Path(root)/row["file"],"rb") as source:
        raw=source.read()
    kind = row.get("dtype", "float32")
    if kind not in ("float32", "float64"):
        raise ValueError("Unsupported displacement dtype")
    width = 8 if kind == "float64" else 4
    if len(raw)!=row["numel"]*width or hashlib.sha256(raw).hexdigest()!=row["raw_sha256"]:
        raise ValueError("Displacement checksum or length differs")
    return torch.from_numpy(np.frombuffer(raw,dtype="<f8" if width == 8 else "<f4").copy()).reshape(row["shape"])
