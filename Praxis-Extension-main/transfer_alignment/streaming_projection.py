"""Exact archived-displacement scoring with bounded temporary memory.

Only scores and per-tensor child digests need be retained after scoring. This
module never removes archives and never calls an update a resumable checkpoint.
"""
import gzip
import hashlib
import math
from pathlib import Path

import numpy as np
import torch


def chunks(root, row, chunk_elements=262144):
    if chunk_elements < 1:
        raise ValueError('Positive chunk size required')
    name = row['file']
    if Path(name).name != name:
        raise ValueError('Displacement filename must be a basename')
    kind = row.get('dtype', 'float32')
    if kind not in ('float32', 'float64'):
        raise ValueError('Unsupported displacement dtype')
    width = 4 if kind == 'float32' else 8
    count = math.prod(row['shape'])
    if count != row['numel'] or count < 0:
        raise ValueError('Displacement shape/size mismatch')
    digest = hashlib.sha256()
    with gzip.open(Path(root)/name, 'rb') as source:
        for start in range(0, count, chunk_elements):
            n = min(chunk_elements, count-start)
            raw = source.read(n*width)
            if len(raw) != n*width:
                raise ValueError('Truncated displacement')
            digest.update(raw)
            value = torch.from_numpy(np.frombuffer(raw, dtype='<f4' if width == 4 else '<f8').copy())
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite displacement')
            yield start, value
        if source.read(1):
            raise ValueError('Trailing displacement bytes')
    if digest.hexdigest() != row['raw_sha256']:
        raise ValueError('Displacement checksum mismatch')


def project_and_apply(gradient, parent, parameters, root, manifest, *, chunk_elements=262144):
    """Return exact-coordinate projection and apply child, or restore on failure.

Gradient and parent stay on CPU; destination parameters may be on GPU. FP64
arithmetic follows the archived likelihood scorer. Reduction order can differ
slightly from a dense tensor dot product. No gradient or child is saved to disk.
"""
    rows = manifest['parameters']
    names = [r['name'] for r in rows]
    if len(names) != len(set(names)) or set(names) != set(parameters) or set(names) != set(gradient):
        raise ValueError('Canonical coordinate coverage mismatch')
    if not set(names).issubset(parent):
        raise ValueError('Missing parent coordinates')
    total = sum(p.numel() for p in parameters.values())
    if total != manifest['canonical_numel']:
        raise ValueError('Canonical scalar count differs')
    alignment = norm2 = gradient2 = 0.
    children = []
    try:
        with torch.no_grad():
            for row in rows:
                name = row['name']
                p, b, g = parameters[name], parent[name], gradient[name]
                if tuple(row['shape']) != tuple(p.shape) or b.shape != p.shape or g.shape != p.shape:
                    raise ValueError('Canonical shape mismatch')
                if b.dtype != torch.float32 or p.dtype != torch.float32 or g.device.type != 'cpu' or b.device.type != 'cpu':
                    raise ValueError('Expected FP32 model endpoints and CPU probe gradient/parent')
                pf, bf, gf = p.view(-1), b.reshape(-1), g.reshape(-1)
                child_digest = hashlib.sha256()
                for start, d in chunks(root, row, chunk_elements):
                    stop = start+d.numel()
                    gd, bd = gf[start:stop].double(), bf[start:stop].double()
                    if not torch.isfinite(gd).all() or not torch.isfinite(bd).all():
                        raise ValueError('Nonfinite gradient or parent')
                    dd = d.double()
                    child = (bd+dd).float()
                    if not torch.isfinite(child).all():
                        raise ValueError('Nonfinite reconstructed child')
                    alignment += float((gd*dd).sum())
                    norm2 += float((dd*dd).sum())
                    gradient2 += float((gd*gd).sum())
                    child_digest.update(memoryview(child.contiguous().numpy()))
                    pf[start:stop].copy_(child)
                children.append(dict(name=name, child_fp32_sha256=child_digest.hexdigest()))
        norm, gnorm = math.sqrt(norm2), math.sqrt(gradient2)
        if not all(math.isfinite(v) for v in (alignment, norm, gnorm)) or norm <= 0 or gnorm <= 0:
            raise ValueError('Invalid projection norms')
        if not math.isclose(norm, manifest['update_norm'], rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError('Displacement norm differs from export')
        return dict(alignment=alignment, update_norm=norm, gradient_norm=gnorm,
                    cosine=alignment/(norm*gnorm), canonical_numel=total,
                    validated_tensors=len(rows), child_tensor_digests=children,
                    chunk_elements=chunk_elements,
                    artifact='Projection receipt, not a model or resumable checkpoint')
    except BaseException:
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(parent[name])
        raise
