"""Opt-in capture of the actual fixed-rollout input before one actor update.

Small input/RNG receipts supplement a shared immutable parent checkpoint. They
are not a replacement for it, and do not establish fresh vLLM rollout replay.
"""
import copy
from functools import wraps
import hashlib
import os
from pathlib import Path
import time

import torch
from .candidate_throughput import optimizer_steps, save_json
from .core import digest
from .praxis_state import capture_worker_state


def input_digest(data):
    return digest(dict(tensors=dict(data.batch.items()), metadata=data.meta_info,
        non_tensor={k:v.tolist() if hasattr(v,'tolist') else v for k,v in data.non_tensor_batch.items()}))


def capture_inputs(worker, data, root, parent_receipt_sha256):
    if len(parent_receipt_sha256)!=64 or any(c not in '0123456789abcdef' for c in parent_receipt_sha256):
        raise ValueError('Frozen parent audit SHA-256 is required')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    before=input_digest(data)
    state=capture_worker_state(worker,rank=0,world_size=1)
    state_hash=digest(state)
    fixed=copy.deepcopy(data).to('cpu')
    path=root/'fixed-update-input.pt'
    temporary=root/'fixed-update-input.pending'
    with temporary.open('wb') as stream:
        torch.save(dict(data=fixed,worker_state=state),stream)
        stream.flush();os.fsync(stream.fileno())
    # This is our newly created local artifact, not an untrusted pickle.
    loaded=torch.load(temporary,map_location='cpu',weights_only=False)
    if input_digest(loaded['data'])!=before or digest(loaded['worker_state'])!=state_hash:
        raise ValueError('Input serialization round trip differs')
    if input_digest(data)!=before or digest(capture_worker_state(worker,rank=0,world_size=1))!=state_hash:
        raise ValueError('Capture changed input or worker state')
    os.replace(temporary,path)
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    report=dict(status='captured',input_digest=before,worker_state_digest=state_hash,
        parent_receipt_sha256=parent_receipt_sha256,input_file_sha256=h.hexdigest(),
        bytes=path.stat().st_size,seconds=time.monotonic()-started,
        serialization_roundtrip_exact=True,capture_state_unchanged=True,
        limitation='Fixed optimizer inputs and exposed worker state only; no fresh-rollout or completed optimizer replay claim')
    save_json(root/'capture.json',report)
    return report


def install_worker_capture(worker_class):
    if os.environ.get('PRAXIS_CAPTURE_CANDIDATE_INPUTS')!='1':
        raise ValueError('Explicit candidate input capture opt-in required')
    original=worker_class.update_actor
    @wraps(original)
    def update(self,data):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size()!=1:
            raise ValueError('Capture supports one rank only')
        if getattr(self,'_candidate_input_capture_started',False):
            raise ValueError('Bounded capture permits one update per worker')
        self._candidate_input_capture_started=True
        steps=optimizer_steps(self.optimizer)
        if not steps or min(steps)<=0:raise ValueError('A warm parent is required')
        root=Path(os.environ['PRAXIS_CANDIDATE_INPUT_DIR'])
        receipt=capture_inputs(self,data,root,os.environ['PRAXIS_PARENT_RECEIPT_SHA256'])
        result=original(self,data)
        after=optimizer_steps(self.optimizer)
        if len(after)!=len(steps):raise ValueError('Optimizer state coverage changed')
        changes=[a-b for a,b in zip(after,steps)]
        if max(changes)!=1 or any(v not in (0,1) for v in changes):
            raise ValueError('Expected one optimizer application')
        receipt.update(status='updated',before_steps=steps,after_steps=after)
        save_json(root/'capture.json',receipt)
        return result
    worker_class.update_actor=update
