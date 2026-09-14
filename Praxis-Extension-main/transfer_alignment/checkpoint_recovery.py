"""Read-only recovery inspection for an owned checkpoint store. Never deletes files."""
import fcntl
import hashlib
import json
from pathlib import Path
from .checkpoint_publication import CheckpointStore


def verify_published(store, step):
    if type(step) is not int or step < 0: raise ValueError('Invalid checkpoint step')
    root = store.root / f'global_step_{step}'
    if root.is_symlink() or not root.is_dir(): raise ValueError('Missing real checkpoint directory')
    if any(p.is_symlink() for p in root.rglob('*')): raise ValueError('Symlink in published checkpoint')
    receipt = json.loads((root/'publication.json').read_text())
    if receipt['owner'] != store.owner or receipt['step'] != step or receipt['validation']['status'] != 'passed':
        raise ValueError('Checkpoint ownership or validation differs')
    files = receipt['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != root/'publication.json'}
    if not files or set(files) != actual: raise ValueError('Published file coverage differs')
    for name, expected in files.items():
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Unsafe receipt path')
        if path.stat().st_size != expected['bytes']: raise ValueError('Published file size differs')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
        if h.hexdigest() != expected['sha256']: raise ValueError('Published file hash differs')
    return str(root)


def inspect_recovery(root):
    store = CheckpointStore(root)
    # The adapter creates this lock before saving. Do not create anything here.
    with (store.root/'writer.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        pointer_path = store.root/'latest.json'
        tracker_path = store.root/'latest_global_step.txt'
        if pointer_path.is_symlink() or tracker_path.is_symlink(): raise ValueError('Symlink tracker')
        pointer = json.loads(pointer_path.read_text())
        if pointer['owner'] != store.owner or pointer['path'] != f"global_step_{pointer['step']}":
            raise ValueError('Invalid publication pointer')
        latest = pointer['step']
        candidate = verify_published(store, latest)
        tracker = int(tracker_path.read_text().strip()) if tracker_path.exists() else None
        if tracker is not None and tracker != latest:
            if tracker > latest: raise ValueError('Resume tracker is ahead of publication pointer')
            verify_published(store, tracker)
        return dict(status='consistent' if tracker == latest else 'tracker_reconciliation_required',
            published_step=latest, compatibility_tracker_step=tracker, verified_candidate=candidate,
            automatic_resume_allowed=tracker == latest,
            scope='Receipt/file integrity only; no worker resume, pointer edits or checkpoint retirement')
