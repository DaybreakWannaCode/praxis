"""Publish-before-retire checkpoints in a newly owned run directory.

A trainer supplies write/validate callbacks. Validation must check loadability,
finiteness, expected optimizer counters and all continuation state. This helper
handles publication order for a single writer only, and is not yet wired to the original trainer.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_json(path, value):
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    sync_directory(path.parent)


class CheckpointStore:
    def __init__(self, root, *, create=False):
        self.root=Path(root)
        if create:
            self.root.mkdir(parents=True,exist_ok=False)
            atomic_json(self.root/'owner.json',dict(version=1,owner=uuid.uuid4().hex))
            sync_directory(self.root.parent)
        if self.root.is_symlink():raise ValueError('Checkpoint root cannot be a symlink')
        owner=json.loads((self.root/'owner.json').read_text())
        if owner['version']!=1:raise ValueError('Unknown store version')
        self.owner=owner['owner']

    def publish(self, step, write, validate, *, before_retire=None):
        if not isinstance(step,int) or step<0:raise ValueError('Invalid checkpoint step')
        pointer=self.root/'latest.json'
        previous=json.loads(pointer.read_text()) if pointer.exists() else None
        if previous and (previous['owner']!=self.owner or step<=previous['step']):
            raise ValueError('Checkpoint steps must increase in this owned store')
        destination=self.root/f'global_step_{step}'
        if destination.exists():raise FileExistsError(destination)
        pending=self.root/('.pending-'+uuid.uuid4().hex)
        pending.mkdir()
        published=False
        try:
            write(pending)
            if (pending/'publication.json').exists():raise ValueError('Reserved receipt filename')
            validation=validate(pending)
            if validation.get('status')!='passed':raise ValueError('Checkpoint validation did not pass')
            files={}
            for path in sorted(pending.rglob('*')):
                if path.is_symlink():raise ValueError('Checkpoint files cannot be symlinks')
                if path.is_file():
                    h=hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                        os.fsync(stream.fileno())
                    files[str(path.relative_to(pending))]=dict(bytes=path.stat().st_size,sha256=h.hexdigest())
            if not files:raise ValueError('Empty checkpoint')
            for directory in sorted((p for p in pending.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                sync_directory(directory)
            receipt=dict(owner=self.owner,step=step,validation=validation,files=files)
            atomic_json(pending/'publication.json',receipt)
            os.replace(pending,destination)
            published=True
            sync_directory(self.root)
            atomic_json(pointer,dict(owner=self.owner,step=step,path=destination.name))
            # A downstream resume tracker must also publish before retirement.
            if before_retire is not None:
                before_retire(destination)
            # Publication and pointer succeed BEFORE retiring the previous copy.
            if previous:
                old=self.root/previous['path']
                if old.parent!=self.root or old.is_symlink() or old.name!=f"global_step_{previous['step']}":
                    raise ValueError('Unsafe previous checkpoint path')
                old_receipt=json.loads((old/'publication.json').read_text())
                if old_receipt['owner']!=self.owner or old_receipt['step']!=previous['step']:
                    raise ValueError('Previous checkpoint ownership differs')
                shutil.rmtree(old)
                sync_directory(self.root)
            return receipt
        except BaseException:
            if not published and pending.exists():shutil.rmtree(pending)
            # A published checkpoint is retained even if pointer/retirement fails.
            raise
