"""Release only a managed temporary export after durable inputs and scores exist.

This does not permit deletion of historical /workspace archives. The controller
must first establish worker termination; receipts alone cannot prove liveness.
"""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
import shutil

from .checkpoint_publication import atomic_json as save_json, sync_directory


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def release(persistent, *, temporary_base, no_live_workers):
    persistent=Path(persistent).resolve()
    plan_file=persistent/'plan.json';plan=json.loads(plan_file.read_text())
    scratch=Path(plan['scratch'])
    base=Path(temporary_base).resolve()
    system_tmp=Path(tempfile.gettempdir()).resolve()
    if base!=system_tmp and system_tmp not in base.parents:raise ValueError('Scratch base is outside system temporary storage')
    # A direct child of the explicit scratch base; never /workspace archives.
    if scratch.is_symlink() or scratch.resolve().parent!=base or persistent==scratch.resolve() or scratch.resolve() in persistent.parents:
        raise ValueError('Not an isolated temporary export directory')
    if not no_live_workers():raise ValueError('Candidate workers are still live')
    launcher=json.loads((persistent/'launcher.json').read_text())
    if launcher.get('status')!='complete' or launcher.get('exit')!=0 or launcher.get('tagged_processes_remaining'):
        raise ValueError('Candidate did not complete cleanly')
    if (persistent/'run.exit').read_text().strip()!='0':raise ValueError('Candidate exit is not successful')
    capture=json.loads((persistent/'inputs/capture.json').read_text())
    if capture.get('status')!='updated' or not capture.get('serialization_roundtrip_exact') or not capture.get('capture_state_unchanged'):
        raise ValueError('Durable input capture incomplete')
    if capture['input_file_sha256']!=sha(persistent/'inputs/fixed-update-input.pt'):
        raise ValueError('Durable update input checksum differs')
    before,after=capture['before_steps'],capture['after_steps']
    if not before or len(before)!=len(after):raise ValueError('Missing optimizer counters')
    changes=[a-b for a,b in zip(after,before)]
    if min(before)<=0 or max(changes)!=1 or any(v not in (0,1) for v in changes):raise ValueError('Invalid optimizer application count')
    if capture['parent_receipt_sha256']!=plan['parent_receipt_sha256']:
        raise ValueError('Captured parent differs')
    summary=json.loads((persistent/'scoring/summary.json').read_text())
    if summary.get('status')!='complete' or not summary.get('parent_replay_exact') or summary.get('validated_tensors')!=824:
        raise ValueError('Scoring and restoration incomplete')
    if not all(math.isfinite(summary[k]) for k in ('alignment','direct_lookahead')):
        raise ValueError('Scoring is nonfinite')
    if not (persistent/'scoring/local-audit.json').exists() or json.loads((persistent/'scoring/local-audit.json').read_text()).get('status')!='passed':
        raise ValueError('Independent score archive audit is required')
    score_manifest=json.loads((persistent/'scoring/manifest.json').read_text())
    if score_manifest.get('status')!='complete' or Path(score_manifest['plan']['candidate_dir']).resolve()!=(scratch/'exports/global_step_1/actor').resolve():
        raise ValueError('Scores belong to another candidate')
    marker=persistent/'cleanup.json'
    existing=json.loads(marker.read_text()) if marker.exists() else None
    if existing and existing['plan_sha256']!=sha(plan_file):raise ValueError('Cleanup plan changed')
    if not scratch.exists():
        if existing and existing['status'] in ('deleting','complete'):
            existing['status']='complete';save_json(marker,existing);return existing
        raise ValueError('Missing unregistered scratch directory')
    st=scratch.stat()
    if existing:
        if existing['status']=='complete':raise ValueError('Completed scratch path was recreated')
        if existing['device']!=st.st_dev or existing['inode']!=st.st_ino:
            raise ValueError('Scratch directory was replaced')
    else:
        owner=json.loads((scratch/'owner.json').read_text())
        if owner['plan_sha256']!=sha(plan_file) or Path(owner['persistent']).resolve()!=persistent:
            raise ValueError('Scratch ownership differs')
        if any(p.is_symlink() for p in scratch.rglob('*')):raise ValueError('Scratch contains symlinks')
        actor=scratch/'exports/global_step_1/actor'
        for rel,name in [('cost.json','cost.json'),('coordinates.json','coordinates.json'),('delta/manifest.json','delta-manifest.json')]:
            if sha(actor/rel)!=sha(persistent/'export-receipt'/name):raise ValueError('Persistent export receipt differs')
        cost=json.loads((actor/'cost.json').read_text())
        if score_manifest['plan']['parent_model']!=cost['parent_model']:raise ValueError('Scoring parent differs')
        if score_manifest['plan']['artifact_sha256'][str(actor/'delta/manifest.json')]!=sha(persistent/'export-receipt/delta-manifest.json'):
            raise ValueError('Scored delta identity differs')
        if cost['optimizer_state_entries']!=len(before) or cost['incremented_entries']!=sum(v==1 for v in changes) or cost['maximum_optimizer_step_increment']!=1:
            raise ValueError('Export and capture optimizer counters differ')
        if not cost['child_reconstruction_exact'] or cost['status']!='complete':raise ValueError('Invalid export')
        existing=dict(status='deleting',plan_sha256=sha(plan_file),scratch=str(scratch),
                      device=st.st_dev,inode=st.st_ino,
                      removed_logical_bytes=sum(p.stat().st_size for p in scratch.rglob('*') if p.is_file()),
                      retained='Captured update input, source/parent identity, canonical manifest and audited scores')
        for folder in ['inputs','export-receipt','scoring']:
            for path in (persistent/folder).rglob('*'):
                if path.is_symlink():raise ValueError('Durable receipts cannot be symlinks')
                if path.is_file():
                    with path.open('rb') as stream:os.fsync(stream.fileno())
            for directory in sorted([persistent/folder]+[p for p in (persistent/folder).rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
                sync_directory(directory)
        sync_directory(persistent)
        save_json(marker,existing)
    shutil.rmtree(scratch)
    existing['status']='complete';save_json(marker,existing)
    return existing
