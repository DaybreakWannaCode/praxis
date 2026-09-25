"""Read-only launch gate for the fixed ordinary baseline. No GPU initialization."""
import hashlib
import json
from pathlib import Path
import subprocess
import yaml

ROOT=Path('/workspace/praxis')
DATA=ROOT/'data/ordinary-baseline-20260914'
config_path=ROOT/'research/praxis-ordinary-baseline.yaml'
cfg=yaml.safe_load(config_path.read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

preflight=json.loads((DATA/'prompt-preflight.json').read_text())
if not preflight['passed'] or preflight['config_sha256']!=sha(config_path):
    raise SystemExit('Prompt preflight missing or configuration changed')
audit=json.loads((DATA/'audit.json').read_text())
for name,digest in audit['output_sha256'].items():
    if sha(DATA/name)!=digest: raise SystemExit('Audited text data changed')
if (audit['train_rows'],audit['val_rows'])!=(1024,32): raise SystemExit('Coverage changed')
if (cfg['trainer']['max_steps'],cfg['trainer']['total_episodes'],cfg['data']['rollout_batch_size'],
    cfg['worker']['actor']['global_batch_size'],cfg['worker']['rollout']['n'])!=(32,1,32,32,5):
    raise SystemExit('Training budget differs from fixed baseline')
if cfg['trainer']['save_limit']!=2 or cfg['trainer']['save_freq']!=16:
    raise SystemExit('Checkpoint plan changed')
receipt=DATA/'verified-storage.json'
if not receipt.exists(): raise SystemExit('Missing live-verified storage expansion receipt; do not launch')
volume=json.loads(receipt.read_text())
if volume.get('id')!='86u1nnngpl' or volume.get('size',0)<250:
    raise SystemExit('Insufficient verified network-volume capacity')
source=Path('/workspace/praxis-original-throughput')
record=json.loads((source/'telemetry-source-manifest.json').read_text())
for relative,hashes in record.items():
    if sha(source/relative)!=hashes['observed']: raise SystemExit('Instrumented source changed')
    if sha(Path('/workspace/original-praxis-clean')/relative)!=hashes['original']:
        raise SystemExit('Original source changed')
for relative in ('verl/workers/actor/dp_actor.py','verl/utils/reward_score/mcq.py'):
    if record[relative]['original']!=record[relative]['observed']:
        raise SystemExit('Actor or reward differs from original Praxis')
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
    raise SystemExit('GPU is busy; do not overlap workloads')
print(json.dumps(dict(status='passed',config_sha256=sha(config_path),data_audit_sha256=sha(DATA/'audit.json'),
                     capacity_gb=volume['size'],source_manifest_sha256=sha(source/'telemetry-source-manifest.json'))))
