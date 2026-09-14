"""Freeze one actual-worker input capture trial using the existing 32-prompt batch."""
import hashlib
import json
from pathlib import Path
import yaml

base=Path('/workspace/praxis')
source=Path('/workspace/praxis-input-capture-original')
code=Path('/workspace/praxis-input-capture-code/Praxis-Extension-main')
data=base/'data/candidate-input-capture-20260915'
data.mkdir(exist_ok=False)
run=base/'runs/candidate-input-capture-20260915-001'
scratch=Path('/tmp/praxis-input-capture-20260915-001')
config=yaml.safe_load((base/'data/candidate-cost-20260914/config.yaml').read_text())
config['trainer']['save_checkpoint_path']=str(scratch/'exports')
config['trainer']['experiment_name']='candidate-input-capture-20260915'
assert config['trainer']['max_steps']==1 and config['worker']['rollout']['n']==5
worker=source/'verl/workers/fsdp_workers.py'
worker.write_text(worker.read_text()+'\nfrom transfer_alignment.candidate_inputs import install_worker_capture\ninstall_worker_capture(FSDPWorker)\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['verl/workers/actor/dp_actor.py','verl/utils/reward_score/mcq.py']:
    assert sha(source/name)==sha(Path('/workspace/original-praxis-clean')/name)
config_path=data/'config.yaml';config_path.write_text(yaml.safe_dump(config,sort_keys=False))
parent_audit=base/'runs/ordinary-baseline-20260914/completion-audit/report.json'
assert json.loads(parent_audit.read_text())['status']=='passed'
paths=[config_path,Path(config['data']['train_files']),parent_audit]
paths+=list(source.rglob('*.py'))+list(code.rglob('*.py'))
plan=dict(scope='one original-worker capture trial; no candidate pool',run=str(run),scratch=str(scratch),
    source=str(source),code=str(code),config=str(config_path),parent_receipt_sha256=sha(parent_audit),
    export_limit_bytes=12000000000,config_sha256=sha(config_path),
    source_basis='Verified working-pod snapshot plus bounded-export and input-capture revisions; file hashes are authoritative',
    source_revision='92b3670 (three updated modules)',files={str(p):sha(p) for p in paths})
(data/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({k:v for k,v in plan.items() if k!='files'},indent=2))
