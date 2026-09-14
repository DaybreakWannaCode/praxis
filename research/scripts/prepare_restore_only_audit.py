"""On-pod preparation only. Never launches a worker."""
import hashlib
import json
from pathlib import Path
import shutil
import yaml

root=Path('/workspace/praxis-restore-only-20260915')
root.mkdir(exist_ok=False)
source=Path('/workspace/original-praxis-clean')
copied=root/'original'
shutil.copytree(source,copied,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
code=root/'code/transfer_alignment';code.mkdir(parents=True)
(code/'__init__.py').write_text('')
for name in ['loaded_checkpoint_audit.py','restore_only_hooks.py']:
    shutil.copyfile(Path('/workspace/praxis-checkpoint-validation-20260915')/name,code/name)
for relative,addition in [
    ('verl/trainer/ray_trainer.py','from transfer_alignment.restore_only_hooks import install_trainer\ninstall_trainer(RayPPOTrainer)'),
    ('verl/workers/fsdp_workers.py','from transfer_alignment.restore_only_hooks import install_manager\ninstall_manager(FSDPCheckpointManager)')]:
    p=copied/relative;p.write_text(p.read_text()+'\n'+addition+'\n')
config=yaml.safe_load(Path('/workspace/praxis/runs/ordinary-baseline-20260914/config.yaml').read_text())
parent='/workspace/praxis/runs/ordinary-baseline-20260914/checkpoints/global_step_16'
config['trainer'].update(load_checkpoint_path=parent,save_checkpoint_path=str(root/'PROHIBITED'),
    experiment_name='restore-only-20260915')
(root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
paths=list(copied.rglob('*.py'))+list(code.rglob('*.py'))+[root/'config.yaml',Path(__file__)]
plan=dict(status='prepared_not_launched',scope='restore-only original 3B worker, no rollout/update/checkpoint save',
    parent=parent,source=str(copied),code=str(code.parent),cap_seconds=1800,
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({k:v for k,v in plan.items() if k!='files'}))
