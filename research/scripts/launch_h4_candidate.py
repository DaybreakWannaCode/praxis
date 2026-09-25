"""Launch one frozen H4 candidate; invoke inside a persistent tmux session.

Requires an existing isolated Praxis environment and opt-in worker patch. This
launcher does not provision GPUs, change storage, or launch visual evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--train-file', type=Path, required=True)
    p.add_argument('--train-sha256', required=True)
    p.add_argument('--prompt-inventory', required=True)
    p.add_argument('--parent-model', type=Path, required=True)
    p.add_argument('--reward-contract', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--praxis-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=7200)
    p.add_argument('--recovery-gate', type=Path, help='Archived four-step gate; optimizer inputs and states must match')
    a = p.parse_args()
    if not os.environ.get('TMUX'):
        p.error('Run inside tmux so an SSH disconnect cannot interrupt the launcher')
    if not 1 <= a.seconds <= 7200:
        p.error('Candidate runtime must be bounded to at most two hours')
    for path, expected in ((a.config,a.config_sha256),(a.train_file,a.train_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            p.error(f'Frozen input changed: {path}')
    for path in (a.parent_model,a.reward_contract,a.python):
        if not path.is_file():
            p.error(f'Missing required file: {path}')
    a.output.mkdir(parents=True,exist_ok=False)
    env = os.environ.copy()
    for key in ('PRAXIS_FIXED_INPUT','PRAXIS_RESUME_DELTA_DIR','PRAXIS_ALIGNMENT_CAPTURE_DIR','PRAXIS_H4_RECOVERY_DIR'):
        env.pop(key,None)
    env.update(PRAXIS_ALIGNMENT_HORIZON='4', PRAXIS_PARITY_DIR=str(a.output/'gate'),
               PRAXIS_H4_PROMPT_INVENTORY=a.prompt_inventory,
               PRAXIS_PARENT_MODEL=str(a.parent_model), PRAXIS_REWARD_CONTRACT=str(a.reward_contract))
    if a.recovery_gate:
        env['PRAXIS_H4_RECOVERY_DIR'] = str(a.recovery_gate.resolve())
    command=['timeout','--signal=TERM','--kill-after=30s',str(a.seconds),str(a.python),
             '-m','verl.trainer.main',f'config={a.config}']
    start=time.time()
    (a.output/'launch.json').write_text(json.dumps(dict(command=command,start_unix=start,
        config_sha256=a.config_sha256,train_sha256=a.train_sha256,
        prompt_inventory=a.prompt_inventory,timeout_seconds=a.seconds,
        recovery_gate=str(a.recovery_gate.resolve()) if a.recovery_gate else None),indent=2))
    with (a.output/'train.log').open('w') as log:
        result=subprocess.run(command,cwd=a.praxis_root,env=env,stdout=log,stderr=subprocess.STDOUT)
    (a.output/'train.exit').write_text(str(result.returncode)+'\n')
    report_path=a.output/'gate/horizon.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    complete=result.returncode==0 and report==dict(status='passed',horizon=4,completed_steps=4)
    (a.output/'completion.json').write_text(json.dumps(dict(complete=complete,
        train_exit=result.returncode,seconds=time.time()-start,horizon_report=report),indent=2))
    raise SystemExit(0 if complete else 1)


if __name__=='__main__':
    main()
