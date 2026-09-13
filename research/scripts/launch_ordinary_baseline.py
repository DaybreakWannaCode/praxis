"""Supervise one original-Praxis baseline and clean up only its tagged processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import psutil

p=argparse.ArgumentParser()
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--hours',type=float,default=12.)
a=p.parse_args()
owner=uuid.uuid4().hex
env=dict(os.environ,PRAXIS_BASELINE_OWNER=owner)
started=time.monotonic()
record=dict(status='running',owner=owner,started_wall=time.time(),cap_seconds=a.hours*3600)
(a.directory/'launcher.json').write_text(json.dumps(record,indent=2))
with (a.directory/'run.log').open('w') as log:
    proc=subprocess.Popen(['python','-m','verl.trainer.main','config='+str(a.directory/'config.yaml')],env=env,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    code=None
    try:
        while proc.poll() is None:
            if time.monotonic()-started>a.hours*3600:
                code=124
                os.killpg(proc.pid,signal.SIGTERM)
                break
            time.sleep(1)
        if code is None: code=proc.returncode
    finally:
        # A Ray child may survive its driver. Match our unique inherited tag,
        # never all Ray processes or all Python processes on the pod.
        owned=[]
        for child in psutil.process_iter():
            try:
                if child.pid!=os.getpid() and child.environ().get('PRAXIS_BASELINE_OWNER')==owner:
                    owned.append(child)
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        for child in owned:
            try: child.terminate()
            except psutil.NoSuchProcess: pass
        _,alive=psutil.wait_procs(owned,timeout=30)
        for child in alive:
            try: child.kill()
            except psutil.NoSuchProcess: pass
        _,remaining=psutil.wait_procs(alive,timeout=10)
        record.update(status='complete' if code==0 else 'failed',exit=code,
                      elapsed_seconds=time.monotonic()-started,finished_wall=time.time(),
                      tagged_processes_remaining=[x.pid for x in remaining])
        (a.directory/'launcher.json').write_text(json.dumps(record,indent=2))
        (a.directory/'run.exit').write_text(str(code)+'\n')
raise SystemExit(code if code is not None else 1)
