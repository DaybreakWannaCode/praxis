"""One-time handoff from one identified live initial endpoint to audited final eval."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import psutil
from transfer_alignment.baseline_analysis import endpoint,sha
from transfer_alignment.checkpoint_publication import atomic_json
import transfer_alignment.baseline_analysis as analysis


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial-pid',type=int,required=True)
    p.add_argument('--check-only',action='store_true');p.add_argument('--plan-sha',required=True);p.add_argument('--launcher-sha',required=True);p.add_argument('--audit-sha',required=True)
    a=p.parse_args();root=Path('/workspace/praxis/runs/independent-baseline-endpoints-20260915')
    launcher=Path('/workspace/praxis/research/scripts/run_baseline_endpoint.sh')
    with (root/'handoff-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'handoff-controller.json').exists():raise ValueError('Controller already registered; inspect before any restart')
        def identities():
            if sha(root/'plan.json')!=a.plan_sha or sha(launcher)!=a.launcher_sha or sha(Path(analysis.__file__))!=a.audit_sha:
                raise ValueError('Frozen handoff dependency changed')
        identities()
        process=psutil.Process(a.initial_pid);birth=process.create_time()
        command=process.cmdline()
        if 'transfer_alignment.baseline_endpoints' not in command or '--endpoint' not in command or command[command.index('--endpoint')+1]!='initial':
            raise ValueError('Initial PID is not the intended endpoint')
        if str(root/'plan.json') not in command:raise ValueError('Initial process uses another plan')
        if a.check_only:
            print(json.dumps(dict(status='preflight_passed',initial_pid=a.initial_pid,initial_created=birth,plan_sha256=a.plan_sha)));return
        record=dict(status='waiting_for_initial',initial_pid=a.initial_pid,initial_created=birth,controller_pid=os.getpid(),plan_sha256=a.plan_sha,started_wall=time.time())
        atomic_json(root/'handoff-controller.json',record)
        try:
            while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
                if process.create_time()!=birth:raise ValueError('PID identity changed')
                if time.time()>birth+7320:raise TimeoutError('Initial endpoint exceeded its deadline')
                time.sleep(10)
        except psutil.NoSuchProcess:
            pass
        # The launcher writes its exit receipt immediately after the child exits.
        for _ in range(12):
            if (root/'initial.exit').exists():break
            time.sleep(5)
        identities()
        plan=json.loads((root/'plan.json').read_text());panel_path=Path(plan['panel'])
        if sha(panel_path)!=plan['artifact_sha256'][str(panel_path)]:raise ValueError('Panel changed')
        panel=json.loads(panel_path.read_text())
        _,digests=endpoint(root,'initial',a.plan_sha,panel['items']+panel['shuffled_controls'])
        atomic_json(root/'initial-independent-audit.json',dict(status='passed',responses=len(digests),response_sha256=digests,plan_sha256=a.plan_sha))
        record.update(status='initial_audited_starting_final',initial_audit_sha256=sha(root/'initial-independent-audit.json'))
        atomic_json(root/'handoff-controller.json',record)
        with (root/'final-launch.log').open('x') as log:
            child=subprocess.Popen(['bash',str(launcher),'final'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(status='final_running',final_launcher_pid=child.pid);atomic_json(root/'handoff-controller.json',record)
            try:code=child.wait(timeout=7500)
            except subprocess.TimeoutExpired:
                descendants=psutil.Process(child.pid).children(recursive=True)
                for owned in reversed(descendants):
                    try:owned.terminate()
                    except psutil.NoSuchProcess:pass
                child.terminate()
                _,alive=psutil.wait_procs(descendants,timeout=30)
                for owned in alive:
                    try:owned.kill()
                    except psutil.NoSuchProcess:pass
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise TimeoutError('Final launcher exceeded bounded runtime')
        record.update(status='final_exited' if code==0 else 'final_failed',final_exit=code,finished_wall=time.time())
        atomic_json(root/'handoff-controller.json',record)
        if code:raise RuntimeError('Final endpoint failed')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),flush=True)
        raise
