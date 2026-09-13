#!/usr/bin/env bash
# One isolated cost measurement. Never launches the candidate pool or nine runs.
set -euo pipefail
: "${TMUX:?Use tmux for disconnect safety}"
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/workspace/praxis-candidate-throughput:/workspace/praxis/Praxis-Extension-main
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PRAXIS_SINGLE_CANDIDATE_COST=1
export PRAXIS_CANDIDATE_PARENT=/workspace/praxis/runs/ordinary-baseline-20260914/checkpoints/global_step_16
export PRAXIS_CANDIDATE_COST_DIR=/workspace/praxis/runs/candidate-cost-20260914-001

# All checks are read-only. Refuse to compete with live GPU work or assume
# backing-cluster df capacity is the user's network-volume allocation.
python - <<'PY'
import hashlib,json,os,pathlib,subprocess
baseline=pathlib.Path('/workspace/praxis/runs/ordinary-baseline-20260914')
record=json.loads((baseline/'launcher.json').read_text())
if record.get('status')!='complete' or record.get('exit')!=0 or record.get('tagged_processes_remaining'):
    raise SystemExit('Baseline has not completed and cleaned up successfully')
if (baseline/'run.exit').read_text().strip()!='0':
    raise SystemExit('Baseline exit artifact is not successful')
for step in (16,32):
    actor=baseline/f'checkpoints/global_step_{step}/actor'
    for kind in ('model','optim','extra_state'):
        f=actor/f'{kind}_world_size_1_rank_0.pt'
        if not f.is_file() or f.stat().st_size==0:
            raise SystemExit('Missing parent or endpoint checkpoint: '+str(f))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if gpu: raise SystemExit('GPU is occupied; do not launch a competing cost run')
receipt=json.loads(pathlib.Path('/workspace/praxis/data/ordinary-baseline-20260914/verified-storage.json').read_text())
capacity=receipt.get('size')
if capacity!=250 or receipt.get('id')!='86u1nnngpl':
    raise SystemExit('Need the verified 250 GB project storage receipt')
used=int(subprocess.check_output(['du','-s','-B1','/workspace'],text=True).split()[0])
parent=pathlib.Path(os.environ['PRAXIS_CANDIDATE_PARENT'])/'actor/model_world_size_1_rank_0.pt'
reserve=2*parent.stat().st_size+5*1024**3
if used+reserve>capacity*10**9:
    raise SystemExit('Insufficient quota headroom for conservative FP64 displacement bound')
data=pathlib.Path('/workspace/praxis/data/candidate-cost-20260914')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(data/'config.yaml')!='149615d99b8c1757d83339d4a3489b513ff0a43319f2a37225b71fe3371e0d8c':
    raise SystemExit('Candidate configuration changed')
audit=json.loads((data/'audit.json').read_text())
if audit['count']!=32 or sha(data/'train.parquet')!=audit['train_sha256']:
    raise SystemExit('Candidate data changed')
source=pathlib.Path('/workspace/praxis-candidate-throughput')
manifest=json.loads((source/'telemetry-source-manifest.json').read_text())
for name,item in manifest.items():
    if sha(source/name)!=item['observed']: raise SystemExit('Instrumented source changed: '+name)
    if name in ('verl/utils/reward_score/mcq.py','verl/workers/actor/dp_actor.py') and item['original']!=item['observed']:
        raise SystemExit('Actor or reward implementation changed')
print(json.dumps({'status':'passed','volume_capacity_gb':capacity,'used_bytes':used,'reserved_bytes':reserve}))
PY

mkdir "$PRAXIS_CANDIDATE_COST_DIR"
cp /workspace/praxis/data/candidate-cost-20260914/config.yaml "$PRAXIS_CANDIDATE_COST_DIR/config.yaml"
cp /workspace/praxis-candidate-throughput/telemetry-source-manifest.json "$PRAXIS_CANDIDATE_COST_DIR/"
cp /workspace/praxis/data/candidate-cost-20260914/audit.json "$PRAXIS_CANDIDATE_COST_DIR/data-audit.json"
export PRAXIS_THROUGHPUT_EVENTS="$PRAXIS_CANDIDATE_COST_DIR/events.jsonl"
export PRAXIS_THROUGHPUT_METRICS="$PRAXIS_CANDIDATE_COST_DIR/metrics.jsonl"
python /workspace/praxis/research/scripts/launch_ordinary_baseline.py --directory "$PRAXIS_CANDIDATE_COST_DIR" --hours 2
