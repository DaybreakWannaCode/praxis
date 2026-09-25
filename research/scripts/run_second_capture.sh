#!/usr/bin/env bash
set -euo pipefail
: "${TMUX:?Run inside tmux}"
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONPATH=/workspace/praxis-input-capture-original:/workspace/praxis-input-capture-code/Praxis-Extension-main
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PRAXIS_SINGLE_CANDIDATE_COST=1 PRAXIS_CAPTURE_CANDIDATE_INPUTS=1
export PRAXIS_CANDIDATE_PARENT=/workspace/praxis/runs/ordinary-baseline-20260914/checkpoints/global_step_16
export PRAXIS_CANDIDATE_COST_DIR=/workspace/praxis/runs/candidate-input-capture-20260915-002
export PRAXIS_CANDIDATE_INPUT_DIR="$PRAXIS_CANDIDATE_COST_DIR/inputs"
export PRAXIS_CANDIDATE_EXPORT_ROOT=/tmp/praxis-input-capture-20260915-002
export PRAXIS_CANDIDATE_EXPORT_MAX_BYTES=12000000000
export PRAXIS_PARENT_RECEIPT_SHA256="$(python - <<'PY'
import hashlib,json,pathlib,shutil,subprocess
p=pathlib.Path('/workspace/praxis/data/candidate-input-capture-20260915-002/plan.json')
plan=json.loads(p.read_text())
previous=json.loads(pathlib.Path(plan['prerequisite_cleanup']).read_text())
if previous.get('status')!='complete' or pathlib.Path(previous['scratch']).exists():raise SystemExit('First candidate scratch release required')
for name,expected in plan['files'].items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=expected:raise SystemExit('Frozen input changed: '+name)
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU occupied')
free=shutil.disk_usage('/tmp').free
if free < plan['export_limit_bytes']+4*1024**3:raise SystemExit('Temporary disk reserve insufficient')
used=int(subprocess.check_output(['du','-s','-B1','/workspace'],text=True).split()[0])
if used+2*10**9+16*1024**3 > 250*10**9:raise SystemExit('Persistent-volume reserve insufficient')
if pathlib.Path(plan['run']).exists() or pathlib.Path(plan['scratch']).exists():raise SystemExit('Refuse to reuse run or scratch')
(plan_path:=p.parent/'preflight.json').write_text(json.dumps(dict(status='passed',persistent_used_bytes=used,
    scratch_free_bytes=free,export_cap_bytes=plan['export_limit_bytes']),indent=2)+'\n')
print(plan['parent_receipt_sha256'])
PY
)"
# Command substitution errors must not be hidden by export's exit status.
[[ "$PRAXIS_PARENT_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]
mkdir "$PRAXIS_CANDIDATE_COST_DIR" "$PRAXIS_CANDIDATE_EXPORT_ROOT"
cp /workspace/praxis/data/candidate-input-capture-20260915-002/config.yaml "$PRAXIS_CANDIDATE_COST_DIR/config.yaml"
cp /workspace/praxis/data/candidate-input-capture-20260915-002/plan.json "$PRAXIS_CANDIDATE_COST_DIR/plan.json"
cp /workspace/praxis/data/candidate-input-capture-20260915-002/preflight.json "$PRAXIS_CANDIDATE_COST_DIR/preflight.json"
python - <<'PYOWNER'
import hashlib,json,os,pathlib
run=pathlib.Path(os.environ['PRAXIS_CANDIDATE_COST_DIR'])
scratch=pathlib.Path(os.environ['PRAXIS_CANDIDATE_EXPORT_ROOT'])
with (scratch/'owner.json').open('x') as stream:
 json.dump(dict(persistent=str(run),plan_sha256=hashlib.sha256((run/'plan.json').read_bytes()).hexdigest()),stream,indent=2)
 stream.flush();os.fsync(stream.fileno())
PYOWNER
export PRAXIS_THROUGHPUT_EVENTS="$PRAXIS_CANDIDATE_COST_DIR/events.jsonl"
export PRAXIS_THROUGHPUT_METRICS="$PRAXIS_CANDIDATE_COST_DIR/metrics.jsonl"
python /workspace/praxis/research/scripts/launch_ordinary_baseline.py --directory "$PRAXIS_CANDIDATE_COST_DIR" --hours 2
