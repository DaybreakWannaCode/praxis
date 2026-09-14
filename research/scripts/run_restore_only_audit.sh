#!/usr/bin/env bash
set -euo pipefail
: "${TMUX:?Run inside tmux}"
root=/workspace/praxis-restore-only-20260915
exec 9>"$root/run.lock"
flock -n 9 || exit 3
test ! -e "$root/launcher.json"
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONPATH="$root/code:$root/original"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PRAXIS_RESTORE_ONLY=1 PRAXIS_RESTORE_RUN="$root"
export PRAXIS_RESTORE_PARENT=/workspace/praxis/runs/ordinary-baseline-20260914/checkpoints/global_step_16
python - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
root=Path('/workspace/praxis-restore-only-20260915')
plan=json.loads((root/'plan.json').read_text())
for name,expected in plan['files'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise SystemExit('Source/config drift: '+name)
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
    raise SystemExit('GPU occupied')
if plan['cap_seconds']!=1800:raise SystemExit('Unexpected run cap')
PY
python "$root/launch_ordinary_baseline.py" --directory "$root" --hours 0.5
