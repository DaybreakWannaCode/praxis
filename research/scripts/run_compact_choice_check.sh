#!/usr/bin/env bash
set -euo pipefail
: "${TMUX:?Run inside tmux}"
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/workspace/praxis/Praxis-Extension-main:/workspace/original-praxis-clean
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run_dir=/workspace/praxis/runs/compact-choice-check-20260915-001
python - <<'PY'
import psutil,subprocess
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
    raise SystemExit('GPU is occupied')
if psutil.virtual_memory().available < 100*1024**3:
    raise SystemExit('Less than 100 GiB host memory available')
PY
mkdir "$run_dir"
cp /workspace/praxis/data/compact-choice-check-20260915/plan.json "$run_dir/plan.json"
set +e
time timeout --signal=TERM --kill-after=60s 45m python -m transfer_alignment.compact_choice_measure \
 --plan "$run_dir/plan.json" --output "$run_dir/measurement" > "$run_dir/run.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run_dir/run.exit"
exit "$result"
