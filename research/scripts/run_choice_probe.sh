#!/usr/bin/env bash
# Bounded development only: no text training and no response generation.
set -euo pipefail
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/workspace/praxis/Praxis-Extension-main:/workspace/original-praxis-clean
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${TMUX:?Run in tmux so disconnecting SSH does not kill measurement}"
run_dir=/workspace/praxis/runs/choice-probe-dev-20260914-002
mkdir "$run_dir"
cp /workspace/praxis/research/choice_probe_development_lock.md "$run_dir/"
set +e
time timeout --signal=TERM --kill-after=60s 2h python -m transfer_alignment.choice_measure \
  --config /workspace/praxis/runs/production-h4-preparation/visual-config.json \
  --manifest /workspace/praxis/data/production-precision/visual-manifest.json \
  --output "$run_dir/measurement" > "$run_dir/run.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run_dir/run.exit"
exit "$result"
