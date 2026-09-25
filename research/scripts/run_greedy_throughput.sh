#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/workspace/praxis/Praxis-Extension-main:/workspace/original-praxis-clean
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${TMUX:?Use tmux for disconnect safety}"
run_dir=/workspace/praxis/runs/greedy-throughput-20260914-001
mkdir "$run_dir"
set +e
time timeout --signal=TERM --kill-after=60s 1h python -m transfer_alignment.greedy_throughput \
 --config /workspace/praxis/runs/production-h4-preparation/visual-config.json \
 --manifest /workspace/praxis/data/production-precision/visual-manifest.json \
 --output "$run_dir/measurement" > "$run_dir/run.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run_dir/run.exit"
exit "$result"
