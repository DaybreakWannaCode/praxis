#!/usr/bin/env bash
# One ordinary baseline, never the nine-arm matrix.
set -euo pipefail
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/workspace/praxis-original-throughput
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${TMUX:?Use tmux for disconnect safety}"
python /workspace/praxis/research/scripts/preflight_ordinary_launch.py
run_dir=/workspace/praxis/runs/ordinary-baseline-20260914
mkdir "$run_dir"
cp /workspace/praxis/research/praxis-ordinary-baseline.yaml "$run_dir/config.yaml"
cp /workspace/praxis-original-throughput/telemetry-source-manifest.json "$run_dir/"
export PRAXIS_THROUGHPUT_EVENTS="$run_dir/events.jsonl"
export PRAXIS_THROUGHPUT_METRICS="$run_dir/metrics.jsonl"
python -c 'import json,time,os; from pathlib import Path; Path(os.environ["PRAXIS_THROUGHPUT_EVENTS"]).write_text(json.dumps({"event":"start","phase":"launcher","wall_time":time.time()})+"\n")'
set +e
python /workspace/praxis/research/scripts/launch_ordinary_baseline.py --directory "$run_dir" --hours 12
result=$?
printf '%s\n' "$result" > "$run_dir/run.exit"
export PRAXIS_BASELINE_EXIT="$result"
python -c 'import json,time,os; p=os.environ["PRAXIS_THROUGHPUT_EVENTS"]; f=open(p,"a"); f.write(json.dumps({"event":"end","phase":"launcher","wall_time":time.time(),"exit":int(os.environ["PRAXIS_BASELINE_EXIT"])})+"\n"); f.close()'
exit "$result"
