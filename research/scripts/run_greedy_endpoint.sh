#!/usr/bin/env bash
# Endpoint decoding timing only, on the already-used 32-image dev panel.
set -euo pipefail
: "${TMUX:?Use tmux for disconnect safety}"
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/workspace/praxis/Praxis-Extension-main:/workspace/original-praxis-clean
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
endpoint_sha="$(python - <<'PY'
import json,pathlib,re,subprocess
root=pathlib.Path('/workspace/praxis/runs')
candidate=root/'candidate-cost-20260914-001'
launcher=json.loads((candidate/'launcher.json').read_text())
if launcher.get('status')!='complete' or launcher.get('exit')!=0 or launcher.get('tagged_processes_remaining'):
    raise SystemExit('Candidate cost run must finish and release its workers first')
cost=json.loads((candidate/'exports/global_step_1/actor/cost.json').read_text())
if cost.get('status')!='complete' or not cost.get('child_reconstruction_exact'):
    raise SystemExit('Candidate export audit did not complete')
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
    raise SystemExit('GPU is occupied')
audit=json.loads((root/'ordinary-baseline-20260914/completion-audit/report.json').read_text())
if audit.get('status')!='passed': raise SystemExit('Baseline checkpoint audit missing')
key='checkpoints/global_step_32/actor/model_world_size_1_rank_0.pt'
value=audit['files'][key]['sha256']
if not re.fullmatch('[0-9a-f]{64}',value): raise SystemExit('Invalid frozen checkpoint hash')
print(value)
PY
)"
run_dir=/workspace/praxis/runs/greedy-endpoint-20260914-001
mkdir "$run_dir"
cp /workspace/praxis/runs/ordinary-baseline-20260914/completion-audit/report.json "$run_dir/checkpoint-audit.json"
set +e
time timeout --signal=TERM --kill-after=60s 1h python -m transfer_alignment.greedy_throughput \
 --config /workspace/praxis/runs/production-h4-preparation/visual-config.json \
 --manifest /workspace/praxis/data/production-precision/visual-manifest.json \
 --endpoint-model /workspace/praxis/runs/ordinary-baseline-20260914/checkpoints/global_step_32/actor/model_world_size_1_rank_0.pt \
 --endpoint-sha256 "$endpoint_sha" \
 --output "$run_dir/measurement" > "$run_dir/run.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run_dir/run.exit"
exit "$result"
