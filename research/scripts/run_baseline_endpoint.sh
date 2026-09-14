#!/usr/bin/env bash
set -euo pipefail
: "${TMUX:?Run inside tmux}"
endpoint=${1:?initial or final}
case "$endpoint" in initial|final) ;; *) exit 2;; esac
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONPATH=/workspace/praxis-endpoint-code/Praxis-Extension-main
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
root=/workspace/praxis/runs/independent-baseline-endpoints-20260915
exec 9>"$root/gpu.lock"
flock -n 9 || exit 3
python - <<'PY'
import shutil,subprocess
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU occupied')
if shutil.disk_usage('/workspace').free<16*1024**3:raise SystemExit('Persistent reserve below 16 GiB')
PY
# Fail rather than overwrite or silently restart a partial endpoint.
test ! -e "$root/$endpoint"
test ! -e "$root/$endpoint.exit"
set +e
time timeout --signal=TERM --kill-after=60s 2h python -m transfer_alignment.baseline_endpoints \
 --plan "$root/plan.json" --endpoint "$endpoint" --output "$root/$endpoint" > "$root/$endpoint.log" 2>&1
result=$?
printf '%s\n' "$result" > "$root/$endpoint.exit"
sync "$root/$endpoint.exit"
exit "$result"
