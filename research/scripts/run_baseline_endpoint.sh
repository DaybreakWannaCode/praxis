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
import os,subprocess
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU occupied')
# This mount reports the shared filesystem capacity, not the purchased quota.
# Conservatively count allocated or logical bytes once per inode against 250 GB.
seen=set();used=0
for directory,_,files in os.walk('/workspace',followlinks=False):
    for name in files:
        path=os.path.join(directory,name)
        if os.path.islink(path):continue
        info=os.stat(path);key=(info.st_dev,info.st_ino)
        if key in seen:continue
        seen.add(key);used+=max(info.st_size,info.st_blocks*512)
if 250_000_000_000-used<16*1024**3:raise SystemExit('Purchased-volume reserve below 16 GiB')
print('Persistent accounted bytes:',used)
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
