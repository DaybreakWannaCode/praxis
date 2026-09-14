#!/usr/bin/env bash
set -euo pipefail
: "${TMUX:?Run inside tmux}"
root=/workspace/praxis/data/independent-visual-inventory-20260915
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
python - <<'PY'
import hashlib,json,pathlib,subprocess
root=pathlib.Path('/workspace/praxis/data/independent-visual-inventory-20260915')
plan=json.loads((root/'download-plan.json').read_text())
script=pathlib.Path('/workspace/praxis/research/scripts/download_visual_pool.py')
if hashlib.sha256(script.read_bytes()).hexdigest()!=plan['script_sha256']:raise SystemExit('Acquisition code differs')
if (root/'progress.json').exists() or (root/'acquisition.exit').exists():raise SystemExit('Already started; inspect existing run before resume')
used=int(subprocess.check_output(['du','-s','-B1','/workspace'],text=True).split()[0])
if used+plan['max_logical_image_bytes']+16*1024**3 > 250*10**9:raise SystemExit('Persistent reserve insufficient')
(root/'acquisition-preflight.json').write_text(json.dumps(dict(status='passed',used_bytes=used,reserved_download_bytes=plan['max_logical_image_bytes']),indent=2))
PY
set +e
timeout --signal=TERM --kill-after=15s 45m python /workspace/praxis/research/scripts/download_visual_pool.py \
 --plan "$root/download-plan.json" > "$root/acquisition.log" 2>&1
result=$?
printf '%s\n' "$result" > "$root/acquisition.exit"
exit "$result"
