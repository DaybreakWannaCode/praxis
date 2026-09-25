#!/usr/bin/env bash
# One frozen H4 visual check; launch inside tmux only after candidates finish.
set -euo pipefail
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/workspace/praxis/Praxis-Extension-main:/workspace/original-praxis-clean
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 VLLM_USE_V1=0
test -n "${TMUX:-}"
python - <<'PY'
import json
from pathlib import Path
for i in range(4):
    p = Path(f'/workspace/praxis/runs/production-h4-{i:03d}')
    v = json.loads((p/'export-verification.json').read_text())
    assert v['status'] == 'passed' and v['all_file_checksums_verified'] is True
PY
python /workspace/praxis/launch_h4_precision.py \
  --config /workspace/praxis/runs/production-h4-preparation/visual-config.json \
  --manifest /workspace/praxis/data/production-precision/visual-manifest.json \
  --output /workspace/praxis/runs/production-h4-precision-001 \
  --python /opt/praxis-original/bin/python
