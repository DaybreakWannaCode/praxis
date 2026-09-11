#!/usr/bin/env bash
# Recreate the archived original-Praxis environment after container replacement.
set -euo pipefail
record=/workspace/praxis/runs/runtime-restore-20260912
mkdir -p "$record"
exec > >(tee -a "$record/restore.log") 2>&1
trap 'rc=$?; echo "$rc" > "$record/restore.exit"' EXIT
export PYTHONNOUSERSITE=1 UV_NO_CACHE=1
test ! -e /opt/praxis-original
uv venv --python /usr/bin/python3.11 /opt/praxis-original
sed '/^flash-attn @/d' /workspace/praxis/runs/original-runtime-001/pip-freeze.txt > "$record/requirements.txt"
uv pip install --python /opt/praxis-original/bin/python -r "$record/requirements.txt"
wheel=flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
curl -fL --retry 2 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$wheel" -o "/tmp/$wheel"
echo "d944fc7d2f962bce83fc4708c2fc0c21eaf8255962a0b350ae919362a51b7ef2  /tmp/$wheel" | sha256sum -c -
uv pip install --python /opt/praxis-original/bin/python --no-deps "/tmp/$wheel"
uv pip install --python /opt/praxis-original/bin/python scipy==1.15.3
uv pip check --python /opt/praxis-original/bin/python
uv pip freeze --python /opt/praxis-original/bin/python > "$record/pip-freeze.txt"
export HF_HUB_OFFLINE=1 HF_HOME=/workspace/cache/huggingface VLLM_USE_V1=0
/opt/praxis-original/bin/python /workspace/check_praxis_runtime.py --source /workspace/original-praxis-clean --config /workspace/praxis-baseline-bounded.yaml --gpu --output "$record/preflight.json"
