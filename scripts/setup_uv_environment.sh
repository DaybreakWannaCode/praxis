#!/usr/bin/env bash
# Isolated Python 3.11 setup for templates with uv instead of conda.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/setup_uv_environment.sh /absolute/uv /absolute/python3.11 /absolute/env-prefix" >&2
  exit 2
fi
task_uv="$1"
task_base_python="$2"
task_prefix="$3"
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$task_uv" = /* && "$task_base_python" = /* && "$task_prefix" = /* ]] || { echo "Use absolute paths" >&2; exit 2; }
[[ -x "$task_uv" && -x "$task_base_python" ]] || { echo "Missing uv or Python executable" >&2; exit 2; }
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export UV_LINK_MODE=copy
if [[ ! -x "$task_prefix/bin/python" ]]; then
  "$task_uv" venv --python "$task_base_python" "$task_prefix"
fi
task_python="$task_prefix/bin/python"
"$task_python" - "$task_prefix" <<'PY'
import pathlib, sys
if pathlib.Path(sys.prefix).resolve() != pathlib.Path(sys.argv[1]).resolve():
    raise SystemExit("Unexpected environment prefix")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("This recipe requires Python 3.11")
print("Verified isolated interpreter:", sys.executable, flush=True)
PY
"$task_uv" pip install --python "$task_python" -r "$task_repo/Praxis-Extension-main/transfer_alignment/requirements-gpu.txt" -r "$task_repo/Praxis-Extension-main/transfer_alignment/requirements-data.txt"
"$task_uv" pip check --python "$task_python"
mkdir -p "$task_repo/runs/environment"
"$task_uv" pip freeze --python "$task_python" > "$task_repo/runs/environment/pip-freeze.txt"
cd "$task_repo/Praxis-Extension-main"
CUDA_VISIBLE_DEVICES="" "$task_python" -m unittest discover -s transfer_alignment/tests -v
CUDA_VISIBLE_DEVICES="" "$task_python" -c 'from transformers import Qwen2_5_VLForConditionalGeneration; from peft import LoraConfig; print("Qwen and PEFT imports passed")'
