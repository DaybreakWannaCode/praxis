#!/usr/bin/env bash
# Dedicated environment only. Safe when invoked from an unrelated activated venv.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/setup_environment.sh /absolute/conda /absolute/new-env-prefix" >&2
  exit 2
fi
task_conda="$1"
task_prefix="$2"
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$task_conda" = /* && "$task_prefix" = /* ]] || { echo "Use absolute paths" >&2; exit 2; }
[[ -x "$task_conda" ]] || { echo "Conda executable missing" >&2; exit 2; }
# A pre-existing unrelated venv may take precedence over conda activation in PATH.
# Never rely on `python` or `pip` resolving to the intended interpreter.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
if [[ ! -x "$task_prefix/bin/python" ]]; then
  "$task_conda" create -p "$task_prefix" python=3.11 pip -y
fi
task_python="$task_prefix/bin/python"
"$task_python" - "$task_prefix" <<'PY'
import pathlib, sys
expected = pathlib.Path(sys.argv[1]).resolve()
if pathlib.Path(sys.prefix).resolve() != expected:
    raise SystemExit(f"Wrong environment prefix: {sys.prefix}")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("This environment recipe requires Python 3.11")
print("Verified dedicated interpreter:", sys.executable, flush=True)
PY
"$task_python" -m pip install --disable-pip-version-check -r "$task_repo/Praxis-Extension-main/transfer_alignment/requirements-gpu.txt"
"$task_python" -m pip check
mkdir -p "$task_repo/runs/environment"
"$task_python" -m pip freeze > "$task_repo/runs/environment/pip-freeze.txt"
cd "$task_repo/Praxis-Extension-main"
# Validation uses CPU even on a busy shared GPU workstation.
CUDA_VISIBLE_DEVICES="" "$task_python" -m unittest discover -s transfer_alignment/tests -v
CUDA_VISIBLE_DEVICES="" "$task_python" -c 'from transformers import Qwen2_5_VLForConditionalGeneration; from peft import LoraConfig; print("Qwen and PEFT imports passed")'
