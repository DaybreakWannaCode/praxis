#!/usr/bin/env bash
# Continue only the two remaining frozen candidates; no visual or main sweep.
set -euo pipefail
record=/workspace/praxis/runs/h4-resume-20260912
mkdir "$record"
exec > >(tee -a "$record/queue.log") 2>&1
trap 'rc=$?; echo "$rc" > "$record/queue.exit"' EXIT
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONPATH=/workspace/praxis/Praxis-Extension-main:/workspace/original-praxis-clean
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
test -n "${TMUX:-}"
test "$(cat /workspace/praxis/runs/runtime-restore-20260912/restore.exit)" = 0
test "$(cat /workspace/praxis/runs/runtime-restore-20260912/tests.exit)" = 0
echo waiting_for_candidate_1_verification > "$record/stage.txt"
for attempt in $(seq 1 180); do
  test ! -f /workspace/praxis/runs/production-h4-001/export-verification.json || break
  tmux has-session -t praxis-verify-h4-001
  sleep 10
done
python -c 'import json; assert json.load(open("/workspace/praxis/runs/production-h4-001/export-verification.json"))["status"] == "passed"'
for candidate in 2 3; do
  root="/workspace/praxis/runs/production-h4-00${candidate}"
  test ! -e "$root"
  echo "training_candidate_$candidate" > "$record/stage.txt"
  bash "/workspace/praxis/runs/production-h4-preparation/launch-candidate-${candidate}.sh"
  echo "verifying_candidate_$candidate" > "$record/stage.txt"
  CUDA_VISIBLE_DEVICES= timeout --kill-after=30s 1800 python /workspace/praxis/verify_h4_export.py "$root/gate" "$root/export-verification.json"
done
echo candidates_complete_visual_not_launched > "$record/stage.txt"
