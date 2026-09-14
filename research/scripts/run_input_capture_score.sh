#!/usr/bin/env bash
set -euo pipefail
: "${TMUX:?Run inside tmux}"
export PATH=/opt/praxis-original/bin:$PATH
export PYTHONPATH=/workspace/praxis-input-capture-code/Praxis-Extension-main:/workspace/original-praxis-clean
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HOME=/workspace/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run=/workspace/praxis/runs/candidate-input-capture-20260915-001
python - <<'PY'
import hashlib,json,pathlib,subprocess
root=pathlib.Path('/workspace/praxis/runs/candidate-input-capture-20260915-001')
run=json.loads((root/'launcher.json').read_text())
if run['status']!='complete' or run['exit']!=0 or run['tagged_processes_remaining']:raise SystemExit('Candidate not complete')
if (root/'run.exit').read_text().strip()!='0':raise SystemExit('Failed candidate exit')
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU occupied')
training=json.loads((root/'plan.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in training['files'].items():
 if sha(pathlib.Path(name))!=h:raise SystemExit('Frozen training/source input changed: '+name)
capture=json.loads((root/'inputs/capture.json').read_text())
if capture['status']!='updated' or capture['input_file_sha256']!=sha(root/'inputs/fixed-update-input.pt'):raise SystemExit('Input capture failed')
base=pathlib.Path('/workspace/praxis')
audit_path=base/'runs/ordinary-baseline-20260914/completion-audit/report.json'
audit=json.loads(audit_path.read_text())
if capture['parent_receipt_sha256']!=sha(audit_path):raise SystemExit('Captured parent audit changed')
actor=pathlib.Path(training['scratch'])/'exports/global_step_1/actor'
for relative,copy in [('cost.json','cost.json'),('coordinates.json','coordinates.json'),('delta/manifest.json','delta-manifest.json')]:
 if sha(actor/relative)!=sha(root/'export-receipt'/copy):raise SystemExit('Export receipt changed')
parent=base/'runs/ordinary-baseline-20260914/checkpoints/global_step_16/actor/model_world_size_1_rank_0.pt'
config=base/'runs/production-h4-preparation/visual-config.json';manifest=base/'data/production-precision/visual-manifest.json'
files=[config,manifest,actor/'cost.json',actor/'coordinates.json',actor/'delta/manifest.json']
plan=dict(parent_model=str(parent),parent_sha256=audit['files']['checkpoints/global_step_16/actor/model_world_size_1_rank_0.pt']['sha256'],
 candidate_dir=str(actor),visual_config=str(config),visual_manifest=str(manifest),artifact_sha256={str(p):sha(p) for p in files})
with (root/'score-plan.json').open('x') as stream:json.dump(plan,stream,indent=2)
PY
set +e
time timeout --signal=TERM --kill-after=60s 45m python -m transfer_alignment.compact_choice_measure \
 --plan "$run/score-plan.json" --output "$run/scoring" > "$run/scoring.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run/scoring.exit"
exit "$result"
