"""Bounded read-only audit; run beside the pinned validator in an isolated directory."""
import hashlib
import json
import time
from pathlib import Path
import torch
from original_checkpoint_validation import capture_schema, validate_checkpoint

root = Path('/workspace/praxis-checkpoint-validation-20260915')
base = Path('/workspace/praxis/runs/ordinary-baseline-20260914/checkpoints')
if (root / 'result.json').exists():
    raise RuntimeError('Audit already has a result; do not overwrite')
started = time.monotonic()
torch.set_num_threads(4)
try:
    schema = capture_schema(base / 'global_step_16')
    schema_bytes = (json.dumps(schema, indent=2, allow_nan=False)+'\n').encode()
    with (root / 'reference-schema.json').open('xb') as stream:
        stream.write(schema_bytes)
    result = validate_checkpoint(base / 'global_step_32', schema, optimizer_step=32,
        scheduler_epoch=32, expected_lr=1e-6,
        tied_aliases=[('lm_head.weight', 'model.embed_tokens.weight')])
    result.update(schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        reference=str(base / 'global_step_16'), target=str(base / 'global_step_32'))
except Exception as exc:
    result = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)}
result.update(elapsed_seconds=time.monotonic()-started,
    validator_sha256=hashlib.sha256((root/'original_checkpoint_validation.py').read_bytes()).hexdigest())
with (root / 'result.json').open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['status'] == 'passed' else 1)
