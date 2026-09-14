# Actual-worker input capture and temporary export trial

One bounded original-Praxis trial is running, using the same audited 32-prompt batch and warm step-16 parent as the previous candidate-cost measurement. It is an integration check, not a new candidate pool or scientific selection result. The original actor and reward computations are unchanged. A separate code/source snapshot is used, with all Python/config/data inputs frozen by SHA-256.

The trial adds the opt-in actual-worker input/RNG capture and a hard cap on compressed displacement output. Twelve billion bytes is the cumulative cap across all gzip files, including headers/trailers. Hitting the limit aborts the export without a success manifest; partial temporary files are not treated as a valid candidate. Four new export-budget tests passed, together with existing candidate and streaming tests (18 in that group).

Persistent run: `/workspace/praxis/runs/candidate-input-capture-20260915-001`.
Temporary scratch: `/tmp/praxis-input-capture-20260915-001`.
Worker source: `/workspace/praxis-input-capture-original`.
Extension snapshot: `/workspace/praxis-input-capture-code/Praxis-Extension-main`.
Supervisor owner: `940f1d8d9c884a86b31151080880d0ee`; two-hour cap.
Live worker PID 62515 was observed restoring the original parent. This is not a completed-run claim.

Preflight measured 217,165,527,040 bytes used on the persistent volume and 23,329,136,640 bytes free on the container disk. The launch reserved 2 GB for durable additions plus 16 GiB persistent headroom, and 12 GB for the temporary export plus 4 GiB container headroom. No volume resize or historical deletion occurred. Container scratch is expendable across pod lifecycle changes; captured optimizer inputs and small export receipts are written on the network volume. Losing scratch may require rerunning a candidate; exact replay has not yet been established by this integration check.

The bounded source revisions are committed as `92b3670`; the complete frozen file manifest, rather than an assumed Git checkout identity, identifies the working-pod source snapshot. The original 20260914/earlier source trees were not overwritten. Config SHA-256: `9730cb9592e012418eb35d86ce2c373eab0a5f151da4cf9bd8e441719fc4f9ff`.

`run_input_capture_score.sh` is prepared but not launched. It requires a successful candidate exit, durable input/export receipt matches and a free GPU before scoring this temporary candidate on the existing 16-image calibration panel. It creates no model or gradient files. No independent development or test result will be inferred from calibration scoring.

`candidate_cleanup.py` adds a guarded release path for managed temporary exports. Eight tests passed: durable-input checks, clean worker completion, score audit and candidate identity checks, symlink protection, interrupted cleanup recovery, and refusal to delete a recreated directory. It flushes the durable receipts and records a cleanup journal before removing only the registered scratch directory. It does not authorize deleting /workspace historical archives. It has not been applied to this live trial.

Next: observe actual input capture and update/export completion; verify serialization, counters, source identities and export receipt; score; independently audit/backup the score receipts; then release managed scratch only after live-worker checks pass. A second distinct candidate and full-size checkpoint integration remain outstanding. The full matrix remains unlaunched.
