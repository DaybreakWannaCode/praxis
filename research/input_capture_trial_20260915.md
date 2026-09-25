# Actual-worker input capture and temporary export trial

One bounded original-Praxis trial completed successfully in 1,414.18 seconds (23.57 minutes), using the same audited 32-prompt batch and warm step-16 parent as the previous candidate-cost measurement. It is an integration check, not a new candidate pool or scientific selection result. The original actor and reward computations are unchanged. A separate code/source snapshot is used, with all Python/config/data inputs frozen by SHA-256.

The trial adds the opt-in actual-worker input/RNG capture and a hard cap on compressed displacement output. Twelve billion bytes is the cumulative cap across all gzip files, including headers/trailers. Hitting the limit aborts the export without a success manifest; partial temporary files are not treated as a valid candidate. Four new export-budget tests passed, together with existing candidate and streaming tests (18 in that group).

Persistent run: `/workspace/praxis/runs/candidate-input-capture-20260915-001`.
Temporary scratch: `/tmp/praxis-input-capture-20260915-001`.
Worker source: `/workspace/praxis-input-capture-original`.
Extension snapshot: `/workspace/praxis-input-capture-code/Praxis-Extension-main`.
Supervisor owner: `940f1d8d9c884a86b31151080880d0ee`; two-hour cap.
The supervisor reports exit 0 and no remaining tagged processes. The GPU was verified idle before the separate scoring job started.

Preflight measured 217,165,527,040 bytes used on the persistent volume and 23,329,136,640 bytes free on the container disk. The launch reserved 2 GB for durable additions plus 16 GiB persistent headroom, and 12 GB for the temporary export plus 4 GiB container headroom. No volume resize or historical deletion occurred. Container scratch is expendable across pod lifecycle changes; captured optimizer inputs and small export receipts are written on the network volume. Losing scratch may require rerunning a candidate; exact replay has not yet been established by this integration check.

The bounded source revisions are committed as `92b3670`; the complete frozen file manifest, rather than an assumed Git checkout identity, identifies the working-pod source snapshot. The original 20260914/earlier source trees were not overwritten. Config SHA-256: `9730cb9592e012418eb35d86ce2c373eab0a5f151da4cf9bd8e441719fc4f9ff`.

`run_input_capture_score.sh` was launched after the completed candidate passed its CPU audit. It requires a successful candidate exit, durable input/export receipt matches and a free GPU before scoring this temporary candidate on the existing 16-image calibration panel. It creates no model or gradient files. No independent development or test result will be inferred from calibration scoring.

`candidate_cleanup.py` adds a guarded release path for managed temporary exports. Eight tests passed: durable-input checks, clean worker completion, score audit and candidate identity checks, symlink protection, interrupted cleanup recovery, and refusal to delete a recreated directory. It flushes the durable receipts and records a cleanup journal before removing only the registered scratch directory. It does not authorize deleting /workspace historical archives. It has not been applied to this live trial.

Next: finish scoring; independently audit/backup the score receipts; then release managed scratch only after live-worker checks pass. A second distinct candidate and full-size checkpoint integration remain outstanding. The full matrix remains unlaunched.

## Verified actual-worker capture and export

The captured original update contains 160 rows and occupies 16,004,368 bytes. Capture took 0.6543 seconds, passed serialization round-trip and unchanged-state checks, and all 37 populated optimizer states advanced from 16 to 17. The CPU audit independently reloaded the capture, checked its input and worker-state digests, required fields and finite floating tensors. This does not establish exact optimizer replay or deterministic fresh rollout generation.

The export contains 824 canonical tensors, 3,754,622,976 unique coordinates, and 6,601,413,568 compressed bytes, under the 12,000,000,000-byte cap. Export and associated validation took 451.84 seconds; the exporter reports exact child reconstruction. Update norm: 0.017204728004674742. The scorer must still consume and verify every raw displacement checksum before cleanup.

The frozen producer reserialized the manifest when saving its durable receipt, dropping a trailing newline. The audit verified semantic equality and saved additional byte-exact receipts without overwriting the original evidence. Future producers now copy receipt bytes directly; the fix is committed as d73d9ea. This metadata issue did not require retraining.

The 6.60 GB scratch export is still retained pending scoring and its independent audit. No historical artifact has been deleted. Candidate integration passed; the complete score-and-release lifecycle remains pending.

## Second candidate preparation

A second 32-prompt batch is frozen under `data/candidate-input-capture-20260915-002` on the pod. It uses the next eligible entries in the fixed source-hash order after excluding the first candidate and baseline train/validation indices, exclusions, and normalized situations. The preparation audit reports zero exact normalized-scene overlap; this is not a semantic/paraphrase or external scene-level audit. No visual score was used to select this batch. Its frozen data/config/audit are also backed up locally.

`run_second_capture.sh` requires the first candidate's completed cleanup journal and absent scratch, then repeats free-GPU, frozen-file and storage-reserve checks. It registers scratch ownership at creation. It retains the same isolated source snapshot and the same audited step-16 parent. Preparation does not launch training.

The cleanup helper now binds its decision to the hashes of independently audited score files, rejecting post-audit changes. Ten tests passed. The live controller additionally requires successful scoring exit, verified local input-backup identities, no tagged trainer workers, no live compact scorer, and an idle GPU. These guards do not replace the outstanding actual score-and-release test.

## Completed first score-and-release lifecycle

Scoring exited 0 after 714.74 seconds, verified all 824 raw displacement tensors, and passed exact parent-forward replay. Alignment was -0.0371968143378855; direct normalized-label log-likelihood change on the same 16-image calibration panel was -0.03729368259897914. This close local agreement validates implementation behavior for this candidate; it is not independent visual correctness or transfer evidence. The likelihood direction is negative for this update. No new model or gradient files were written.

The local backup audit checked parent/export identities, panel rows, direct mean change, alignment/cosine/norm consistency, canonical coverage and child digests. Its score-file hashes were copied back to persistent storage and checked before deletion. The cleanup controller initially refused because the unrelated Nginx service denied environment inspection. The controller now excludes unreadable processes only when their real/effective/saved UIDs all differ from the workload UID; unreadable same-owner processes still block cleanup. No experiment process was killed by cleanup.

Guarded release completed, removing 6,601,870,709 logical scratch bytes (compressed updates plus metadata). A fresh check confirmed scratch absence, retained input checksum, 17,161,090 bytes of persistent run evidence and 23,322,218,496 bytes of free container disk. The cleanup journal was backed up locally. Historical /workspace archives were untouched. Input capture is not a proof of exact future optimizer replay; regeneration can still require remeasurement.

The second guarded candidate launcher was started in tmux `praxis-input-capture-002` after this release. Its completion and second score/release cycle remain pending. The full matrix is not launched.

## Second candidate construction passed

The distinct second batch finished with exit 0 in 1,607.24 seconds (26.79 minutes), leaving no tagged worker processes. Capture saved 15,999,824 bytes in 0.955 seconds and passed unchanged-state and serialization checks. Its CPU audit reloaded all 160 rows and checked digests, finite floating inputs, the 37 populated optimizer counters advancing 16 to 17, and export identities.

The second export contains 824 canonical tensors and 6,575,536,693 compressed bytes under the 12 GB cap. Update norm is 0.01641572724451903. Export and associated validation took 676.02 seconds. This variation relative to the first candidate must be included in future runtime budgets; do not extrapolate only the faster trial. Byte-exact persistent metadata copies were verified through the same audit mechanism.

Scoring was launched in tmux `praxis-input-capture-score-002` after the audit passed. Calibration scoring, independent score-backup audit and second scratch release are pending. There are still no independent visual-transfer results from these integration checks.

## Completed second score-and-release lifecycle

The second scorer exited 0 in 839.27 seconds, verified all 824 displacement tensors and exact parent-forward replay. Alignment was -0.03503486836605385; direct same-panel likelihood change was -0.03507749206367006. The independent local score audit passed and was persisted before guarded cleanup. These are calibration results, not independent visual correctness.

Cleanup removed 6,575,993,835 logical temporary bytes. Fresh verification confirmed scratch absence, retained input checksum, 17,158,973 bytes of durable run evidence and 23,314,268,160 free container bytes. The cleanup journal is backed up locally. Two complete candidate lifecycles now demonstrate bounded temporary retention; exact optimizer replay and full checkpoint publication integration remain unvalidated. Historical artifacts are untouched.
