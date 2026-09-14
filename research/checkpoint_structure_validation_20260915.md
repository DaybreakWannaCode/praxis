# Original checkpoint structural validator

The original FSDP worker writes model, optimizer and extra state per rank. Extra state includes scheduler plus CPU, CUDA, NumPy and Python RNG. The trainer separately writes the stateful dataloader. Worker-internal saving does not retire old checkpoints; that unsafe ordering is in the original trainer save method addressed by the opt-in adapter.

Implemented `transfer_alignment/original_checkpoint_validation.py` for trusted, single-rank project checkpoints. It compares tensor coordinates/dtypes, optimizer slots and parameter-group mapping against a separately captured reference schema. It scans model and Adam moments for nonfinite values in bounded chunks, rejects negative second moments, checks explicit optimizer/scheduler counters and next learning rate, verifies specified tied aliases, tests CPU/NumPy/Python RNG loading on isolated generators, and checks CUDA RNG and dataloader structure. Empty optimizer slots are explicitly allowed only where the reference schema has them. The baseline has 69 total slots, 37 populated and 32 empty; all 37 populated step-32 counters equal 32. Counter expectations must be supplied explicitly for warm-parent runs and not inferred from the new driver's step number.

Fifteen local tests passed across checkpoint publication, adapter and validator. Validator tests include fractional/wrong Adam counters, NaN/Inf tensors, negative second moments, changed parameter mapping, mismatched tied weights, wrong scheduler epoch, missing RNG/dataloader fields, extra rank files, schema JSON round-trip and no modification of global CPU/NumPy/Python RNG state.

A bounded read-only real-checkpoint audit was launched in tmux session `praxis-checkpoint-validation`, timeout 600 seconds, under `/workspace/praxis-checkpoint-validation-20260915`. It captures the step-16 layout and validates step-32 against it; source checkpoints are never modified or copied. Only schema/report files are written. The latest live check confirmed Python PID 76933 running. This document does not claim a completed real audit until a terminal result is recorded below.

A structural pass cannot prove CUDA RNG compatibility, actual FSDP optimizer loading, dataloader restoration or subsequent rollout replay. The publication adapter is not yet connected to this validator in a production launcher; real worker save/resume and archive capacity remain required before bounded-retention training. Do not delete historical checkpoints based on this audit.

## Save-time quota guard

Added `checkpoint_quota.py`, a preflight callback compatible with the save adapter. It counts each inode once using the larger of logical size and allocated blocks, includes directory allocation, refuses inaccessible/special files rather than silently undercounting, and requires room for the entire replacement plus an explicit reserve while existing checkpoints are still present. It checks that the checkpoint store is within the accounted volume. It does not mistake shared-filesystem free space for the purchased quota and does not reserve space against unrelated concurrent writers.

Four filesystem tests passed: sparse-file accounting, hard-link deduplication, the exact replacement-plus-reserve boundary, and refusal of an out-of-volume store. This guard does not establish a safe production replacement bound; that must be chosen from measured complete checkpoint sizes with headroom before a launch.

Local archive check during this stage found approximately 17.26 GB free on the Mac, with no external volume mounted. That is insufficient for one roughly 41.27 GB full recovery checkpoint. No large archive transfer or deletion was attempted. The running read-only structural audit was rechecked at 3 minutes 21 seconds (Python PID 76933 still live); its ten-minute cap remains unchanged. No duplicate audit was launched.

## Terminal real-checkpoint result

The real step-32 audit completed with status `passed` in 393.681 seconds (6.56 minutes), within its original cap. It checked 825 saved model tensors (including the tied alias), 69 optimizer entries with 37 populated Adam states, optimizer step 32 and scheduler epoch 32. The report and reference schema were copied locally to `runs/checkpoint-structure-validation-20260915/`. Local verification matched the schema SHA-256 `30d6d4b7b960ea5e96afe361516a4749a866a74e69e694da4be9348b843e2428` and validator SHA-256 `7ba496e9cd6b12769140e3353d4fd24bead72b1261fdd066a28ec8db19f8e457` to the completed remote result.

This supersedes all live-process snapshots above: the audit computation is complete, not awaiting restart. Existing model/optimizer files were only read. The result establishes CPU structural/finite-state validation against the step-16 schema and isolated CPU/NumPy/Python RNG loadability. CUDA RNG restoration and actual worker/dataloader continuation remain untested. No real-worker save/resume claim or historical-checkpoint deletion follows from this pass.

## Original-manager CUDA RNG restoration passed

A separate two-minute-capped check on the A100 used the unmodified original `BaseCheckpointManager.load_rng_state` with the saved step-32 extra state. CPU Torch, CUDA Torch, NumPy and Python all reproduced random draws exactly after restoration; advancing without restoration changed all four sequences. The check restored its own prior process RNG state afterward. It completed in 0.713 internal seconds with 4,096 peak CUDA tensor bytes; process/import/SSH startup is outside this timer.

Local backup: `runs/rng-restore-validation-20260915/result.json`. Its executed-script SHA was verified locally: `2d9e8fa08404d99e47ebed101799dc803ec1276233a7ccd992e76a79e8bc14b8`. Original manager SHA: `c9da37082806e122860c2804f38f95e29a30116749137cd7977e36965638a6dd`. Saved extra-state SHA: `7fbafaef84df2f9a2d76e42adbecf5ccfd0a76eaec3041a75260c10e38c5533c`. Environment: Torch 2.5.1+cu124, CUDA 12.4, A100-SXM4-80GB.

This closes the isolated CUDA RNG compatibility/restoration check previously listed as untested. Actual FSDP model/optimizer restore, dataloader continuation and subsequent rollout replay remain separate requirements. No new model/checkpoint or scientific training was produced.

## Original dataloader continuation passed

The actual original trainer `_create_dataloader` method was called with the baseline config, pinned local tokenizer/processor and audited original training Parquet. An uninterrupted CPU pass produced 32 batch digests. A separately constructed loader restored from the saved step-16 dataloader state produced exactly the last 16 uninterrupted digests, covering tensor bytes and non-tensor metadata. The first 16 batches differed, confirming this was continuation rather than an accidental fresh start. The audit completed in 3.215 internal seconds; imports/SSH startup are excluded. No model was loaded or trained and no visual test data was used.

Local report: `runs/dataloader-restore-validation-20260915/result.json`. Its 16 continuation digests and executed-script hash were verified locally. Script SHA: `58b93f704c10211d90944283745a65af05e0cc3fe9530e77dcd74b50a3c4ff95`; saved dataloader state SHA: `c135e937f68c8f9b70cf4866ae8b755a70c71fbcb18677aead9f677331d4541f`; training input SHA: `8ff1442fe4653ef9070adad636c549bfa10c9da16085c0becd4f59464452aee4`.

The isolated dataloader and RNG continuation checks now pass. Full FSDP model/optimizer loading and subsequent update/rollout replay remain unverified as an integrated continuation. Safe save publication still needs actual trainer integration and enough capacity for its two-checkpoint transaction. These results do not authorize historical deletion or the full scientific sweep.

## Loaded model/optimizer comparison prepared

The existing candidate hook was inspected: after the original manager loads, it records populated Adam step counters and checks positive learning rates, but does not compare all loaded values with disk. Added `loaded_checkpoint_audit.py` to close that gap in a future bounded original-worker invocation. It checks RNG and scheduler immediately after load, materializes the original one-rank FSDP model state in CPU RAM, compares every model tensor with the saved model, and compares the optimizer's complete nested state/parameter groups with the saved optimizer. Tensor comparison transfers bounded chunks to CPU and writes only a small result receipt. FSDP model materialization still costs CPU memory; this is not a zero-memory audit.

Three local comparison tests passed, covering nested optimizer/RNG state, a mismatch beyond the first 1,048,576 elements, and wrong dtype, scheduler counter, parameter mapping, key coverage or sequence type. These are comparison-unit tests, not a real-manager invocation. The module has not yet been installed in a worker or launched against the full model. It does not replace the pending integrated model/optimizer restoration check and does not prove subsequent update/rollout replay.

Next integration should invoke this audit immediately after the original manager's load and end before rollout/training/save. Use a separate source snapshot, declare the existing parent, lock one-rank scope, cap runtime, preserve source/config hashes and record clean worker termination. Do not install candidate-export hooks or write another full checkpoint for a restore-only audit.
