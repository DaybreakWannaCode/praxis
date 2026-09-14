# Storage strategy: bounded working set instead of retaining every trial

The previous 1.8 TB forecast was for literal retention of 128 full displacements and two full checkpoints per training run. It is not required by the selection method. Do not expand to 2 TB based on that forecast alone.

## Proposed 250 GB working-set budget

Run candidates and training jobs sequentially. Separate the expensive working set from an archive on a machine/storage tier that does not need a rented GPU.

| Item | Budget, decimal GB | Policy |
|---|---:|---|
| Common warm parent, including optimizer and RNG | 41.28 | Immutable, retained throughout selection/training |
| Previous recovery checkpoint plus replacement being verified | 82.56 | At most two full checkpoint slots for the active run |
| One final model awaiting evaluation/archive | 16.27 | Original FP32 checkpoint; no BF16 downcast |
| One temporary candidate displacement | 32.00 | Covers the approximately 30 GB all-FP64 raw-coordinate bound plus gzip/metadata overhead |
| Preselected audit candidates | 32.00 | Byte-capped, typically four at the measured 6.6 GB; stop or archive if the cap would be exceeded |
| Pretrained cache and tokenizer | 8.00 | One shared copy |
| Durable receipts, rollout inputs and small datasets | 4.00 | Account for actual image downloads before freezing a test set |
| Free-space margin | 16.00 | Refuse work that would consume the reserve |
| **Total planned envelope** | **232.11** | Fits 250 GB only after unrelated/legacy artifacts are archived |

This is an explicit planning envelope, not a measured peak of the new integrated workflow. It assumes checkpoint sizes remain close to the measured 41.27 GB, only one candidate/job is active, and final model copies are moved to an independently verified archive. Larger images, exports, populated optimizer states or archive delays must fail a quota guard, not silently consume the reserve.

The volume already contains about 217 GB of historical artifacts and inputs. Keeping all of that history in place while adding nine final models and two new recovery slots would need roughly 450 GB before headroom; about 500 GB is the alternative if everything must remain on RunPod. We have not purchased storage, removed existing evidence or measured archive-transfer throughput. Moving historical artifacts requires an explicit file inventory, destination capacity and checksum verification before any source removal.

## What survives candidate scoring

For every candidate, retain its ID and text-source indices, full configuration/source/parent identities, captured update inputs and RNG provenance, actual optimizer counters, canonical displacement manifest/digests, alignment/cosine/norms, child scoring outputs, and completion/failure receipts. Keep scoring data distinct from independent development and final outcomes.

A compact score receipt is not a checkpoint. It cannot by itself reconstruct the discarded child. The future lifecycle must capture the original trainer's replay inputs before allowing temporary displacement cleanup. Fresh-rollout exact replay is not assumed. A small audit subset, chosen before looking at scores/outcomes and bounded in bytes, retains complete displacements for independent checks. Missing provenance, failed checks or interrupted scoring must prevent cleanup.

After durable score/provenance copies are verified, future *managed temporary* candidate exports can be removed; existing historical exports remain protected. This bounds full-candidate storage by the scratch slot plus audit budget rather than growing with 128 candidates. Retaining scalar scores alone without their provenance is insufficient.

## Checkpoint publication and final models

The upstream `_save_checkpoint` calls `remove_obsolete_ckpt` before saving the new checkpoint. Do not merely reduce `save_limit` and inherit that failure window. The new workflow must write to a staging directory, verify the complete model/optimizer/scheduler/RNG/dataloader state, publish the new checkpoint and update the pointer atomically, and only then retire the previous checkpoint. Budget both old and replacement during this transaction.

At a completed training run, evaluate its final FP32 model, copy it and its configuration/audit to the archive, verify destination hashes, then free the working copy. Keep only the common parent's optimizer state and the active run's recovery state unless a specific follow-up requires more. Nine final model-only files total about 146 GB, much less than nine pairs of 41 GB full-state checkpoints. Archival saves RunPod space; it does not make transfer time or archive capacity free.

## Implementation and validation on September 15

`streaming_projection.py` reads and verifies displacement chunks, accumulates FP64 alignment/norms and reconstructs the child without loading a whole displacement tensor at once. Failures restore the parent. Seven CPU tests passed, including dense-projection agreement, exact reconstruction, FP64 archives, malformed coverage, corruption and nonfinite-input handling.

`compact_choice_measure.py` holds the shared likelihood gradient in CPU RAM and writes only score/digest receipts. A bounded real-model check is running on the existing one-step candidate; its frozen plan SHA-256 is `947d4c33efc152a443d62cd111e11bad86db762239a45ba8eb18b2108a98e596`. No new training, final-test access or archive deletion is involved. The runner has a 45-minute cap and persistent progress. Large gradient/model output files are prohibited by its output contract.

This scoring check does not yet remove candidate construction/export costs. The original file-based scorer already kept its gradient in RAM; the new contributions are bounded chunk reconstruction and a one-candidate receipt path suitable for a future managed lifecycle. The opt-in input-capture hook now passes five CPU tests, including serialization equality and unchanged input/RNG state. A single-writer checkpoint publication helper passes six failure-ordering tests and syncs data/directory metadata before retiring its previous checkpoint. Both still need original-worker/trainer integration validation. Automatic temporary cleanup, cross-job parent reuse and archive offload are **not yet integrated**. Credit no runtime or full-matrix peak-storage improvement until those stages are measured together.

Next gate: pass the real-model scoring check, capture replay inputs in the candidate adapter, and demonstrate two sequential candidates through the durable-receipt/temporary-cleanup lifecycle. Then revise the execution estimate. The full matrix remains unlaunched.
