# Original checkpoint save adapter: local integration stage

Inspected the deployed original trainer's `_save_checkpoint` and `_load_checkpoint` on September 15. The save method calls `remove_obsolete_ckpt` before actor/dataloader writes and updates `latest_global_step.txt` non-atomically. Resume reads actor state and `dataloader.pt` from an explicitly supplied global_step directory. Frozen trainer copies were not changed.

Implemented an opt-in, per-instance `_save_checkpoint` replacement in `transfer_alignment/original_checkpoint_adapter.py`. It requires an explicitly owned CheckpointStore, actor-only GRPO, matching configured path, quota preflight, serializer and semantic validator. It holds a nonblocking writer lock, calls the original actor worker save interface into staging, saves the dataloader state, validates, publishes the full directory, updates the compatibility tracker with fsync/atomic replacement, then retires the previous owned checkpoint. The original class and historical directories are untouched. The publication helper now exposes a before-retirement callback so a tracker failure cannot erase the old resume target.

Twelve local tests passed across the helper and adapter. New adapter cases exercise the replaced save entrypoint with test workers, original directory layout and tracker naming, validation failure, tracker-write failure preserving both complete copies, quota rejection before worker activity, concurrent writer rejection, and contract/duplicate-install rejection. Test checkpoint bytes and serialization are fixtures, not real Torch optimizer state. This is not yet a real distributed-worker or full-model checkpoint save/resume validation.

Remaining before production use:

1. Supply a real semantic validator covering actor model/optimizer/scheduler/RNG and dataloader continuation state, finiteness, topology and expected optimizer counters. File presence and test-validator success are insufficient.
2. Bind the adapter in a newly owned original-trainer entrypoint with the actual Torch serializer and network-volume quota accounting. Audit worker-internal save/retention behavior too.
3. Verify the actual worker's complete save and resume behavior on a bounded run only after capacity is available. Do not generate a 41 GB checkpoint on the current nearly full 250 GB volume merely to test this adapter.
4. Handle interrupted publication explicitly. A tracker failure can intentionally leave latest.json pointing to the new directory and latest_global_step.txt pointing to the old one, with both complete copies preserved. No automated recovery or cleanup is implemented for that state; inspect receipts before choosing a resume target.
5. Establish and hash-verify a real archival destination before removing any historical checkpoint. Neither the adapter nor its tests creates that destination or reduces existing history.

No new training, checkpoint deletion, pod changes or final-test access occurred in this stage. Full selection remains unlaunched.

## Read-only interrupted-publication inspection

Added `checkpoint_recovery.py` for newly owned checkpoint stores. It takes a shared nonblocking lock, verifies the publication pointer, ownership, full file coverage, byte counts and SHA-256 hashes, and compares the compatibility tracker. A missing/lagging tracker yields `tracker_reconciliation_required` with automatic resume disabled; an ahead tracker or corrupt file is rejected. It does not modify pointers, choose an unpointed orphan, delete files, or assert actual worker resumability. Reconciliation remains explicit rather than silently trusting the highest directory name.

Sixteen tests passed across recovery inspection, publication ordering and the adapter. New recovery cases cover same-size corruption, an interrupted tracker update retaining both complete copies, active-writer refusal, unexpected files and an ahead tracker. This is local filesystem validation; full-size original-worker publication remains untested. The separately completed real-worker restore-only audit verifies loaded values for an existing checkpoint, not this new publication path.
