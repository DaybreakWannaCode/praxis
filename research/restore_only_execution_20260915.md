# Full original-worker restore-only audit

Prepared a separate copy of original Praxis under `/workspace/praxis-restore-only-20260915/original`, with only opt-in restore-only hook additions. Its private plan records hashes of source, configuration and audit modules. Parent is the existing ordinary-baseline step-16 full checkpoint. The original worker initialization and original checkpoint load remain in use. `fit` calls the original driver's `_load_checkpoint`, validates the loaded-state receipt, then returns. The manager compares all model/Adam/scheduler/RNG values with saved state immediately after loading. Saving is explicitly prohibited. No candidate-export hooks, rollouts, updates or new full checkpoints are allowed.

Six local tests passed for the comparator and restore-only guards. The supervisor was launched in tmux session `praxis-restore-only` with a 1,800-second cap. It tags its child processes, terminates only those tagged descendants on completion/timeout, and writes launcher/exit receipts. A passed worker receipt alone does not prove clean overall completion: require successful driver receipt, launcher exit 0 and no tagged workers remaining.

Remote run root: `/workspace/praxis-restore-only-20260915`. The run is complete; the verified terminal result is below. Do not restart this audit. The 30-minute cap includes original model/reference/rollout-engine initialization and full saved-state comparison. No change to the scientific experiment scope or full-matrix authorization is implied.

Remaining after the successful restore-only result: subsequent update/rollout replay is not tested here, nor is new-checkpoint publication with the real worker. Historical files remain protected; no archive destination with sufficient verified capacity exists yet. This run is a restoration audit, not transfer evidence.

## Completed result

The original full 3B worker restoration audit passed. The loaded-state comparison checked 825 model tensors (4,065,787,904 elements including the tied alias), 6,248,305,189 optimizer tensor elements and 37 populated Adam states, plus exact scheduler and RNG equality. The comparison itself took 12.937 seconds after original loading. The original driver restored step 16 and returned without rollouts, updates or checkpoint saves.

The supervisor completed with exit 0 in 679.087 seconds (11.32 minutes), inside its unchanged 30-minute cap, and reported no tagged processes remaining. An immediate GPU check briefly still listed the defunct worker; the subsequent check showed no GPU compute process and the old worker PID was absent. Treat the job as complete, not waiting or eligible for restart. The pod itself remains running.

Local backups under `runs/restore-only-validation-20260915/` contain plan/config, loaded-state and driver receipts, launcher record, exit code and updated handoff. Local checks verified all passing statuses, exit 0, restored driver step, empty remaining-process list and the executed audit-module SHA `47bbc97f705ca8fac1fd5ec02f0af5357fca00673209273a7a82e1278a2e2ee0`.

This closes the original-worker model/Adam/scheduler/RNG load-equality requirement for the declared single-rank step-16 parent. It does not prove subsequent optimizer-update/rollout replay, new checkpoint save/resume publication, a multi-rank setup or visual transfer. Those claims remain separate. Existing checkpoints were read only; no large checkpoint was written or deleted.
