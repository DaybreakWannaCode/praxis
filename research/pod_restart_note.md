# Pod pause and recovery note

2026-09-08: first H4 candidate stopped with exit 1 after all four update parity checks passed. Total-displacement export failed; no completed H4 child export exists. No later candidate or H4 visual evaluation ran. The training PIDs were absent and GPU memory/use were zero at the pause-readiness check.

All four fixed rollout inputs, state digests, optimizer records and logs are backed up locally under the ignored `runs/production-gate-20260907/production-h4-000/archive/` directory. The evidence tar SHA256 is `71854dc8d047b96ba98f32334a0c7b07bb5f6e5dc340391ae23507b32271a27c`, verified identical to the remote copy. This archive is scientific evidence and replay input, not a resumable step-four model checkpoint.

Keep the existing network volume. It contains the warm model/Adam checkpoint, base weights, original source, H1 exports and frozen configs. The active Python environment is under `/opt/praxis-original`, outside that volume; after a container replacement it may need rebuilding. Do not assume the old SSH endpoint remains valid after restarting.

On resume: verify the volume mount and files, runtime versions, reward contract and saved parent hashes. Repair exact displacement encoding and preserve the failed attempt. Recover by replaying saved inputs with state matching against the archived step reports; do not silently substitute fresh updates or relax the reconstruction check. Only then finish the remaining frozen candidates and visual evaluation. No automatic main sweep.

Local recovery implementation now accepts `--recovery-gate` in the bounded
candidate launcher. It consumes archived optimizer inputs, verifies all four
archived reports first, compares each current parent before updating, then
compares input and parent/control/observed model, buffer, Adam and scheduler
digests with the archived trajectory. Internal control/observer parity still
includes worker RNG. Cross-process driver RNG replay is explicitly not claimed.
Twenty local tests passed, including four real CPU AdamW recovery updates
with deliberately different discarded driver tensors. Full GPU recovery and
its launcher paths remain to be configured and verified after pod availability.

Recovery started after the pod was confirmed still available: 34 relevant tests
passed in `/opt/praxis-original`. The failed run was preserved as
`/workspace/praxis/runs/production-h4-000-attempt-001`; the recovery writes to the
original frozen candidate output path in tmux `praxis-h4-recovery-000`, capped
at 7,200 seconds. Its config and text input hashes are unchanged. The source
rollouts and four old state reports come from the preserved attempt. This new
process is active, so pausing now would interrupt recovery. Completed evidence
from the failed attempt remains locally archived.
