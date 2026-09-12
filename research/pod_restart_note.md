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

## Connection interruption during candidate 1

Candidate 0 subsequently completed and passed independent verification of all
824 tensor files. Its result and integrity manifests are backed up locally.
Candidate 1 (second frozen candidate) launched with unchanged inputs and a
two-hour cap. Last successful observation: trainer PID 101269 / worker PID
103347 were active, parent checkpoint restored, first 20-response text rollout
generated. No completed optimizer-step report had yet been observed.

Two subsequent direct-SSH attempts were closed by the endpoint. This does not
establish whether the pod was paused, stopped, or merely unreachable. Do not
restart candidate 1 based on this observation failure. On reconnection inspect
its process, completion.json, train.exit and gate reports first; preserve any
partial attempt before deciding on recovery.

Latest pause-readiness check: direct SSH again closed the connection; the
user-provided RunPod SSH gateway also timed out during banner exchange.
Candidate 1 remains unobserved, not confirmed failed or completed. No duplicate
was launched. Candidate 0's full displacement stays on the network volume;
only its completion/integrity evidence is backed up locally. The Python runtime
under `/opt/praxis-original` is outside the persistent volume and may require
rebuilding after container replacement. Resume requires a working SSH endpoint
and inspection of the existing candidate before starting further work.

## Resumed on replacement A100 — 2026-09-12

RunPod MCP and direct SSH are working. The 150 GB volume in US-KS-2 is mounted
on pod e8lcnuph2r1z3r (A100-SXM4-80GB, reported compute $1.59/hour), SSH
root@216.81.245.126 port 19499. Original H100 hardware changed; record that
distinction in the experiment analysis. No model/reward/config changes.

Candidate 1 completion report survived: exit 0, four steps passed, 2344.50 s.
Independent full-tensor verification is running in `praxis-verify-h4-001`.
The original runtime was rebuilt in /opt/praxis-original from the archived
freeze and checksum-pinned FlashAttention wheel. GPU preflight and 34 tests
passed. Records are at runs/runtime-restore-20260912 on the pod and privately
backed up locally under runs/production-gate-20260907/runtime-restore-20260912.

`praxis-h4-resume` waits for candidate 1 export verification, then runs frozen
candidates 2 and 3 sequentially, each capped at two hours, verifying each
export afterwards. It stops on error and does not launch visual evaluation.
Queue records: /workspace/praxis/runs/h4-resume-20260912/{stage.txt,queue.log,queue.exit}.
If paused, inspect those records and existing candidate directories before
resuming: the queue intentionally refuses to overwrite an existing attempt.
Restoration and queue scripts are saved directly under /workspace as well as
in research/scripts locally. Runtime still resides on ephemeral container
storage; the restoration script is persistent.

Candidate 1 independent verification subsequently passed: all 824 tensor files,
3,754,622,976 canonical elements, 254 FP64 tensors, total update norm
0.09635380528837313. Completion and verification JSON records are backed up in
the local ignored production-h4-001/completed directory. The queue advanced to
training_candidate_2. The final visual launcher is saved as
/workspace/run_h4_precision.sh (not launched); it requires all four independent
export checks, uses the frozen manifest/config, and retains the ten-hour cap.

Candidate 2 step 1 passed control/observer parity in 413.235 s: parameters,
buffers, optimizer, scheduler and worker state all match. Its initial parameter,
buffer, optimizer and scheduler digests also exactly match candidates 0 and 1,
confirming the same warm parent on the replacement A100 (this does not assert
cross-hardware rollout replay). The parity report is backed up locally under
runs/production-gate-20260907/production-h4-002/step-1. Remaining steps and total
displacement export are not yet verified.
