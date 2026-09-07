# H=4 fallback execution status

H=1 completed with zero of six jointly resolved pairs. The predeclared fallback in
`production_precision_lock.md` therefore applies. Its original protocol remains
unchanged; this document records implementation and execution preparation.

## Frozen inputs

The four candidate text/answer multisets were checked against the actual saved
H=1 optimizer inputs, including five rollout rows per prompt. Each candidate uses
four prompts and repeats that same batch for four consecutive original updates
from `global_step_2`. The driver must generate fresh on-policy rollouts each time.
The worker rejects an inventory different from the frozen H=1 inventory.

The visual configuration retains the H=1 panels, parser, sampling settings and
1,800-response budget. Only the horizon, candidate-export paths and predeclared
seed 20260917 change. Its SHA256 is
`7bce8e2b9bd65435fd35da077a494a789507180be16c5d815c970a0e20288eab`.
Private manifests retain the exact paths and file checksums.

## Implemented checks

The opt-in worker wrapper verifies control/observer equality for each update,
checks parameter/buffer/Adam/scheduler continuity, and requires exactly one applied
optimizer step per driver call. It retains the initial parameters and exports the
total initial-to-fourth-step displacement. The evaluator verifies all four reports
before visual sampling. Incomplete trajectories and relabelled H=1 exports fail.
An unresolved H=4 result terminates this bounded precision study with an
insufficient-budget finding; it does not authorize further sweeps.

Thirty-one relevant tests passed in the original Praxis environment before launch.
These include CPU AdamW trajectory and export checks; they are not GPU evidence
for a four-update original-Praxis run.

## Launch limits and remaining work

Each candidate is capped at two hours plus a 30-second forced-exit allowance.
The visual check is capped at ten hours plus the same allowance. These are
maximum execution bounds, not runtime estimates or authorization for retries.
The launcher requires tmux and verifies frozen file hashes and final completion.

The first H=4 candidate launched on 2026-09-07 at 19:30:43 UTC in tmux
`praxis-h4-000`, with the frozen candidate-0 config and a 7,200-second timeout.
The launcher passed its input checks and Ray initialized. Four-step GPU parity
and the total export remain unverified; candidates 1–3 and visual evaluation
have not launched. Preserve all completed H=1 artifacts. The user confirmed network-volume expansion on 2026-09-08; the exact new quota
has not been independently verified. The pod was idle at the decision review.

## First GPU step evidence

Candidate 0 step 1 passed fixed-rollout control/observer parity: parameters,
buffers, Adam, scheduler and worker state match exactly. Its parity phase took
443.50 seconds. Reports and optimizer-step records were archived privately.
The driver generated a fresh rollout for step 2. This is one passing step,
not a completed four-step chain or visual-transfer result.

## First candidate export failure

All four GPU steps passed control/observer parity and state-chain validation.
The process exited 1 after 2,194.98 seconds when total-displacement export
rejected a nonfinite or non-reconstructing FP32 difference. No visual sampling
ran and no other candidate launched. The GPU was idle after termination.

The exporter assumes FP32 subtraction followed by addition exactly recovers
the child. This is not generally true: a CPU counterexample with parent 1.0
and child float32(1e-8) is finite but fails reconstruction; FP64 succeeds.
That demonstrates an encoding limitation, not proof of this failed tensor
being finite. The existing error does not distinguish the two causes.
Repair must preserve exact child reconstruction, validate finite values, retain
this failed attempt and its cost, and verify any replay against all four saved
step reports. Do not relax the exactness gate or count this as a completed
H4 visual experiment.

## Local encoding repair (not GPU validated)

H4 now permits per-tensor FP64 displacement storage only when FP32 cannot
reconstruct the FP32 child. Every chunk still checks finite endpoints and exact
child reconstruction. Old H1 files keep their FP32 format and remain readable.
The loader, dot product and child application preserve the promoted precision.
Very large endpoint exponent separations can still fail even in FP64; this
remains an explicit error rather than a silently approximate child.

Twenty local tests passed, including cancellation, legacy loading, replay-byte
verification, nonfinite rejection and the four-step CPU AdamW trajectory. The
actual failed GPU tensor is unavailable, so this is a tested repair for a
confirmed encoding limitation, not proof that the failed candidate is recovered.
Promoted tensors increase storage and host-memory requirements; recheck headroom
before a GPU replay. No new GPU run was launched during the local repair.

## GPU recovery trajectory verified

All four recovered optimizer inputs and parameter/buffer/Adam/scheduler
parent/control/observed states matched the preserved original attempt. Each
step also passed its internal control/observer parity including worker RNG.
All four recovery and parity reports are archived locally. The selective-FP64
export started and passed the original failure point; its complete manifest,
reconstruction validation and process exit are still pending. This is recovered
trajectory evidence, not yet a completed H4 candidate export.
