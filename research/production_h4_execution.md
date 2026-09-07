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

H=4 has not launched. Confirm network-volume headroom, deploy the launcher, and
run the first frozen candidate to validate the full GPU path before proceeding
with the other three. Preserve all completed H=1 artifacts. The user confirmed network-volume expansion on 2026-09-08; the exact new quota
has not been independently verified. The pod was idle at the decision review.
