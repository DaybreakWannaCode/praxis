# Bounded original-Praxis alignment experiment

This is the next development gate, not the 72-branch study. No final-test images,
data-selection training or new LoRA calibration are authorized by this protocol.

## Frozen reward roles

`production_reward_contract.json` pins the original MCQ reward source and the
visual parser plus its legacy dependency by SHA256. Text training uses the
original returned composite score without coefficient changes. Per-response
correctness, format, tag and length components are checked against the actual
terminal token reward tensor, saved with token IDs and grouped by original GRPO
UID. Group diagnostics include covariance: component variances need not add.

Visual scoring and outcomes use `explicit_final_v2` binary extracted correctness.
The probe uses sequence-summed log probabilities and RLOO with independent draws;
no standard-deviation normalization, token averaging, KL, format or length bonus.
Unconditional correctness is primary. Parse success, conditional correctness and
truncation are separate diagnostics. Conditional correctness is not proof of
improved reasoning. The answer-only likelihood surrogate remains disabled.

The current parser hash is a development lock. An archived fixed 32-response
parent panel was inspected for unparsed responses (four: two invalid final labels,
two absent final choices). This does not prove absence of false-positive parses or
complete the new production audit. Audit a fixed production development panel
before confirming the parser lock; never tune it using final-test outcomes.

## Gate 1: fixed-rollout update parity

The opt-in `patches/praxis-production-parity.patch` wraps the original worker's
`update_actor` entry. It preserves the original loss, clipping, optimizer and
scheduler functions. For one warm input with nonzero next LR it:

1. Checks and records per-response released-code rewards and saves exact input
   tensors, responses, advantages, old/reference log probabilities and metadata.
2. Snapshots optimizer parameters, buffers, full Adam/scheduler state, module modes,
   process RNG and exposed rollout-manager state.
3. Runs the uninstrumented original update, records complete post-state digests,
   restores the parent, and verifies the restored state exactly.
4. Runs the same fixed input through the original update with the observer enabled
   and requires exact agreement in all captured post-state categories.

This gate supports one rank only. Delta norms are measured but giant raw delta
files are omitted for this parity-only run. The source patch disables child
checkpoint saving in this mode to retain the existing warm parent and avoid
rotating it. All activation is explicit through `PRAXIS_PARITY_DIR`; ordinary
Praxis runs are unchanged. Use a fresh directory per attempt. Failed parity stops
the run and attempts to restore the captured parent.

Passing this gate does not establish fresh-rollout replay, driver replay, canonical
visual-gradient coordinate matching, or distributed alignment. Those are the next
gates, not assumptions justified by a fixed-input comparison. The original
first-step zero LR is retained; the parent is the saved nonzero-LR step-two state.

### Executed status (2026-09-07)

The H100, one-rank original full-parameter gate passed using source `cbd78e3`:
all captured control/observer post-state categories matched exactly after warm
restoration. This was a fixed-input integrity check, not visual-transfer evidence.
Detailed logs, reward audit and hashes remain in ignored local
`runs/production-gate-20260907/`. The warm parent was retained.

Follow-up memory changes avoid extra complete post-state snapshots and stream the
parity-only delta norm. Small-model tests check equivalence, intentional corruption,
and restoration on failure. The optimized gate subsequently passed on the GPU
for both fixed candidate batches (`production-parity-004` and `003`), including
complete lossless canonical displacement export. Candidate norms were
0.03164402032795481 and 0.02904879719391052. The recovery run verified every reused
tensor against the replayed update; it did not trust incomplete export files.
`production_coordinates.py` provides a fail-closed one-rank FSDP mapping with
padding exclusion and tied-alias validation. Its value check against the retained
checkpoint passed for 3,754,622,976 canonical elements, with `lm_head.weight`
verified as an alias of `model.embed_tokens.weight`. The full visual backend has
loaded that same layout and completed its first eight-image gradient/projection
pass. Independent visual outcomes are still running; no transfer claim follows
from this integration pass alone.

### Bounded precision implementation

`transfer_alignment.production_precision` consumes four to eight fixed, passed
canonical gate artifacts from the same complete parent. It rejects final-test
rows, overlapping scene families, changed manifest/audit hashes, changed candidate
identities and response counts that differ from a frozen configuration. The
current exporter supports H=1 only; the command explicitly rejects relabelling
these one-step artifacts as H=4. A future H=4 run needs actual four-step total
displacements and a separately frozen budget.

Independent probe repeats estimate Monte Carlo variation in pairwise projected
scores, conditional on the fixed image panel. Paired repeat differences retain
covariance from the shared visual gradient. Approximate Student-t intervals with
few repeats are diagnostics, not exact small-sample guarantees. Across-image
variation is recorded separately and is not replaced by more response repeats.

Parent and child outcomes use a fresh seed for every individual completion and
common random numbers within each image/completion pair. This preserves each
policy's marginal distribution while potentially reducing difference variance.
An independent unchanged-parent control and exact replay of the first parent
image are separate controls. Conservative sign-category Chernoff-KL intervals
cover fixed-panel mean correctness differences under independent response cells,
including heterogeneous image probabilities. In particular, observing no changed
answers does not produce a zero-width confidence interval. Pointwise and
Bonferroni candidate-pair intervals are reported separately. These intervals do
not establish generalization beyond this development panel or parent.

The first 25 contract/statistics/parity tests passed locally and in the original
production runtime. Its only added runtime dependency is SciPy 1.15.3, pinned in
`production-precision-requirements.txt`; existing NumPy 1.26.4 and PyTorch are
unchanged. An additional image-byte integrity guard is tested before deployment.
The larger
response budget and additional candidate batches have not yet been frozen or run;
they will be fixed using integration cost and parser checks before sampling.

### Fixed first integration budget

Before visual scores, the two-branch integration controls were fixed at seed
20260912, eight existing engineering scoring images × four responses and four
existing independent development images × four responses for each of parent,
same-seed parent replay, independent parent repeat and two children (112 responses
total). Scoring IDs: 388, 872, 631, 1026, 731, 899, 985, 398. Development IDs: 973,
1152, 142, 789 (all `viva-` prefixed). The second four-prompt candidate uses seed
20260911, excluding exact normalized original train/validation prompts; source
indices are 9683, 3890, 2858, 944 in the pinned text parquet. No alignment outcome
was used to choose them. Full scene-family auditing remains open.

This closes integration only; it is not the larger production precision check.
The original full-parameter trainer produces displacements; the visual-only
backend has no optimizer and defines one consistent HF/BF16-autocast policy for
sampling and gradient scoring with FP32 master coordinates. Its GPU validation
is pending. Child loading, compressed roundtrips and failure guards have small
CPU regression checks. The visual run has a 45-minute external limit; no favorable
seed retries or extra images are added based on outcomes.

## Subsequent bounded measurement decision

After coordinate/replay validation, freeze four candidate text batches and their
realized H=1 updates. Use independent score-response repeats on fixed, larger
scoring images and an independent development outcome panel with a parent/null
control. Prespecify image IDs, generation settings, parser hashes, response budget,
seeds, stopping budget and uncertainty calculations before generation. Retain
per-image projections/outcomes and report unresolved rankings honestly.

The exact larger image panel is not yet frozen: the engineering split needs a
scene-family audit. Do not launch it merely by interpreting this document as a
complete pre-registration. Fix its size using measured production cost, not
whether preliminary correlations are favorable. H=4 is a single predeclared
fallback, scored by parent gradient dot total four-step displacement; it is not
the first-step score relabeled as a four-step forecast.

Deliver two reproducible full-Praxis branches first, then the bounded precision
decision with score/outcome uncertainty, cost, memory and a locked H=1 or H=4
recommendation. A reliable null is distinct from inadequate measurement precision.
