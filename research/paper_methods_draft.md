# Working methods text — development draft

This draft defines the study; it does not claim completed prediction or selection
results. The full-parameter fixed-input observer gate has passed. Production visual
measurements, final split auditing and the confirmatory design freeze remain open.

## Question and objectives

We study when an update induced by text-only reinforcement learning improves a
vision-language model's expected correctness on image-grounded questions. Text
updates follow the pinned released Praxis implementation, including its composite
reward, group-relative advantages, clipping, KL term, optimizer and scheduler.
Visual correctness is a separate objective: a completion receives one if a fixed
parser extracts the correct benchmark choice, and zero otherwise. The visual
probe and independent visual outcome evaluation use the same extraction rule and
generation configuration. Format and length bonuses are excluded from both.

For a parent parameter state θ and a realized text update u, define transfer
alignment A = ∇J_V(θ)ᵀu. Under differentiability and a sufficiently small update,
Taylor expansion gives J_V(θ+u) − J_V(θ) = A + a higher-order remainder. The sign of
A alone does not guarantee a finite-step gain when that remainder is material.
For a multi-step window, u denotes the complete displacement from the same parent
to the window endpoint. We do not substitute the first step's displacement.

The inner product is the primary predictor. Cosine, update norm, text reward and
response length are comparison measurements. Alignment is local and dependent on
the chosen parameter coordinates and visual policy. Numerical precision and
sampling implementation must therefore be recorded; an automatic-differentiation
gradient through finite-precision computation is an approximation, not a proof
that tiny master-weight updates change sampled outputs.

## Visual gradient estimator

For each probe image, draw K ≥ 2 independent completions at the frozen parent.
The weight for completion j is its binary reward minus the mean reward of the
other K−1 completions for that image. Multiply that detached weight by the
gradient of the sequence log probability, summed over generated response tokens
including EOS when emitted. Average over completions and images. There is no
reward-standard-deviation normalization or response-length averaging. Images
with constant sampled reward produce a zero estimated contribution and remain
in the denominator. Record this frequency rather than silently dropping them.

Independent response repeats estimate Monte Carlo uncertainty for fixed probe
images and fixed candidate displacements. They do not by themselves estimate
generalization uncertainty across images, parents or tasks. Preserve per-image
projections so those sources of variation can be assessed separately.

## Controlled branches and intervention

Compare candidate text updates from the same warm parent, including its Adam
moments, scheduler, RNG and required rollout/driver state. Verify replay and
ensure the next learning rate is nonzero. Audit fixed-input observer parity
against the uninstrumented original update. Map visual gradients and realized
updates into the same canonical parameters, excluding padding and duplicate tied
aliases. No image-gradient optimizer step is applied.

Predictive outcomes use independent development images; final-test data remain
untouched during development. Parent/null controls quantify sampling noise and
shared-parent effects. Within-parent comparisons are primary: branches sharing
a parent and images are not independent observations. The bounded development
check will lock the horizon and affordable response budget before confirmatory
runs, even if the result is unresolved or null.

The subsequent selection experiment compares alignment, random and cosine
selection with matched training-update budgets and recorded token/scoring costs.
Because selection uses a labeled visual probe, this is **visual-guided selection
of text-only updates**. It must not be described as learning without any visual
supervision. Higher final accuracy alone does not establish compute efficiency;
candidate-update and visual-scoring costs belong in that comparison.

## Reporting limits and supporting analyses

Unconditional extracted correctness is primary. Parse success, correctness
conditional on parsing and truncation are separate diagnostics. An improvement
in producing usable final answers does not alone demonstrate stronger semantic
reasoning. Report all released text-reward components and group variation to
distinguish correctness-driven from formatting-driven signals.

Training dynamics and retrospective stopping diagnostics are supporting analyses
using preserved checkpoints. An early-stopping method would need its own frozen
rule and independent validation; the current study does not yet make that claim.
Related-work attribution and the exact final sample sizes remain to be completed
before submission. No result placeholders in this draft constitute evidence.
