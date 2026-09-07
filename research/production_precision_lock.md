# Bounded production precision check — frozen 2026-09-07 10:06 UTC

This development check follows the completed two-candidate integration and parser
audit. It does not authorize the main branch sweep or selection training. No
final-test images enter the command.

## Fixed inputs and budget

- One retained full-parameter Qwen2.5-VL-3B Praxis parent at original step two,
  including its Adam, scheduler and worker state; next learning rate is nonzero.
- Four fixed text batches of four prompts each. Candidate order is gates
  `004,003,005,006`. The last two were sampled with seed 20260915 before their
  updates or visual scores; their source indices are respectively
  `[3832,1325,7165,6608]` and `[4121,9147,8900,7706]`.
- Released composite text reward remains unchanged. Visual reward is binary
  extracted correctness under the separately pinned `explicit_final_v3` contract.
- Sixteen fixed scoring images, four independent response repeats, four responses
  per image per repeat: 256 probe responses. RLOO uses sequence-summed log
  probabilities without standard-deviation or token-length normalization.
- Thirty-two separate development images, eight responses each for the parent,
  four children and an independently sampled unchanged parent: 1,536 responses.
  Eight additional responses exactly replay the first parent image.
- Total: **1,800 responses**, seed **20260916**, temperature 1, top-p 1, top-k
  disabled, 512 generated-token cap, 200704 maximum image pixels. The integration's
  image prompt is retained. All IDs, image hashes, scene audit, parent location,
  candidate identities and manifest hash are frozen in the private run config.
- Ten-hour external wall-clock cap, with a further 30 seconds for termination.
  At the existing $2.89/hour compute price, this is at most approximately $28.90
  plus termination time and storage. The completed integration is the timing
  reference; this cap is not a prediction that all ten hours will be needed.

The 48-image pool underwent assistant visual screening, source/byte duplicate
checks and perceptual-hash screening. No suspected shared scenes were identified.
This is not independent human annotation or proof about unknown originating
events. The availability-limited pool is a development panel, not an unbiased
sample of the entire benchmark. Existing score/dev assignments were retained;
new assignments were deterministic and preceded precision outcomes.

## Sampling and uncertainty

All candidate projections use each same sampled visual gradient. Four independent
probe repeats quantify response-sampling variation conditional on the score panel.
Pairwise differences are formed before estimating their standard errors; the
shared-gradient covariance must be retained. Approximate 95% Student-t intervals
with three degrees of freedom are reported with their limited-repeat caveat.

Parent and child outcomes share a seed for each individual image/completion pair.
Each completion resets its seed, so a preceding response's EOS length cannot
shift the next stream. The independent unchanged-parent control uses another seed
namespace. Fixed-panel correctness differences receive conservative 95%
sign-category Chernoff-KL intervals. This permits different correctness
probabilities across images and nonzero uncertainty after zero observed changes.
Candidate-pair pointwise intervals and Bonferroni simultaneous intervals are both
reported; the feasibility rule below uses pointwise intervals and is exploratory,
not a familywise confirmatory test. Across-image variation is reported separately.

## Predeclared feasibility and horizon decision

Use an operational resolution of **0.02 expected-correctness units** (two
percentage points), fixed before precision sampling. This is a development
resolution target, not a claim that a single optimizer step must have that effect.
For each of the six candidate pairs, classify each score/outcome interval as:

1. Direction resolved if the entire interval lies strictly above or below zero.
2. Practically tied if the entire interval lies within [-0.02, 0.02].
3. Otherwise unresolved. If both the first and second conditions hold, record
   both; a small nonzero difference may also be practically negligible.

A pair is jointly resolved when both its score and its independent outcome
contrast meet at least one resolution condition. Retain H=1 as the measured
endpoint if at least four of the six pairs are jointly resolved. Report practical
ties as such; this does not imply useful variation for ranking or justify a main
prediction study. Report disagreement between alignment and outcomes even if the
measurement itself is precise. No positive correlation or beneficial transfer is
required to pass this measurement criterion.

If fewer than four pairs are jointly resolved, evaluate the predeclared H=4
alternative once: the same four text batches, each used for four consecutive
original-Praxis updates with fresh on-policy rollouts, from the same warm parent.
Measure the **total** displacement from the initial parent to step four and its
projection on that parent's visual gradient. Do not multiply an H=1 score by four
or select different batches. Keep the same panels, response counts and parser;
use seed 20260917 for its fixed independent measurement. Apply the same resolution
rule and explicitly discuss greater finite-step approximation error. The current
H=1 exporter rejects H=4; a verified total-displacement implementation is required
before that fallback can run.

If H=4 is also unresolved, report that this bounded budget does not support the
proposed comparison. Do not retry favorable seeds, enlarge the study adaptively,
or launch a main sweep. Timeouts, parity failures, parser-contract mismatches and
missing artifacts are engineering incompleteness, not null scientific findings.
An engineering repair may replay the exact frozen inputs, with both attempts
retained and its additional cost recorded.
