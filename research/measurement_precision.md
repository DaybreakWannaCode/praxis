# Measurement precision follow-up

This supplements the master protocol. It changes how uncertainty and parent/child
outcome differences are measured; it does not change the correctness objective,
parser, text reward or optimizer.

## Two separate sources of uncertainty

1. **Visual-probe Monte Carlo noise:** repeated responses on fixed scoring images
   change the estimated gradient and its projection onto a fixed update.
2. **Outcome sampling noise:** independently sampled parent/child responses can
   obscure a small actual change. Additional image coverage is a third, distinct
   concern; response repeats do not substitute for new independent situations.

A ranking must be allowed to remain unresolved. Sorting noisy point estimates
always produces an ordering, even when no difference is statistically resolvable.

## Implemented fixed-budget diagnostic

`transfer_alignment.precision` uses four prespecified independent scoring repeats,
eight responses per scoring image, and two **previously saved actual** candidate
updates. It logs each image's contribution and checks that their mean agrees with
the global gradient/update dot product. The existing gradient estimator's accumulation
is unchanged when the optional callback is enabled.

The report averages candidate scores across all four repeats. It separately reports:

- Standard error and an approximate two-sided t interval (df=3) for the candidate
  alignment **gap**, conditional on the fixed scoring images.
- Across-image standard error of the repeat-averaged gap. This addresses coverage,
  not solely response sampling.
- An unresolved conditional ranking if the Monte Carlo interval includes zero.

Four repeats and a normal/t approximation are diagnostic, not calibrated
certification. A conditionally resolved ranking does not automatically permit
selection or establish target-population reliability. No optional stopping or
favorable-seed retries are used. Cases with zero sample reward variance remain
in the average with their zero contribution.

## Paired outcomes

For the same development images and batch shapes, sample parent and child using
the same item-level random seeds, independent of scoring and training seeds.
Each policy retains its own sampling distribution; responses are regenerated
under the child, never reused as if they were child samples.

The diagnostic includes a parent/parent exact replay check, a paired actual-child
comparison, and an independent-seed actual-child comparison. All use the saved
child weights, verified against the recorded parent and first candidate delta.
The parent is restored even on failure. No new training is performed.

For an outcome difference, Var(child - parent) = Var(child) + Var(parent) -
2 Cov(child, parent). Positive coupling can reduce variance, but its success must
be measured: sharing seeds alone is not a universal guarantee. This is the
[common-random-numbers approach](https://pubsonline.informs.org/doi/10.1287/mnsc.34.1.65).
A same-policy replay with zero difference verifies determinism only; it does not
establish sensitivity to nonzero changes. A single comparison panel also does not
estimate a reliable variance-reduction factor.

If a tiny actual update produces no changed sampled answers, report that the
endpoint is unresolved at this budget. An empirical standard error of zero from
identical observed differences is not proof of equivalence or zero population
variance. Do not replace the primary sampled-correctness endpoint with answer
likelihood or a training surrogate to manufacture resolution.

## Running it

From `Praxis-Extension-main`, in the existing isolated GPU environment:

```bash
python -m transfer_alignment.precision \
  --config /path/to/config512-batch8-newseed.json \
  --manifest /path/to/fixed-score-dev-manifest.json \
  --parent /path/to/original-run/parent.pt \
  --child /path/to/original-run/branch_0.pt \
  --delta /path/to/original-run/branch_0_step_0_delta.pt \
  --delta /path/to/original-run/branch_1_step_0_delta.pt \
  --output /persistent/new-precision-run
```

The cap is 512 tokens, generation batch size 8. The diagnostic permits up to 16
score images and uses the first eight development records in the supplied fixed
manifest. The manifest records the exact selected records. The current bounded
follow-up uses the same eight score images to isolate response-noise behavior.
`completed.json` is written only after all probes and outcomes finish.

The general branch runner also accepts `evaluation_coupling: "paired"`. Historical
configurations without that field retain `"independent"` behavior. The selected
coupling is recorded in the summary. Scores remain sealed before any child
outcomes, and scoring and outcome sets/seeds remain independent in either mode.

## Next gate

Use the results to set a fixed measurement budget and a practically relevant
minimum difference before a larger study. Expand distinct scoring images to assess
coverage, and evaluate the 1- versus 4-step window under the actual Praxis runtime.
Keep engineering precision checks separate from the main predictive-validity and
selection experiments. Detailed diagnostic outputs stay outside the public repo.

## Versioned answer-parser correction

An audit found that the legacy untagged fallback could select an option mentioned
in the reasoning before an explicit final declaration. For example, `Option A:
this is unsafe ... Final Answer: D.` was parsed as A. The new
`answer_parser: "explicit_final_v2"` profile gives a line-start final answer/option/
choice declaration precedence over that fallback. Tagged answers retain their
existing behavior; invalid or explicitly ambiguous final choices are rejected.
This is a targeted priority fix, not a complete semantic parser.

Historical configurations without `answer_parser` retain `legacy`. New diagnostic
configs explicitly request `explicit_final_v2`. The backend records the effective
profile. The original Task 0 parser and released Praxis reward remain unchanged.
Do not silently relabel archived metrics as if their gradients had used the new
parser: changing a reward can change the RLOO advantage of every response in its
group, and cached response rescoring does not recompute those gradients.

The legacy-parser precision follow-up was stopped after discovery of this
instrument defect, with partial outputs retained. Its cancellation is not a
statistical stopping decision or a favorable-seed retry. The corrected follow-up
uses a new output directory, the same declared fixed-repeat budget and seed, and
the explicitly recorded new parser version.
