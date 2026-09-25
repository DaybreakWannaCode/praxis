# Decision after the H=1 precision check

Date: 2026-09-08. User confirms network volume expanded; the exact new quota was not independently verified. Live SSH check found H100 idle and no first H=4 output directory. No experiment was launched for this decision review.

## Evidence from archived measurements

The original full-parameter Praxis H=1 experiment completed, not a LoRA run. Parent replay passed. The frozen result remains 0/6 jointly resolved pairs. The following reanalysis uses existing summary.json only; no model responses were added.

Values below are percentage points (alignment multiplied by 100 is first-order predicted reward change). Intervals are pointwise 95%; score intervals are approximate Student-t intervals from four response repeats. Outcome intervals use the frozen paired fixed-panel procedure.

| Pair | Alignment difference [interval] | Visual outcome difference [interval] | Uncertain direction |
|---|---:|---:|---|
| 0–1 | 0.55 [-0.15, 1.25] | 3.91 [-6.25, 13.90] | Both directions uncertain |
| 0–2 | 1.29 [0.22, 2.36] | 1.95 [-8.75, 12.57] | Outcome only |
| 0–3 | 0.95 [0.05, 1.84] | 5.47 [-4.41, 15.11] | Outcome only |
| 1–2 | 0.74 [0.33, 1.15] | -1.95 [-12.76, 8.95] | Outcome only |
| 1–3 | 0.40 [-0.19, 0.98] | 1.56 [-8.83, 11.88] | Both directions uncertain |
| 2–3 | -0.34 [-0.87, 0.18] | 3.52 [-5.58, 12.46] | Both directions uncertain |

Three score differences have pointwise distinguishable directions; none survives the saved six-comparison Bonferroni analysis. All six score comparisons pass the frozen direction-or-practical-tie rule, but ties are not evidence of useful ranking variation. All six outcome comparisons fail that rule. Thus the immediate frozen-criterion bottleneck is visual outcomes, while reliable score ranking also remains limited.

Observed outcome contrasts are 1.56–5.47 percentage points in absolute value, with interval half-widths roughly 9–11 points. Useful gains remain plausible, but undetectable here. Four candidates from one parent cannot support credible power estimates across training trajectories. Six pairwise contrasts are not six independent experiments. A future aggregate design needs parent-level replication and simulation that preserves shared score probes, images, and updates; do not replace the frozen criterion after seeing this result.

## Decision and execution order

1. Finish the already-frozen H=4 check once. It is launch-ready and directly addresses the observed outcome bottleneck. Start with the first bounded candidate to verify four real GPU optimizer applications, positive learning rates, state continuity and exact total-displacement export before the other three. Score the complete four-step displacement at the initial parent, seal scores, then run independent development outcomes. Keep all reward/parser/panel/count/seed decisions fixed. Estimated total is 10–12 GPU hours based on H=1 timing, not a measured H=4 duration; budget bounds remain those of the existing lock.
2. Prepare the ordinary one-stage before/after development baseline while H=4 executes; run it next on the same GPU. This is the next substantive reproduction question regardless of whether the local precision criterion succeeds. The existing two-step/four-prompt engineering baseline does not establish meaningful visual transfer. Audit the released decision-training entry point and data, declare one coverage/update budget, and record every deviation needed for one GPU. Do not copy the generic math example or the LoRA pilot schedule. Evaluate fixed greedy endpoints and bounded sampled endpoints, with correctness, parsing and truncation diagnostics. Save only a small declared set of checkpoints with sufficient storage headroom.
3. Return a combined decision report before any main branch sweep or selector training. A failed H=4 precision criterion ends this bounded local study; it does not prohibit separately designed baseline reproduction or an aggregate study. No automatic H=8/16 continuation. Positive local correlation is not a prerequisite for an honest predictive-validity study, but affordable measurement precision is.

H=4 may increase visual effect size; it does not automatically improve alignment signal-to-noise, and the larger displacement increases first-order approximation error. Full rollout/branch/scoring/evaluation cost must count toward any future efficiency claim.

## Scope

Keep the pinned original composite text reward and binary visual correctness. Final-test data remain untouched. Early stopping, caption analyses and 7B stay deferred. This document is a prospective decision, not a new frozen training budget or evidence that ordinary training already transfers. No long baseline run is ready until its data coverage, endpoint evaluation and runtime cap are specified.
