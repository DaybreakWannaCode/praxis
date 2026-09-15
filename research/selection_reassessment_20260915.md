# Selection decision after the independent baseline

The completed ordinary baseline is inconclusive, not a failed prerequisite proving that selection cannot work. Preserve its fixed analysis. The paper still needs independent prediction and a useful intervention; two same-probe likelihood checks do not meet either requirement.

## Concrete cost comparison: intervention only

Recomputed from both completed candidate lifecycles and the completed 288-answer endpoints using `scripts/reassess_selection_budget.py`. All scenarios retain 32 selected batches, 1,024 prompts, 32 training updates per arm and three paired seeds. Each pool is constructed once and shared across seeds. Candidates use a common warm parent, as do the subsequent fresh-rollout training arms. The old pretrained-to-step32 baseline is not the warm-parent random control and cannot replace it.

| Candidates | Training arms | Measured-profile projection, allocated hours | Compute at historical $1.59/hour |
| --- | --- | ---: | ---: |
| 64 | Alignment, Random | 58.59–64.23 | $93.15–102.13 |
| 64 | Alignment, Random, Direct lookahead | 68.57–74.22 | $109.03–118.01 |
| 128 | Alignment, Random | 96.43–107.73 | $153.33–171.29 |
| 128 | Alignment, Random, Direct lookahead | 106.42–117.71 | $169.21–187.16 |

These ranges are two timing profiles, not confidence bounds. A 25% planning allowance puts the 64-candidate/two-arm option at 73.24–80.29 hours before unmeasured work. Candidate construction includes export; it is not charged twice. Cold visual scoring, including recalculating the probe gradient for each candidate, is charged as actually measured. No hypothetical shared-parent speedup is credited. The completed baseline is sunk work and is not charged again. Model evaluation uses the entire measured 288-answer job as a proxy; this does not freeze the future test panel.

The table excludes unmeasured backup/cleanup orchestration, engineering, idle rental, archival transfers and storage fees. Training/decoding length changes invalidate these rates. Reproducible private output is `runs/selection-reassessment-20260915/estimate.json`, with input hashes. A per-hour cap is not approval for this total spend.

## Recommended scope and claim

Prefer 64 candidates over 128 if proceeding: this halves candidate generation while retaining the full training exposure. A 32-candidate pool selecting 32 batches would have no selection contrast. Do not halve training updates merely to make a cheaper result more likely to fit the budget.

The minimum intervention is Alignment versus Random, three paired seeds, with fresh training rollouts after scores and selection are sealed. Direct lookahead can remain a score diagnostic unless its additional training cost is justified. Cosine and update norm remain mandatory diagnostic comparators. A two-arm experiment cannot claim inner-product selection outperforms cosine selection or other selection algorithms. Equal-update improvement is not compute efficiency: charge selection overhead and report the lack of an equal-total-compute random control if one is not run.

Prediction must be evaluated on a disjoint visual panel with predictor values sealed before outcomes. Same-calibration Taylor agreement is numerical validation only. Likelihood-gradient alignment predicts a likelihood surrogate; generated-answer gains remain the independent practical endpoint. A static selector trained/evaluated at one warm parent supports a limited local/data-selection claim, not universal transferability across models or training stages. Shared candidate scores across three seeds do not supply three independent selector replications.

The two completed lifecycle candidates are engineering examples, not a randomly sampled scientific candidate pool. Freeze a new eligible pool without selecting it based on their scores. Any candidate reuse must be declared and justified before outcomes, not silently counted as fresh evidence.

## Statistical resolution

For a paired image difference D in {-1,0,1}, let q=P(D!=0) and d=E[D]. Under independent images, Var(mean D)=(q-d^2)/N. This is an algebraic sensitivity calculation, not a guarantee of power, normality or scene independence.

At N=256, near zero effect, discordance of 5%, 10% or 20% gives approximate 95% half-widths of 2.74, 3.87 or 5.48 percentage points. At N=512 they are 1.94, 2.74 or 3.87 points. Thus a practically worthwhile two-point effect may remain unresolved even after doubling the panel. Do not treat three seeds as 3N independent images. Report per-seed effects, between-seed variability and paired image uncertainty conditional on the runs.

There are 557 reserved annotation IDs, not 557 verified independent images. A 512-image test is only an inventory scenario. If acquiring this set, freeze eligibility, overlap rules and all retained image identities without model outcomes before final evaluation. Do not repeatedly expand the test until significance appears. Do not convert the observed baseline discordance into a confident forecast for selected-versus-random models.

## Storage and execution gates

Two real candidate lifecycles have demonstrated small durable evidence plus bounded temporary export deletion. This fixes candidate accumulation, not the entire checkpoint footprint. The current 250 GB volume already held approximately 214.5 GB at final baseline launch. A new 41.27 GB recovery checkpoint plus reserve does not fit alongside that history.

Next implementation work is safe checkpoint publication in the original training path and verification of a real archival destination. The tested standalone publication helper is not yet proof of original-trainer integration. Preserve historical checkpoint files until an independently verified destination copy exists; no deletion or volume enlargement is authorized by this document. Full-fidelity optimizer state cannot be replaced by a displacement or a score receipt.

Decision: hold the full sweep. Advance the checkpoint/archive integration and freeze a concrete prediction-plus-intervention protocol, test inventory and total budget before another scientific GPU launch. The reduced cost table is a decision artifact, not a launch instruction. No extra H1/H4 sampling, early-stopping experiment, prompt tuning or final-test inspection follows from the inconclusive baseline.

## Correction: independent candidate outcomes were not included

The table above includes calibration scoring and final trained-model outcomes. It does **not** include independent generated-answer evaluation of every tentative candidate for the paper's predictive-validity experiment. Consequently the proposed $150 additional compute cap was an intervention-only allowance, not a complete prediction-plus-intervention budget. No spend or storage expansion was performed in response to that proposal.

The revised calculator now reports these scopes separately. Using the measured 288-answer endpoint job as a proxy for each of 64 child models plus one common parent on a distinct development panel adds 49.986 allocated hours. The two measured candidate profiles then project 108.57–114.22 hours total, or $172.63–181.61 at the historical $1.59/hour rate. A 25% allowance gives $215.79–227.01. Storage, engineering, idle time and unmeasured additional child-reconstruction/verification overhead remain excluded. These are planning projections, not confidence bounds or authorization. The endpoint panel size and decoding policy are still to be frozen; the 288-answer job is a cost proxy, not a required study design.

This correction does not justify increasing sampling or repeating the closed H1/H4 study. Nor can same-calibration likelihood agreement be relabeled independent visual prediction to fit the smaller budget. A disjoint likelihood outcome is a legitimate *surrogate* prediction endpoint if declared as such, with generated-answer improvement tested by the intervention; it supports a narrower estimand than predicting expected visual correctness. Freeze that scientific distinction before committing the full compute allocation.

Private reproducible output: `runs/selection-reassessment-20260915/with-prediction-estimate.json`. The earlier estimate is retained unchanged for provenance. The full study remains unlaunched; the previous pending spending question must not be interpreted as approval for this larger scenario.
