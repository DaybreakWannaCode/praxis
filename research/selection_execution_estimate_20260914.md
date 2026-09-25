# Selection-study execution estimate — measurements complete

The bounded timing work is complete. For the current file-based implementation, plan **about 105 GPU-allocated hours and $167 compute** for one static 128-batch pool, nine 32-step runs, audits and the 256-final-scene evaluation scenario. With a 25% planning allowance, budget about **131 hours / $210 compute**, plus storage and engineering. This is a conditional estimate, not approval to launch the full matrix; no matrix has launched.

## Measured execution projection

| Future component | GPU-allocated hours | Compute at $1.59/hour |
|---|---:|---:|
| 128 fresh candidate constructions | 62.63 | $99.59 |
| Shared likelihood probe + 128 candidate scores | 9.86 | $15.67 |
| Nine 32-step training jobs, including warm restores and saves | 20.91 | $33.25 |
| Nine two-checkpoint integrity audits | 2.12 | $3.38 |
| Final evaluation: 10 models × 256 scenes + replays | 7.54 | $11.98 |
| Baseline before/after: 576 dev/control answers + replays | 1.66 | $2.63 |
| Evaluation command startup overhead | 0.02 | $0.03 |
| **Total** | **104.74** | **$166.53** |

Using the older parent's decoding/load profile gives 103.45 hours / $164.48; using the newly measured endpoint gives 104.74 hours / $166.53. This two-profile span is not a confidence interval. A 25% planning contingency on the latter is **130.92 hours / $208.16 compute**. Storage fees, engineering, idle rental time and unmeasured offload transfers are additional.

With three separate candidate pools (384 constructions), the same 256-scene scenario rises to 249.71 hours / $397.04. With one shared pool and 1,000 final scenes, the scaling-only scenario is 121.94 hours / $193.88; it is not a claim that enough disjoint test scenes exist.

The nine warm training jobs total 20.91 hours (about 2.32 hours each). Their baseline full-checkpoint saves already contribute 1.78 hours; parent restores add 1.38 hours. Across the candidate pool alone, restores contribute 19.67 hours and the export routine 24.23 hours, both included in the construction row. This identifies where time goes without counting nested timers twice.

Already completed calibration work is not charged again as future training: the 32-step ordinary baseline took 2.170 hours, its integrity audit 0.236 hours, the fresh candidate 0.489 hours, the likelihood check 0.521 internal-measurement hours, the older greedy profile 0.138 internal-measurement hours, and the endpoint command 0.237 hours. These mixed-boundary measurements are not a reconstruction of the complete historical bill; earlier setup and idle gaps remain separate.

## Scope and accounting

- Pool: 128 distinct batches × 32 prompts, one actual candidate update per batch from the same warm parent. Construct once and share the static pool across three training seeds.
- Training: three seeds × Random / Alignment / Direct lookahead, nine runs with 1,024 selected prompts and 32 actual optimizer applications each. Include full measured job overhead, two checkpoint saves, and the real warm-parent restoration cost.
- Scoring: one shared 16-image visual likelihood gradient and parent preparation, then load/check/dot/apply and direct-score forward work for every candidate. Historical H4 scoring rates are an explicit proxy for the unimplemented fresh-H1 pool scorer.
- Final evaluation scenario: starting parent plus nine final models × 256 untouched scenes = 2,560 generated answers. Add ordinary-baseline before/after evaluation on 256 development scenes plus 32 shuffled-image controls at both endpoints = 576 answers. Twelve model loads and twelve single-image greedy replays are counted. Development and test panels are prospective inventory scenarios, not already downloaded/frozen splits.
- Checkpoint verification: extrapolate the completed two-checkpoint audit to each of the nine runs. This is outside training job time and inside GPU-allocated cost if performed on the rented pod.

GPU-allocated hours mean time occupying a rented pod, including CPU work, network I/O, offload and serialization. They are not GPU-kernel hours. The baseline timer includes saves/validation; the estimate does not add them twice. Candidate scoring is additional to candidate construction. Replaying archived updates was never substituted for the measured fresh-update construction cost.

The file-based scorer's parent/setup, gradient and final restoration are included once per pool. Per-candidate development outcomes are not included: the selection protocol ranks on the scoring split and evaluates final training models independently. The old four-candidate development result is already completed and is not charged as future work.

## Conditions that matter

This is a linear extrapolation, not a statistical runtime interval. Training responses were unusually short (roughly 17–19 tokens), with the released choice-only suffix retained. Changing the prompt/reward contract or producing long reasoning sequences invalidates the training-rate assumption. Different input lengths, I/O cache state, future model response lengths and failures may change costs. A 25% scheduling allowance is a planning contingency, not a measured confidence bound.

A separate pool per seed triples candidate construction and scoring, rather than just training; it is a substantially more expensive design. The 1,000-test-image scenario is only a scaling calculation: the current unique-image inventory may not support it after disjoint calibration/development splits. External benchmarks, dense learning-curve evaluations, caption controls and early-stopping experiments are not silently included.

## Storage is a launch requirement

The current 250 GB volume held about 210.49 GB before this candidate, whose compressed update adds 6.60 GB. A 128-candidate retained pool requires about 845 GB; nine pairs of full checkpoints require another 743 GB. Including existing evidence, literal full retention reaches about **1.80 TB before extra datasets, temporary files and headroom**. A nominal 2 TB volume is therefore a minimum planning size for this retention policy, with usage guards still required. No storage purchase or deletion has been made.

The alternative is a verified streaming archive/offload policy, retaining compact endpoint models and only the optimizer checkpoints needed for continuation. That policy needs implementation and an offload throughput measurement; its transfer time is not contained in this estimate. A candidate displacement is not a resumable checkpoint. Do not silently delete old evidence to make the run fit. The quoted execution time is conditional on sufficient storage and similar I/O performance; it is not an executable 250 GB retention plan.

## Engineering and calendar time

Allow **12–24 person-hours**, a planning judgment rather than measured work time: 6–10 hours to generalize the protected one-candidate runner and fresh-parent scorer with recovery checks; 3–6 hours for image acquisition and scene/source overlap audit; 2–4 hours for retention/restart handling; and 1–4 hours for sealed selection, evaluation manifests and reporting. These activities can partly overlap and should largely occur without an idle rented GPU. They do not prove the full matrix is ready to run.

Serial compute is about 4.3 continuous days at the central estimate. Add scheduling contingency and engineering availability; parallel pods reduce calendar time only if available and appropriately provisioned, and do not erase total allocated-hour cost. Do not interpret an hourly rental cap as a total-budget authorization.

The baseline ended at timestamp 1789332776.3387136 and the candidate began at 1789349077.4425857: a gap of 16,301.104 s (4.528 h). Only 849.938 s is attributed to its checkpoint audit; the remaining 15,451.166 s is unclassified elapsed time, not measured hands-on engineering. If continuously billed, the entire gap costs approximately $7.20 at $1.59/hour. Historical billing was not audited. This gap is excluded from productive throughput and preserved separately rather than hidden in training speed.

## Recommendation

Keep the current A100 for the bounded measurements. Parent restore plus validation/export consumed 70.1% of the fresh candidate's time; changing GPU models alone has no measured remedy for that overhead. Before a full pool, implement and benchmark bounded parent reuse / scoring-before-export, or consciously accept the measured cold-process cost and larger storage requirement. Treat any savings as unverified until measured. No numerical speedup is credited here.

Finish the broader ordinary before/after visual development check and freeze the inventory/retention policy before deciding on a full selection study. The completed likelihood check establishes that the surrogate can be measured, not that it predicts generated-answer transfer; the older H1/H4 study remains inconclusive. Do not replace a missing transfer effect with a runtime or likelihood claim.

Reproducible calculation: `research/scripts/estimate_selection_execution.py`; private measured inputs and checksums are recorded in `runs/selection-execution-estimate-20260914/final.json`. Public reports contain aggregate measurements, not raw examples or model weights.
