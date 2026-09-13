# Completed likelihood-surrogate development check

The four-candidate check completed with exit 0, all finite/norm gates, exact parent replay and final parent-restoration replay. All 12 completed artifact hashes match the private local backup. All candidate panel sizes and development means were independently reconstructed. This is a new fixed answer-label likelihood surrogate, not a repair or replacement of the inconclusive H1/H4 expected-correctness result.

## Scientific result

All values below are mean natural-log probability changes per image. They are not accuracy changes or percentage points. Score panel: 16 images; development panel: 32 independent-from-score images. No new reasoning responses or text updates were generated.

| Candidate | Gradient prediction on score panel | Direct score-panel change | Independent dev change | Exploratory scene-bootstrap dev 95% interval |
|---|---:|---:|---:|---:|
| 0 | -0.048077 | -0.058629 | -0.026117 | [-0.083859, 0.022578] |
| 1 | -0.051203 | -0.055173 | -0.001668 | [-0.026666, 0.019066] |
| 2 | -0.025869 | -0.028588 | +0.012364 | [-0.016619, 0.046234] |
| 3 | -0.038837 | -0.044789 | -0.020263 | [-0.054987, 0.011570] |

Descending orders: alignment 2 > 3 > 0 > 1; direct lookahead 2 > 3 > 1 > 0; development 2 > 1 > 3 > 0. Alignment agrees with development on 4/6 pair directions; direct lookahead on 5/6. Both rank candidate 2 first. The six comparisons are dependent, not six independent trials. Cosine order is 2 > 0 > 3 > 1. Do not apply a six-trial binomial test or claim validated selection from this.

Both score methods predict deterioration on every score-panel candidate, whereas candidate 2 improves development likelihood at the point estimate. Ranking and absolute transfer calibration are different questions. All exploratory per-candidate scene-bootstrap intervals include zero. These intervals describe sensitivity to scene resampling on an availability-selected panel; they do not establish benchmark-population coverage or uncertainty across training seeds.

The surrogate is numerically measurable under FP32, and its same-panel gradient predictions track the signs of directly measured changes. It is not exact for finite updates. Parent repeats are identical, but exact repeatability is not proof against all floating-point rounding or gradient bias. No on-model finite-difference sweep was run. The CPU directional finite-difference and sign tests passed. Normalization conditions on valid labels and ignores the total probability mass assigned to non-label responses; generated-answer evaluation remains necessary.

The four archived updates mix H100 and A100 training, use one common warm parent, and were already development artifacts. The measurement supports continuing the bounded ordinary baseline and evaluating actual answers; it does not authorize the full selection matrix or a claim that alignment improves visual reasoning. Direct lookahead remains at least as serious a practical candidate as alignment.

## Measured runtime and overhead

| Component | Measured time |
|---|---:|
| Entire measurement (including model setup, excluding outer launcher setup) | 1,876.977 s = 31.283 min |
| Parent forward panels plus replay | 18.185 s |
| Shared score-panel gradient | 492.113 s = 8.202 min |
| Candidate 0 load/check/dot/apply | 266.756 s |
| Candidate 1 load/check/dot/apply | 245.257 s |
| Candidate 2 load/check/dot/apply | 264.485 s |
| Candidate 3 load/check/dot/apply | 284.914 s |
| Direct score forward per candidate | 5.987–6.006 s |
| Development forward per candidate | 11.331–11.479 s |

The combined load/check/dot/apply phase is about 17.69 minutes, roughly 57% of runtime. Its current timer does not separate file I/O, decompression, checksums, dot products and parameter restoration; do not call the whole quantity pure checkpoint I/O. The residual covers setup, gradient-norm calculation, final restoration/replay and orchestration. Peak torch allocation was 33,469,194,240 bytes (about 31.17 GiB); observed nvidia-smi usage was higher because it includes memory outside torch live tensor allocation.

At the previously verified A100 rate of $1.59/hour, measurement GPU-allocated time corresponds to about $0.83 compute; this is a rate-based estimate, not an invoice, and excludes idle time, storage and engineering. The first launcher attempt exited before model loading because /usr/bin/time was missing; it is retained separately. No extra scientific attempt was discarded.

For these four candidates, direct score forwards cost about 24 seconds versus about 492 seconds for the shared gradient, before dot-product overhead. At 128 candidates, forwarding the same score panel projects to about 12.8 minutes, versus 8.2 minutes for one shared gradient plus unisolated dot-product work. This is only a scoring-component projection. Loading/reconstructing 128 archived updates using the current file-based path would project to about 9.44 hours; that is not a measured fresh candidate-pool construction time, and does not include GRPO generation, optimizer work, export or optimizer-state restoration. The full-matrix estimate must therefore wait for the ordinary baseline and candidate-construction measurements.

## Evidence and next action

Private archive: runs/choice-probe-dev-20260914-002/completed; analysis.json and remote-sha256.json alongside it. Public source: choice_probe.py, choice_measure.py, and research/scripts/analyze_choice_probe.py. Frozen new objective: choice_probe_development_lock.md.

Next: measure sustained original-Praxis training and checkpoint saves with the prepared 1,024-prompt baseline. Its data and prompt checks pass. Storage expansion is pending; no baseline or full matrix has launched. Complete greedy-evaluation timing and tentative-update construction accounting before returning a complete candidate-pool/nine-run/final-evaluation estimate. The active goal remains incomplete.
