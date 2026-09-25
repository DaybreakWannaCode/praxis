# Completed H1/H4 production precision experiment

**Decision: this bounded budget is insufficient to resolve candidate transfer comparisons.** Both H1 and its single predeclared H4 fallback resolved 0/6 pairs; 4/6 were required. Do not choose either endpoint for a main branch sweep from this evidence. This is a completed measurement with inconclusive scientific precision, not a demonstration that alignment is ineffective.

## Execution and integrity

H4 completed with exit 0, all 1,800 archived responses and exact token replay for the eight retained parent responses. Four candidates each contain four consecutive original-Praxis updates from the same warm parent, with actual total displacements measured in the verified canonical coordinates. Original composite text reward is unchanged; the visual objective is binary correctness under the pinned explicit_final_v3 parser.

The complete archive is backed up privately under runs/production-gate-20260907/production-h4-precision-001/completed. All 16 copied remote result/terminal files have matching local SHA-256 hashes. The score and projection files match local copies archived at the start of independent outcome evaluation, after scoring was sealed. Every raw response reproduces its saved reward, parse status and extracted choice. The archive audit reconstructs outcomes, paired replay, pointwise and simultaneous uncertainty, and the frozen decision. Source, image and parity evidence is summarized in production_completion_audit.md.

## Candidate outcomes and parsing

Each ordinary policy uses 32 fixed development images with eight sampled responses per image. These are sampled correctness rates, not greedy benchmark accuracies. Conditional correctness is diagnostic; unparsed responses count as zero in the primary objective.

| Policy | Correct / 256 | Correctness (%) | Parse success (%) | Correct given parsed (%) | Truncated |
|---|---:|---:|---:|---:|---:|
| parent | 163 | 63.67 | 89.06 | 71.49 | 0 |
| null_independent | 172 | 67.19 | 92.97 | 72.27 | 0 |
| child_0 | 168 | 65.62 | 90.62 | 72.41 | 0 |
| child_1 | 159 | 62.11 | 88.28 | 70.35 | 0 |
| child_2 | 166 | 64.84 | 89.45 | 72.49 | 0 |
| child_3 | 173 | 67.58 | 88.67 | 76.21 | 1 |

## Alignment and independent outcome contrasts

All contrasts use zero-based candidate indices and first minus second. Score units are multiplied by 100 to match the first-order predicted percentage-point scale; they are not observed accuracy changes. Score intervals are approximate 95% Student-t intervals from four independent response repeats conditional on 16 score images. Outcome intervals use the conservative paired sign-category Chernoff-KL method conditional on the separate 32-image development panel.

| Pair | Alignment difference ×100 | Score 95% interval ×100 | Outcome difference (pp) | Outcome 95% interval (pp) |
|---|---:|---:|---:|---:|
| 0−1 | 0.84 | [-2.33, 4.02] | 3.52 | [-6.54, 13.42] |
| 0−2 | 3.38 | [-2.90, 9.66] | 0.78 | [-10.93, 12.46] |
| 0−3 | 2.63 | [-0.71, 5.97] | -1.95 | [-13.14, 9.32] |
| 1−2 | 2.54 | [-0.98, 6.07] | -2.73 | [-13.31, 7.97] |
| 1−3 | 1.79 | [0.39, 3.18] | -5.47 | [-15.56, 4.87] |
| 2−3 | -0.76 | [-4.05, 2.54] | -2.73 | [-14.43, 9.08] |

Only score pair 1−3 resolves a direction pointwise; none of the six outcome contrasts resolves a direction or lies entirely within the ±2 pp practical-tie band. Consequently no pair is jointly resolved. The pointwise score ranking for 1−3 opposes its outcome point estimate, but outcome uncertainty prevents a reliable directional conclusion.

Bonferroni intervals allocate 0.05/6 to each pair within each reported family. They do not jointly cover both families at 95%. The predeclared feasibility rule uses pointwise intervals and is exploratory.

| Pair | Simultaneous score interval ×100 | Simultaneous outcome interval (pp) |
|---|---:|---:|
| 0−1 | [-5.37, 7.06] | [-8.44, 15.26] |
| 0−2 | [-8.91, 15.68] | [-13.11, 14.62] |
| 0−3 | [-3.91, 9.16] | [-15.22, 11.43] |
| 1−2 | [-4.36, 9.45] | [-15.28, 9.98] |
| 1−3 | [-0.95, 4.52] | [-17.43, 6.82] |
| 2−3 | [-7.21, 5.69] | [-16.59, 11.29] |

## Changes relative to the parent

| Policy minus parent | Difference (pp) | Pointwise 95% interval (pp) |
|---|---:|---:|
| null_independent | 3.52 | [-10.48, 17.35] |
| child_0 | 1.95 | [-8.75, 12.57] |
| child_1 | -1.56 | [-12.49, 9.43] |
| child_2 | 1.17 | [-9.72, 12.01] |
| child_3 | 3.91 | [-7.46, 15.10] |

All child-versus-parent intervals include zero, so none establishes beneficial or harmful transfer. The unchanged independent-null estimate also fluctuates; its uncertainty is explicitly retained rather than treated as an improvement.

## Score magnitude diagnostics

| Candidate | Mean alignment ×100 | Actual update norm | Mean cosine across repeats |
|---|---:|---:|---:|
| 0 | -1.15 | 0.116036678 | -0.000366367 |
| 1 | -1.99 | 0.096353805 | -0.000960254 |
| 2 | -4.53 | 0.105477509 | -0.002093 |
| 3 | -3.78 | 0.091783999 | -0.00196859 |

Cosine is a diagnostic, not the primary transfer predictor. Four repeated gradient estimates and all per-image projections remain in the archive. Aggregate-gradient and mean per-image projection paths differ by at most 3.26e-11 due to floating-point accumulation; this does not materially affect the reported intervals.

## Runtime, memory and retained attempt costs

H4 measurement: 28287.040 seconds (7.858 hours). Launcher process elapsed time: 28303.237 seconds (7.862 hours), below the 10-hour cap. Peak torch allocation: 35.314 GiB. Linux maximum child RSS: 166.830 GiB; this is not summed process-tree memory. The A100 was idle with 0 MiB allocated after the evaluation exited.

H1 measurement took 6.876 hours, with 35.306 GiB peak torch allocation and 119.696 GiB maximum child RSS. Its outcome intervals also left 0/6 pairs jointly resolved; the completed H4 fallback supersedes the old pending-execution note.

The following compute estimates apply observed durations to the recorded GPU rate; they are not billing invoices. Storage, provisioning, idle time, runtime restoration and read-only verification overhead are excluded.

| H4 stage | Seconds | GPU | Rate ($/h) | Estimated compute ($) |
|---|---:|---|---:|---:|
| Candidate 0 failed export attempt | 2194.98 | H100 PCIe | 2.89 | 1.76 |
| Candidate 0 exact recovery | 2719.25 | H100 PCIe | 2.89 | 2.18 |
| Candidate 1 | 2344.50 | H100 PCIe | 2.89 | 1.88 |
| Candidate 2 | 2763.33 | A100 SXM | 1.59 | 1.22 |
| Candidate 3 | 3149.97 | A100 SXM | 1.59 | 1.39 |
| Visual evaluation including process overhead | 28303.24 | A100 SXM | 1.59 | 12.50 |

Recorded-stage compute subtotal: approximately $20.94. A candidate-3 read-only verification attempt additionally hit its 1,800-second cap before a longer read-only retry passed; no extra candidate training was performed for that verification retry. Its wall-clock and other verification time are outside the subtotal, so the subtotal is not the full project cost.

## Recommendation and limits

Close this bounded precision experiment with an insufficient-budget finding. Neither H1 nor H4 supplies sufficiently precise candidate outcome comparisons for the planned prediction/selection study. Do not retry favorable seeds, increase the horizon, expand evaluation adaptively, or launch main sweeps under this protocol. A subsequent redesigned study needs a new prospectively fixed measurement/power plan.

The evidence applies to one warm parent and an availability-limited 48-image development panel, not benchmark-wide or population transfer. Four-repeat score intervals are approximate. H4 has greater finite-step Taylor error. Candidate 0/1 training used H100 PCIe and candidate 2/3 used A100 SXM; all H4 visual scoring/outcomes used the A100. Hardware is confounded with candidate identity, and H1 versus H4 is not a clean horizon-only comparison. No final-test data was used.
