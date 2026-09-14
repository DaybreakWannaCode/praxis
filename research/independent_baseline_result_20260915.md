# Independent development baseline result

The adapted ordinary text-only Praxis baseline produced an unresolved change in greedy visual accuracy: 201/256 (78.52%) before training and 203/256 (79.30%) afterward. The net gain is 0.78 percentage points; the prospectively specified paired image-bootstrap 95% interval is [-1.95, +3.91] percentage points. This does not establish reliable visual improvement, equivalence, or harm. It is not evidence that alignment-based selection cannot help.

| Paired correctness transition | Images |
| --- | ---: |
| Incorrect to correct | 8 |
| Correct to incorrect | 6 |
| Correct at both endpoints | 195 |
| Incorrect at both endpoints | 47 |

All 256 images remain in the analysis. Parsing improved from 254 to 256 successful parses; neither endpoint truncated any correct-image response. Of the two initial parse failures, viva-681 became correct and viva-672 remained incorrect. This descriptive follow-up does not alter the primary endpoint or remove either image. Seven improvements and six regressions occurred among initially parsed answers.

On the same preselected 32 control targets, both endpoints scored 26/32 with the correct image and 21/32 with a shuffled image. Thus the correct-minus-shuffled gap was 15.625 points at both endpoints, with zero net change. This small control panel provides no observed increase in the gap; it does not localize a reasoning mechanism or establish invariance of individual answers. Do not compare the 256-image rate with the 32-image shuffled rate as if they covered the same targets.

## Evidence and execution

Plan SHA-256: `8f0db76d4e2c6ddc3f4d865bf70aae7c625791ceaa802b49d86048750f4d99bc`.
Both endpoints use the pinned pretrained model architecture, the same reviewed development images, prompt, parser, preprocessing and greedy decoding. Initial weights are the original pretrained revision; final weights are the audited step-32 actor. The 1,024-prompt, single-pass, one-seed training trajectory is an adapted baseline, not a full reproduction of the released Praxis settings.

Both endpoints exited 0 and passed exact terminal greedy replay. Initial runtime was 2,634.357 seconds (43.91 minutes); final runtime was 2,768.410 seconds (46.14 minutes), including initialization/integrity work. Each saved 288 primary response records; the two terminal replay checks bring generation count to 578. No new model or gradient files were written by these evaluations. Startup and metadata/storage work remain part of the actual cost.

The final archive contains 296 files, 2,150,400 bytes, SHA-256 `0891378a89ccf57f57c9d47b443a49c92ba3ae6c0a782dbe81ea641ee8f0727e`. Its local backup verified every file and repeated the entire paired analysis, exactly matching the remote result. The initial complete archive and its audit are also verified locally. Raw responses, labels, images and checkpoint files remain outside Git. The source and report are committed separately.

Result artifacts reside in `runs/independent-baseline-endpoints-20260915/`: `paired-analysis.json`, `initial-local-audit.json`, `final-local-audit.json`, complete endpoint archives, and each endpoint's responses and execution receipts. On the pod this directory is under `/workspace/praxis`.

## Interpretation and next decision

The interval is conditional on this development panel and this single trajectory. It is not between-seed uncertainty. Image groups are scene proxies, assistant visual review is not independent human annotation, and semantic scene independence remains uncertain. Panel membership was fixed after training but before endpoint outcomes. The reserved final-test inventory remains unused.

The storage intervention succeeded independently of the scientific result: two actual candidate lifecycles retained about 34.3 MB of durable evidence after releasing about 13.18 GB of temporary exports. This avoids multiplying displacement retention by the candidate count. Exact future optimizer replay and full-checkpoint publication integration are still unvalidated; old checkpoints have not been deleted. The purchased volume remains 250 GB, with about 214.5 GB accounted for at final launch.

Do not launch the proposed 128-candidate/three-arm study merely because infrastructure works. Do not expand H1/H4 horizons, sampling, or tune evaluation labels/prompts to improve this observed result. The current evidence supports (1) reliable measurement and storage machinery, (2) local likelihood-surrogate agreement in two calibration updates, and (3) an unresolved real-answer baseline gain. It does not yet support transfer prediction on independent outcomes or useful selection.

Before a new GPU experiment, reassess a concrete intervention budget and its identifiable claim. A flat random baseline does not logically preclude selection benefit, but candidate-scoring cost, fresh-rollout training and independent test outcomes must all be included. Preserve this baseline as the completed result; any altered training coverage, response cap, selector pool or prompt is a new declared experiment. The full selection matrix remains unlaunched. The completed jobs released the GPU; the pod itself has not been stopped.
