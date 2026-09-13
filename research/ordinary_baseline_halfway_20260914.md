# Ordinary baseline: measured halfway costs

Step 16 and its scheduled checkpoint save completed successfully. The worker remained live and continued training. This is a halfway measurement, not terminal run success, checkpoint reload validation, or a visual-transfer result.

| Quantity | Observed |
|---|---:|
| Completed outer training steps | 16 |
| Sustained window, excluding first two steps | 14 steps |
| Mean wall-clock step, excluding checkpoint/validation | 239.118 s (3.985 min) |
| Observed step range in that window | 229.799–254.186 s |
| First full checkpoint save | 355.887 s (5.931 min) |
| Checkpoint files, total logical bytes | 41,272,696,651 (41.27 GB; 38.44 GiB) |
| Mean output length over sustained steps | 18.291 tokens |

The step timing includes inter-step orchestration and CPU/offload work. The checkpoint timing includes checkpoint materialization, synchronization and writes; dividing bytes by this duration is not a raw network-bandwidth benchmark. One completed save cannot characterize checkpoint-time variability. The first two startup steps remain part of any complete-run estimate.

The larger model file includes a tied output-weight alias. The prior validated canonical mapping identifies `lm_head.weight` as an alias of `model.embed_tokens.weight`. The prepared candidate exporter has been corrected before launch to validate canonical values and tied aliases, then export each canonical parameter once. Seven guard tests pass; the fresh GPU cost check remains pending. The baseline training/source/checkpoint implementation was not changed.

The current API-confirmed GPU rate is $1.59/hour on one A100 SXM4 80 GB; the network volume remains 250 GB. This is a current-rate input, not a historical invoice. Storage and idle time remain separate from measured job execution.

Evidence is backed up under `runs/ordinary-baseline-20260914/step-016-observation`, including metrics, events, launcher snapshot, `throughput-step16.json` and checkpoint file inventory. Full checkpoint reads/checksums will follow after baseline completion to avoid adding large competing I/O during the sustained run.

Remaining: complete baseline through step 32, final validation/cleanup and second checkpoint measurement; audit saved model/optimizer state; run the isolated fresh-candidate cost check; measure endpoint greedy speed; then deliver the full 128-candidate/nine-run/evaluation estimate. The full matrix is unlaunched.
