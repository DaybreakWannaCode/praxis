# Completed ordinary-baseline throughput measurement

The planned 32-step, one-pass original-Praxis baseline completed with exit 0. Both scheduled saves and final text validation completed, the supervisor reported no remaining owned processes, and a live GPU-process query was empty. The final logs and timing summary are backed up locally under `runs/ordinary-baseline-20260914/completed`.

The separate full checkpoint hash/load/finiteness audit is still running. This report establishes completed execution and timing, not yet that audit's success or any visual-transfer benefit.

| Component | Measured wall time |
|---|---:|
| Entire supervised baseline | 7,810.861 s = 2.170 hours |
| Training steps, excluding saves/validation | 6,884.437 s |
| Worker initialization | 177.011 s |
| Two full checkpoint saves combined | 711.552 s = 11.86 minutes |
| Final 32-prompt text validation | 5.932 s |
| Other startup, orchestration and cleanup | 31.923 s |

The no-op checkpoint-load phase adds 0.006 s; it is **not** a measurement of restoring a trained parent. Components are counted once: checkpoint/validation timers are removed from enclosing step timers before their phase totals are added. Fit time is not added again to its children. The first two training steps consumed 554.516 s and remain included in total cost.

After excluding the first two steps from the sustained-rate calculation only, mean wall-clock step time excluding saves/validation was 211.027 s, with an observed range of 168.627–254.186 s. Steps 3–16 averaged 239.084 s of within-step training work; steps 17–32 averaged 186.422 s. The speed change is observed, not established as a causal effect of checkpointing. Use the full measured trajectory for the primary projection and disclose the variation.

Sustained component means: rollout 11.775 s, old-policy scoring 27.269 s, reference scoring 55.494 s, and actor update 116.403 s. Generation-only tokens/sec would therefore substantially understate training cost. GPU-allocated wall time includes CPU work, parameter/optimizer transfers and storage waits; it is not GPU-kernel time.

Each checkpoint save took about 355.8 s (355.887 and 355.665 s), together accounting for 9.11% of the full job. The first checkpoint contained 41,272,696,651 logical bytes; final file inventories and checksums are being audited separately.

At the live-verified $1.59/hour compute rate, this one job costs approximately $3.45 in compute. This excludes storage charges, other experiments, audit work, setup/idle time and historical billing. The rate receipt is saved privately in `runs/ordinary-baseline-preparation-20260914/pod-cost-receipt.json`.

The current setup fits one A100 SXM4 80 GB: 3B full-parameter training, 32 prompts per batch, five responses per prompt, microbatch one and output cap 512. This does not establish fit or speed for larger batches/models or longer response distributions. Observed sustained mean response length was only 18.531 tokens. Every released text problem retains the choice-only suffix, despite the reasoning system instruction; extrapolation to a changed prompt contract requires a new bounded timing check.

Remaining before the complete matrix estimate: audit model/Adam/scheduler/RNG checkpoints and optimizer counters; measure a fresh candidate's real parent restore, update and canonical displacement export; check endpoint decoding speed; then combine those with the completed likelihood profile and explicit evaluation-inventory scenarios. Nine runs starting from a trained common parent need a real restore cost in addition to this from-pretrained baseline. No full matrix has launched.
