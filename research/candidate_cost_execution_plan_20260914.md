# One fresh candidate: cost measurement plan

The single candidate cost run has been launched after the baseline completed, released its GPU processes, and passed the checkpoint audit. This is one candidate, not the 128-batch pool or a selection experiment. Subsequent launch requirements below document the guards that passed.

Live launch record: tmux `praxis-candidate-cost`, worker PID 55088 observed initializing, supervisor owner `a171a2de236d4e789fd2497864fb7ba5`, two-hour cap. Output `/workspace/praxis/runs/candidate-cost-20260914-001`; launch timestamp 1789349077.4425857. Preflight found 210,493,066,752 bytes used and reserved a conservative 37,895,799,740 bytes within the verified 250 GB quota. The run subsequently completed with exit 0; see candidate_cost_result_20260914.md for the measured component ledger and export checks.

The baseline finished at timestamp 1789332776.3387136. The gap before this candidate launch is 16,301.104 s (4.528 hours); only 849.938 s is currently attributed to the checkpoint audit. The remaining gap is unclassified elapsed time, not measured training or measured hands-on engineering. If the pod was billed continuously, that gap also incurs compute rental charges; historical billing has not been audited. Keep it visible in the eventual execution ledger rather than folding it into per-step throughput.

Prepared on the pod: `/workspace/praxis/data/candidate-cost-20260914` and isolated source `/workspace/praxis-candidate-throughput`. Config SHA-256: `149615d99b8c1757d83339d4a3489b513ff0a43319f2a37225b71fe3371e0d8c`. Intended run directory: `/workspace/praxis/runs/candidate-cost-20260914-001`. Launcher: `research/scripts/run_candidate_cost.sh`, which refuses an incomplete baseline, occupied GPU, changed data/config/source or insufficient quota headroom. Preparation did not require or launch a model.

Use the ordinary baseline's step-16 model and Adam/scheduler/worker-RNG state as the common warm parent. Keep the original pretrained reference policy. Select a new fixed 32-prompt batch from the same released source, excluding baseline training/validation indices and all exclusions from its eligibility audit. Preserve the released prompt text, including its choice-only suffix, so the timing regime matches this baseline. The broader scene audit remains outstanding.

Run the original trainer in a separate telemetry copy with one rollout batch, five responses per prompt, one PPO epoch and one actor minibatch. Restore the parent's worker checkpoint but deliberately retain the new candidate dataloader; resetting the driver's progress counter to zero labels this single cost step. This is not a claim of complete rollout-engine replay. The parent restore has a separate timer.

Original rollout, reward, actor update and optimizer source are unchanged. The opt-in cost module changes only parent-loading orchestration and terminal serialization. At the end, check that populated Adam counters increment by at most one, with at least one actual increment, then export the lossless before/after model-state displacement. Measure model-state materialization, parent file opening and lossless export separately. Memory-mapped parent reads occur during export and belong to that time. The result is **not a resumable checkpoint**; it contains the displacement and audit records rather than another full Adam copy.

The export uses the already-tested per-tensor FP32/FP64 lossless format and requires exact endpoint reconstruction and a nonzero norm. Its public aggregate cost report contains no raw examples. The original final text-validation call remains timed separately and must not be counted as update work. The supervisor must clean up only its own tagged workers.

Before launch, verify:

- Successful baseline exit, complete step-16 and step-32 checkpoint artifacts, and no active baseline GPU workers.
- Source/data/config hashes, explicit one-candidate opt-in, exactly 32 disjoint candidate prompts, one loader batch and nonzero parent learning rate.
- Network quota and current usage. Reserve the conservative uncompressed FP64 displacement bound plus temporary headroom; do not infer user quota from the backing filesystem's free space. Do not delete old evidence.
- Persistent tmux supervisor, bounded runtime, a new private output directory, and explicit extension/source paths.

Expected evidence: `parent-restore.json`, standard `metrics.jsonl` and phase events, `exports/global_step_1/actor/cost.json`, lossless delta manifest/chunks, and terminal launcher/exit records. Validate them before extrapolating. Checkpoint export here means displacement serialization; full baseline checkpoint-save costs are a separate measurement.

Cost projection will distinguish a conservative cold-process implementation (startup and restore per candidate) from a hypothetical persistent-worker implementation. Do not claim the latter's reduced startup cost is measured. Shared visual gradient and direct-score forward timings come from the completed likelihood profile, with parent/panel differences disclosed. Its old file-replay timing can inform scoring cost, but cannot replace this new construction measurement. A complete candidate cost must include restore, rollout, reference/old-policy scoring, update, serialization, score loading/dot product and visual forward work exactly once.
