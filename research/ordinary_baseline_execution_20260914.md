# Ordinary baseline execution — 2026-09-14

Later observation: the same worker completed step 10 and remained live. The first eight post-startup steps have been backed up locally under `runs/ordinary-baseline-20260914/step-010-observation`. The analyzer correctly refuses to issue its combined throughput summary because no checkpoint-save event has completed yet. Earlier three-step observations below remain historical. Candidate preparation is complete with 32 distinct source indices verified disjoint from baseline train/validation and eligibility exclusions; its GPU run remains unlaunched.

Status: running at the last live observation, September 13 at 18:59 UTC. This is a throughput run, not evidence of visual transfer. The full candidate pool and nine-run matrix have not been launched.

## Persistent run and recovery

- Pod: `e8lcnuph2r1z3r`; existing A100 SXM4 80 GB.
- Network volume: `86u1nnngpl`, API-verified at 250 GB after the user's expansion. About 120 GiB was used before this run's checkpoints.
- Run: `/workspace/praxis/runs/ordinary-baseline-20260914`.
- tmux: `praxis-ordinary-baseline`; owned Ray worker PID 46347 was verified alive, not inferred from the launcher file.
- Supervisor: `research/scripts/launch_ordinary_baseline.py`; 12-hour cap and cleanup restricted to its unique inherited process tag. The cap does not stop pod billing.
- Config SHA-256: `86654cf5a2aeda87ae4c2a835ef00e08dcd7a96ea4366d22c03057ae84bd4d5d`.
- First two planned checkpoints: `checkpoints/global_step_16` and `checkpoints/global_step_32`. Neither was observed complete at the time of this note. Before step 16 there is no new saved training state to resume.

The temporary Codex usage-limit rejection prevented monitoring, not remote execution. After the user reset the limit, the same worker was observed training. No replacement run was started. A future interruption must be diagnosed from process state, exit status, and checkpoint files before any restart.

Private local snapshot: `runs/ordinary-baseline-20260914/observed-20260914-0304/` contains the current metrics, phase events, launcher record, config, and prompt-instruction audit. These are partial observations, not terminal completion evidence. Raw data, responses and checkpoints remain excluded from public Git.

## Observed timings, not the completed estimate

Worker initialization took 177.01 seconds. The first three training steps took 307.76, 246.76 and 229.77 seconds. Each reports 160 generated responses and one actor minibatch; checkpoint optimizer counters still need inspection to confirm actual optimizer applications. Sustained-rate acceptance requires at least eight post-startup steps and a completed checkpoint save. It has not yet been met.

Step 3 breakdown: generation 9.62 s, old-policy scoring 28.61 s, reference scoring 57.56 s, actor update 133.94 s. Thus generation speed alone would substantially understate training cost. These are GPU-allocated wall times, including CPU work and offload stalls, not GPU-kernel time.

## Prompt conflict discovered during live inspection

The rendered live training prompt contains the configured reasoning/tag system instruction. However, every retained released `problem` ends with `Now answer the question. Just output the choice:`. The persistent CPU audit counted this suffix in 1,024/1,024 training and 32/32 validation rows. This instruction conflicts with the request for tagged reasoning.

In steps 1–3, response lengths averaged 20.73, 18.27 and 18.30 tokens; format, tag-count and length rewards were all zero. The suffix is a plausible explanation, not a controlled causal finding. The original composite reward code is present and unchanged; zero observed components must not be described as having removed those reward terms.

Continue the fixed released-input run without editing live prompts or code. Report its throughput as conditional on this short-response regime. Do not silently use its rate for a corrected prompt that elicits long reasoning. Before selection training is frozen, decide and document the prompt contract; a changed contract needs a bounded timing check and matching candidate scoring. The current run is an adapted one-GPU baseline, not a complete reproduction of the paper's training recipe.

## Remaining deliverables

1. Completed sustained baseline timings, actual checkpoint bytes/save times, optimizer application evidence, and terminal cleanup status.
2. A bounded fresh candidate timing measurement: parent model/optimizer/scheduler/RNG restore, rollout and one update, parameter export/displacement work, and scoring. Old H4 replay timing cannot substitute for construction.
3. Final evaluation projections for the prospectively fixed inventory; keep the measured 256-scene and 1,000-scene scenarios conditional until inventory and endpoint response lengths are checked.
4. Complete estimate for 128 candidates, nine training runs and evaluations, separating engineering time, GPU-allocated execution, checkpoint overhead and idle time. No full matrix before this estimate.
