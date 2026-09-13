## Current update

Greedy timing is complete: see greedy_throughput_result_20260914.md. Storage is now verified at 250 GB, with about 120 GiB used. The storage expansion gate is resolved; ordinary training and fresh candidate construction are still required for the complete estimate. Earlier pending-storage notes below are historical.

# Ordinary Praxis throughput measurement — 2026-09-14

Goal: finish the archived-checkpoint likelihood check, measure sustained end-to-end throughput in the planned ordinary baseline, and estimate the complete candidate pool, nine training runs and final evaluations. Do not launch the full matrix before the estimate exists.

The previous goal turn made progress: it implemented and launched the likelihood check. This turn verified the same live PID and continued it without restarting. The likelihood run then completed with all four candidates, final restoration replay and a verified 12-file backup. Its measured result is in choice_probe_result_20260914.md; ordinary training and complete matrix costing remain pending.

## Measured inputs required

- Original one-stage full-parameter baseline, 1,024 distinct eligible textual situations, one pass, rollout batch 32, five responses per prompt, actor global batch 32, microbatch 1, output cap 512. Worker code multiplies global batch by rollout count, so the effective actor minibatch is 160 responses. One actor optimizer application is expected per rollout iteration when ppo_epochs=1; verify live metrics/worker logs, not just the 32-step trainer counter.
- Report startup through worker initialization separately. Retain the first two steps in total cost, exclude them only from sustained-rate estimation. Require at least eight subsequent steps and one completed full checkpoint save. Prefer the whole 32-step baseline; if interrupted, label estimates from the observed sustained window as provisional and preserve the trained checkpoint.
- Capture rollout, old-policy/reference scoring, reward, update, validation, checkpoint and whole-step timers as JSONL. Record wall times of logging calls to include inter-step orchestration. Include fit and initialization wall times, final validation and final checkpoint events even outside the trainer loop.
- Capture actual response lengths, truncation and full step count. Peak generation tokens/sec alone is not end-to-end throughput. Do not use the earlier control/observer parity runs as ordinary training speed: those runs included duplicated updates, snapshots and export work.
- Measure checkpoint bytes and elapsed saves on the network volume. Count temporary save headroom and parent restore/load time for candidate work. GPU-allocated wall time includes CPU computation and storage stalls; distinguish it from GPU kernel time. Storage charges and idle pod time are separate.
- Measure greedy evaluation responses/sec from the declared development evaluation, with image resolution, prompt/output lengths and parsing/truncation recorded. A likelihood forward pass cannot stand in for greedy generation speed.

## Data preparation evidence

The source file contains 10,000 rows. CPU preparation found 9,631 eligible distinct normalized situations after template, label, duplicate and exact visual-overlap checks, selected 1,024 for training and 32 disjoint text-validation situations by a fixed seeded hash, and wrote source/output hashes. This does not establish independence against paraphrases or unknown shared source scenes. Only problem/answer are fed to training; embedded source conversations or rationales are not used. All 1,024 training prompts passed the actual processor length check: 254–412 tokens, mean 321.66, zero over the 2,048-token cap. All 32 text-validation prompts also passed. The config hash is recorded in prompt-preflight.json.

The broader 256-scene visual development audit remains outstanding. Preserve the existing score/dev split and final test exclusion. No visual outcome tuning is permitted during throughput planning.

## Matrix estimate to produce from those measurements

Primary planned matrix: one static pool of 128 candidate batches x 32 prompts (4,096 prompts), scored once from one common parent; three training seeds for each of Random / Alignment / Direct lookahead = nine training runs on selected 1,024-prompt subsets. Candidate construction rollouts are discarded; training uses fresh rollouts. Report a separate sensitivity estimate for rebuilding the 128-candidate pool per seed (384 candidate constructions), rather than silently charging or assuming it.

For each estimate, show GPU-hours and elapsed wall-clock hours on one GPU; parallel scaling must state the additional GPUs and nonparallel I/O. Break out:

1. One-time setup, ordinary baseline, common-parent creation and cold model load.
2. 128 actual tentative updates, parent restores, displacement extraction/dot products, shared visual gradient, and direct score-set forward evaluations. Existing H4 reconstruction measures scoring/load costs but cannot alone establish fresh H1 candidate construction cost. Measure/instrument that stage after the baseline before treating the pool estimate as verified.
3. Nine independent training runs: startup + observed per-step end-to-end work x 32 + planned checkpoint saves + validation + shutdown. Count observed optimizer applications, not just outer-loop steps.
4. Final greedy evaluations: one shared parent plus nine children, multiplied by the prospectively fixed final test inventory. Show per-256-scenes and per-1,000-scenes cost until the final inventory is frozen; do not label either as the complete benchmark cost. Add ordinary pre/post development evaluation and the 32-scene shuffled-image control separately. External benchmarks, if retained, need their own counts and measured/prompt-length-adjusted rates.
5. Disk peak/retention, export and backup time, pod idle/setup time, failed attempts and reruns. Report numerical ranges using observed sustained-window variability; no fabricated confidence intervals from one run.

Engineering time: keep a separate ledger of code changes, dependency repair, data audit, source verification and manual analysis. It may overlap GPU execution and must not be added to GPU-hours. Historical untracked hands-on time is unknown, not zero. Forecast remaining engineering as a labeled planning allowance, not measured training throughput.

## Current storage gate

API verifies a 150 GB standard network volume. du reports about 104 GiB for the project and 7.1 GiB for model files; other volume contents may add more. One existing full parent checkpoint occupies about 39 GiB. Two baseline checkpoints require about 78 GiB additional retention. A request to expand to 250 GB is pending; no baseline training starts under an insufficient disk budget and no old evidence is deleted to make it fit. Published standard storage is $0.07/GB/month, so 100 GB extra is about $7/month (https://docs.runpod.io/pods/storage/types, checked September 14).
