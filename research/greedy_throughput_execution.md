## Completed; storage gate resolved

The greedy job completed with exit 0, 32 responses and exact replay. Four artifact hashes and all response-level quantities were audited. See greedy_throughput_result_20260914.md. After the user updated storage, the live API verified 250 GB. The notes below describe earlier progress.

# Greedy throughput execution — 2026-09-14

The likelihood check is complete (choice_probe_result_20260914.md). The active goal still requires ordinary training throughput, checkpoint costs, fresh candidate-construction accounting, greedy evaluation timing and a complete matrix estimate. No full matrix has launched.

This goal turn made concrete progress by implementing and launching a bounded greedy decoding profile using the existing 32 audited development scenes and saved warm parent. It does not replace the proposed broader baseline evaluation. One completion per image, greedy decoding, max 512 new tokens, max image pixels 200704, original reasoning prompt and explicit_final_v3 parser. The 33rd response is a deterministic first-image replay with a different RNG seed. No new training, score-set use or final-test access.

Source commit: 05bd31f. Launch: /workspace/praxis/research/scripts/run_greedy_throughput.sh in tmux session praxis-greedy-throughput. Output: /workspace/praxis/runs/greedy-throughput-20260914-001. Cap: one hour. PID 43420 was verified live at 41 seconds elapsed, loading checkpoint shards; not yet a completed timing result. The backend holds FP32 parameters and wraps generation in BF16 autocast; its initialization-only FlashAttention warnings are expected, not proof of failure. Judge actual execution by the live process, terminal exit and completed manifest.

Save every generated response and timing immediately on /workspace. Completion requires exit 0, 32 ordinary records, summary.json and exact greedy replay. Back up and checksum completed artifacts before reporting throughput. Separate cold model loading, decoding, replay and total elapsed. Do not treat label-likelihood forward speed as generated-answer speed or HF batch-one speed as vLLM speed.

Storage rechecked through RunPod API: still 150 GB. Approval to grow to 250 GB remains pending; ordinary training is not launched. Its 1,024-row data and prompt-length checks pass, and the isolated telemetry copy is ready. Do not delete prior evidence or silently remove checkpoints to bypass the storage gate.
