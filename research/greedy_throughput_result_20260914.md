# Completed greedy-evaluation throughput profile

The 32-image development profile completed with exit 0 and exact token replay under a different RNG seed. Four output-file hashes match the local backup. The audit reconstructed all image IDs, output token counts, response timings, parsed answers and correctness from the archived responses. This is timing evidence from one saved warm parent, not a before/after transfer result or a final-test evaluation.

| Measurement | Observed |
|---|---:|
| Model initialization and parent loading | 226.315 s (3.772 min) |
| All 32 ordinary greedy completions | 261.632 s (4.361 min) |
| Mean complete response time | 8.176 s |
| Response time range | 3.963–12.764 s |
| Total measured execution, including replay | 495.641 s (8.261 min) |
| Mean output tokens | 106.78 |
| Output-token range | 52–166 |
| Response tokens / second, including per-response overhead | 13.060 |
| Peak torch allocation | 21.287 GiB |
| Parsing / truncation | 32/32 parsed; 0 truncated |

Generation uses HF batch size one, FP32 stored parameters with BF16 autocast and FlashAttention2, max 512 new tokens and 200704 image pixels, the original reasoning prompt and explicit_final_v3 parser. Each response timer includes input preparation, image work, generation and parsing, with CUDA synchronization. It excludes the subsequent JSONL write; the full elapsed timer includes those writes. This is not a vLLM or training-throughput benchmark. Recorded correctness is 26/32 on these already-used development scenes; it must not be compared directly with old sampled scores or presented as a transfer gain.

## Conditional final-evaluation component estimates

If the same mean response length, panel difficulty, software, resolution and batch-one runtime persist:

- One parent plus nine trained models, each evaluated on 256 scenes: about **6.44 GPU-allocated hours**, including ten model loads.
- The same ten models on 1,000 scenes each: about **23.34 GPU-allocated hours**.
- One 256-scene model evaluation: about 34.88 minutes of response work plus 3.77 minutes loading.

These are linear planning projections, not complete-matrix estimates or uncertainty intervals. Final checkpoints may generate longer reasoning, especially with the original length reward. The current mean of 107 tokens is far below the 512-token cap. Re-evaluation on the ordinary baseline endpoint is required to assess that length sensitivity. Loading costs depend on file layout and storage cache state. Additional benchmarks, shuffled-image controls and engineering/idle time are not included in the figures above. The final test inventory is not frozen by this timing profile.

## Remaining goal work

The likelihood probe and this decoding profile are complete. Sustained ordinary-training throughput, full checkpoint save/load overhead and fresh tentative-candidate construction remain unmeasured. They are required before the total 128-candidate / nine-run / final-evaluation estimate can be completed. No full matrix has launched.

Storage was subsequently verified at 250 GB after the user's update. Total existing volume usage is about 120 GiB; the two roughly 39 GiB checkpoints fit with headroom. The prior storage block is resolved. The ordinary-baseline preflight and scoped supervisor are ready; live training status must be read from its own run artifacts, not inferred from this document.

Private evidence: runs/greedy-throughput-20260914-001/completed, remote-sha256.json and audit.json. Source: transfer_alignment/greedy_throughput.py. Bounded run: /workspace/praxis/runs/greedy-throughput-20260914-001.
