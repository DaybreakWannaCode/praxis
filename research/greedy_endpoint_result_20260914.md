# Ordinary baseline endpoint: greedy timing complete

The step-32 endpoint passed the timing check with exit 0, 32 archived development answers and exact greedy token replay on the first image. A live process/GPU check found the endpoint process gone and no GPU processes. All four measurement files match the remote SHA-256 receipt; an independent local audit reconstructed their counts, token totals, correctness/parsing/truncation totals and decode time.

The audited endpoint model SHA-256 is `f9f1382119ebd080bddd52abe3e23d7233751a6710433bc4a64d0a2a16ac3fac`. The file was checked again before loading; its hash matched the frozen full-checkpoint audit.

| Component | Measured seconds |
|---|---:|
| Shell-timed command, including Python startup | 854.449 |
| Inside measurement, including one replay | 847.936 |
| Startup, including checkpoint checksum | 573.962 |
| Checkpoint checksum alone (part of startup) | 489.290 |
| Remaining startup after checksum | 84.672 |
| Decode 32 answers | 266.330 |
| Mean seconds per answer | 8.323 |

Command overhead outside the internal measurement was 6.513 s. The shell timer excludes the preceding lightweight launcher preflight. The execution estimator includes this measured command overhead once per evaluation invocation, and approximates one replay by one mean answer time. The checksum is not counted twice when included in startup.

Generated tokens totaled 3554, averaging 111.062 per answer (range 67–167); no answers were truncated. All 32 parsed, and 26 were correct. These are timing-panel observations, not a full baseline before/after result. The older warm parent is not the ordinary baseline's pretrained starting checkpoint; equal 26/32 totals cannot establish unchanged performance or a transfer effect.

The older warm-parent timing was 8.176 s/answer and 106.781 tokens/answer; this endpoint was 8.323 s/answer and 111.063 tokens/answer. Decoding was similar across these two observed models, but loading/checksum overhead differed considerably. Cold network reads and cache state are not controlled throughput experiments, so report both profiles as scenarios, not confidence bounds.

This uses batch-one Hugging Face generation, FP32 stored parameters, BF16 autocast and FlashAttention2, greedy decoding, a 512-token cap and the existing 32-image development panel. It is not vLLM throughput or a final-test evaluation. The separate 16-image calibration panel was not used for generated endpoint outcomes.

Private backup: `runs/greedy-endpoint-20260914-001/completed`, including responses, manifests, summary, remote hashes, local audit and command timing. Persistent original: `/workspace/praxis/runs/greedy-endpoint-20260914-001`.
