# Fresh one-step candidate: completed cost measurement

The single candidate completed with exit 0 and no remaining supervisor-owned processes. It restored the ordinary baseline's step-16 parent, generated 160 fresh text responses from 32 fixed prompts, applied one original-Praxis actor update, and exported a lossless displacement. This is a cost measurement, not a visual-selection or transfer result.

| Component | Seconds | Minutes |
|---|---:|---:|
| Worker initialization | 203.114 | 3.39 |
| Actual warm-parent restoration | 553.142 | 9.22 |
| Training step excluding save | 282.023 | 4.70 |
| Save wrapper including displacement export | 684.274 | 11.40 |
| Final text validation | 4.905 | 0.08 |
| Other startup, orchestration and cleanup | 34.097 | 0.57 |
| **Entire supervised candidate** | **1,761.556** | **29.36** |

At the frozen $1.59/hour compute rate, the job consumed approximately $0.78 of compute rental. GPU-allocated wall time includes CPU execution, offload and network storage waits. The enclosing fit/step timers are not added to their children twice.

The actual training step included generation 8.057 s, old-policy scoring 26.171 s, reference scoring 119.057 s, and actor update 128.668 s. Mean text response length was 16.744 tokens, under the unchanged released choice-only suffix and the 512-token cap. This one step does not supply a variance estimate for the candidate pool. Its reference scoring was slower than the sustained ordinary baseline, so generation-only speed is particularly misleading.

Inside export: materialization 0.518 s, parent mmap opening 0.119 s, canonical/alias validation 19.583 s, and lossless serialization 661.195 s. The export routine's total was 681.486 s; the save wrapper above includes an additional 2.788 s. Parent mmap page reads are charged where they occur, rather than treating the cheap opening call as total read time.

All 824 canonical parameter tensors were covered, totaling 3,754,622,976 unique scalar coordinates. The tied output/input embedding alias was validated and not exported twice. All 824 deltas fit FP32 with exact child reconstruction. The exporter checked finite parent/child/delta values chunk by chunk and recorded raw checksums. Update norm was 0.017204253378707783. All 37 populated optimizer-state entries advanced exactly one step. These entries reflect the optimizer's state organization, not 37 canonical model tensors.

The compressed displacement totals 6,601,372,705 bytes (6.60 GB decimal), excluding small metadata. It is **not a resumable training checkpoint**: it omits the child Adam state. Full checkpoint-save timings and sizes come from the separate baseline measurement.

A separate local metadata consistency audit passed: distinct canonical names, coordinate count, compressed-byte sum, norm agreement, counter increment and terminal cleanup. It did not perform a second decompression of every tensor; finiteness and exact reconstruction were checked inline by the exporter. Visual replay/scoring must independently verify raw chunk checksums when loading these artifacts.

Private evidence is backed up under `runs/candidate-cost-20260914-001/completed`: launcher, exit, phase events, step metrics, parent-restore timing, cost report, delta manifest and metadata-audit hashes. Large displacement chunks remain on the persistent network volume at `/workspace/praxis/runs/candidate-cost-20260914-001/exports/global_step_1/actor/delta`.

## Consequence for the pool

A literal 128-fold extrapolation of this cold-process construction path is 62.63 GPU-allocated hours before visual scoring, and approximately 845 GB if all deltas are retained. Parent restoration plus the export routine account for 70.1% of the measured candidate wall time. This includes CPU validation/compression as well as I/O; it is not an isolated disk bandwidth benchmark.

A persistent worker, reuse of an in-memory parent, or scoring before serialization could reduce overhead, but no such speedup has been measured. Do not subtract these costs from the budget based on an unimplemented optimization. Likewise, a faster GPU alone cannot be assumed to solve the dominant overhead.
