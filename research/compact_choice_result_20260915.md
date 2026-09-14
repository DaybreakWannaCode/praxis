# Compact candidate scoring: real-model check passed

The bounded scoring-only run completed with exit 0. All 824 canonical tensors were read in chunks, checked against their raw archive checksums and used to reconstruct the child. The parent was restored and its first scoring-image likelihood replayed exactly afterward. The GPU was empty after completion. All output files match the remote SHA-256 receipt, and a local audit independently reconstructed the direct score and checked scalar consistency.

| Measurement | Result |
|---|---:|
| Existing input displacement archive | 6,601,372,705 bytes |
| New measurement JSON receipts, combined | 146,098 bytes |
| New model or gradient files | 0 |
| Shared CPU gradient RAM | 30.04 GB |
| Parent checksum verification | 156.78 s |
| Backend loading | 85.60 s |
| Sixteen-image gradient | 491.88 s |
| Chunked projection, checksum and child application | 166.76 s |
| Direct child scoring | 5.99 s |
| Internal end-to-end elapsed time | 922.44 s (15.37 min) |

The 146 KB figure describes newly written measurement receipts. The pre-existing 6.6 GB input archive remains intact. This does not yet demonstrate zero-export candidate construction or automatic reclamation across a pool. The old file-based scorer also held its gradient in RAM; do not attribute all storage reduction to a new gradient-caching technique.

Alignment was -0.037198434486; direct likelihood change was -0.037291825712, an absolute difference of 0.000093391226 natural-log-probability units per image. Both use the same fixed 16-image calibration panel and normalized answer-label-prefix objective. Their close agreement is a useful local implementation check, **not independent visual transfer evidence, an accuracy change, or proof of candidate-ranking validity**. No development or final-test outcome was evaluated in this run.

Twenty-five local CPU tests passed across streaming projection, candidate guards, input capture and checkpoint publication. The eighteen new component tests also passed under the pod's PyTorch 2.5.1 environment. A separate CPU-only smoke test used the original `verl.protocol.DataProto`: input serialization was exact and produced a 15,964-byte synthetic receipt. On the actual network filesystem, tiny checkpoints supported file/directory synchronization, replacement publication and preservation of the previous checkpoint after a simulated interrupted write. This did not run a real actor update or validate full-size checkpoint integration.

Code is opt-in and has not changed the completed baseline/candidate training paths. Original-worker input-capture validation, managed temporary cleanup, full-size transactional checkpoint integration and archive migration remain the next gate. The full matrix has not launched. No historical model, checkpoint or displacement was removed.

Evidence is private under `runs/compact-choice-check-20260915-001/completed` and `runs/storage-components-20260915-001`. Frozen run-plan hash: `947d4c33efc152a443d62cd111e11bad86db762239a45ba8eb18b2108a98e596`.
