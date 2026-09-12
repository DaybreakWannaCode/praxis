# Production experiment completion audit — in progress

This ledger distinguishes inspected evidence from pending completion. The H4
measurement is still running; this document does not establish a transfer result.

## Verified before final H4 outcomes

- The frozen protocol retains original composite text reward, binary visual
  correctness under `explicit_final_v3`, fixed score/development panels, four
  candidates, 1,800 responses and a ten-hour evaluation cap. H1's unresolved
  precision result triggers the single predeclared H4 fallback.
- Re-evaluation of all 1,800 archived H1 response texts and the first 1,280 H4
  responses reproduces every saved reward, parsing status and extracted choice.
  All three parser-source hashes match `production_reward_contract_v3.json`.
  The local visual manifest hashes to
  `beb7f1130403d5daf5dd6e938f2f02b1b5391ba298392119b8cba8e5983e17da`.
- Ten live remote measurement source files match the locally inspected sources.
  The private evidence is `measurement-source-audit.json` in the local H4 run
  directory. Source equality establishes which implementation is being audited;
  it does not itself establish estimator correctness.
- Inspection of `visual_gradient` and `loo_advantages` confirms detached
  leave-one-out reward baselines, positive sequence-log-probability gradients,
  response/image averaging, and no standard-deviation normalization, token
  averaging, reference KL or auxiliary reward in the visual estimator. The
  production backend samples and scores with BF16 autocast over FP32 master
  parameters, with temperature/top-p one and top-k disabled.
- All four H4 completion records report complete, exit zero and four steps.
  Their export-verification records report 824 tensors, 3,754,622,976 canonical
  elements and verified file checksums. These are retained verification records;
  this audit did not repeat the large tensor reads while evaluation was live.
- All four parity reports captured at H4 launch pass their control/observer
  equality fields. Their parent parameters, buffers, optimizer and scheduler
  digests agree across candidates. Worker digests differ across candidate
  rollouts; worker equality is verified within each fixed-rollout
  control/observer comparison, not claimed across candidates.
- The read-only archive audit passes the completed H1 archive, reconstructs its
  outcome tables and uncertainty, and reproduces zero jointly resolved pairs.
  Missing-response, altered-replay and altered-interval fixtures are rejected.

## Still required

1. H4 terminal exit zero, complete manifest/summary, all 1,800 responses and exact
   eight-response parent replay. A live process or partial outcome files do not
   satisfy this requirement.
2. Full H4 archive backup with remote/local checksum agreement; rerun the archive
   audit and parser reconstruction over the complete panel.
3. Confirm final input/config identities and compare sealed scores with the
   pre-outcome local backup. The saved boolean seal alone is insufficient.
4. Report all six pairwise score/outcome intervals, simultaneous intervals,
   parent comparisons, parse/truncation diagnostics, and the locked feasibility
   decision, without choosing a favorable subset.
5. Report terminal runtime and GPU/host peak memory, including the separate cost
   of retained failed/recovery attempts. Label observed peaks and billed runtime
   appropriately.
6. Final recommendation respecting the H1/H4 rule. If H4 is also unresolved,
   report insufficient precision at this bounded budget; do not enlarge the
   study, retry seeds or launch a main sweep.

Final interpretation must retain the fixed, availability-limited development
panel, approximate four-repeat score intervals, exploratory pointwise decision,
greater H4 Taylor error, and mixed H100/A100 candidate-training hardware caveats.
No final-test data or new training is authorized by this audit.
