# Agreed execution scope

6 September 2026. This narrows the master protocol; it does not replace its estimator,
split, provenance or independence requirements.

1. Build the small engineering pipeline: one parent, two text candidates, actual
   displacements, exact alignment, full state restoration, independent outcomes.
2. A short full-parameter 3B feasibility run sets evaluation count and a fixed 1- or
   4-step displacement window. Do not run a separate full LoRA science sweep first.
3. Main prediction study: 3 parent seeds x 3 checkpoints x 8 candidates (72 branches),
   with within-parent analysis and uncertainty preserving shared parent/run effects.
4. Main intervention: alignment, random and cosine-selected text, 3 seeds each,
   identical initialization and declared training budgets. Log actual tokens and cost.
5. Preserve 6-8 checkpoint opportunities on random trajectories for a limited transfer
   saturation / exploratory stopping analysis if the measurement budget permits.
   Stopping thresholds are developed without final-test access. Probe cost is not free.
6. Save illustrative text examples and a small score-staleness analysis where existing
   checkpoints permit it. Defer 7B, caption grids and a deployed stopping method.

The optional analysis budget is about 10-15%, subordinate to independent prediction
and selection versus random. The full-parameter production backend has priority over
expanding local adapter experiments.

## Implementation evidence (updated 7 September 2026)

The CPU checks cover an exactly enumerated RLOO expectation, finite differences,
FP64 reduction, RNG isolation, warm Adam/scheduler/reference restoration, clipping/KL,
split rejection, two-branch end-to-end replay, response-token masking, temperature,
EOS handling and sequence-summed visual gradient extraction. This is engineering evidence only.

The first pinned Qwen3B LoRA two-branch smoke completed on an H100, including exact
branch replay and independent real-image outcomes. A bounded generation calibration
also completed. Detailed results and provenance remain outside Git in the local run
archive. These establish the small backend's operation, not a transfer benefit.

The original Praxis optimizer boundary has an opt-in observer and an additional
checkpoint sidecar for exposed rollout-manager RNG state. See
[integration status](praxis_integration.md) for tested behavior and limitations.
Distributed Praxis parity, complete rollout replay, full-parameter memory fit and
training-effect detectability remain unvalidated. Repeated probe and no-update
outcome checks precede scaling; full checkpoint storage and scoring overhead must
be measured for the production backend.
