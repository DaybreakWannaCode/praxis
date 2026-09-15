# Praxis transfer alignment

Independent experiment work for predicting visual-task transfer from text-only RL updates.
Repository: https://github.com/DaybreakWannaCode/praxis

The primary quantity is the inner product between an expected-visual-correctness gradient
and an **actual** text optimizer displacement. Cosine and norms are diagnostics.
The later likelihood probe targets a different surrogate; see the
[methods and estimand distinction](research/paper_methods_draft.md). Its gradient
is not a gradient of greedy accuracy or expected generated-answer correctness.

## Current status

September 15: the [independent visual baseline](research/independent_baseline_result_20260915.md)
is complete: greedy accuracy changed from 201/256 to 203/256 (+0.78 percentage
points; paired 95% interval −1.95 to +3.91 points). This is inconclusive, not a
confirmed transfer benefit. The reserved final test remains unused.

Two actual candidate lifecycles retained about 34.3 MB after releasing about
13.18 GB of managed temporary exports. The [storage strategy](research/storage_strategy_20260915.md)
bounds candidate accumulation; it does not yet make full training fit alongside
historical artifacts on the 250 GB volume. No historical checkpoint was deleted.

[Original-worker restoration](research/restore_only_execution_20260915.md) passed
exact model/Adam/scheduler/RNG value comparison. Separate original-manager RNG
and original-dataloader continuation checks also passed; see
[validation evidence](research/checkpoint_structure_validation_20260915.md).
The [save adapter and interrupted-publication inspection](research/checkpoint_adapter_progress_20260915.md)
have local tests, but real-worker checkpoint publication and post-restore update
or rollout replay remain unverified.

The [revised cost/scope decision](research/selection_reassessment_20260915.md)
compares 64- and 128-candidate studies. The smaller two-arm proposal keeps three
paired seeds and 32 training updates per arm. It is not launched: final protocol,
storage capacity and total compute budget remain unresolved. No independent
alignment-prediction or beneficial selection result is established.

September 14: the bounded likelihood and execution-cost checks are complete. A
single-A100, full-parameter, adapted original-Praxis baseline finished 32 updates;
its step-16/32 model and optimizer checkpoints passed integrity checks. One fresh
candidate restored the warm parent, advanced the optimizer once and exported an
exactly reconstructable canonical displacement. Endpoint greedy timing and replay
also passed. This does not establish a visual-transfer benefit or distributed replay.

The older sampled-correctness H1/H4 study remains inconclusive. The subsequent
answer-likelihood probe is a smooth surrogate measurement, not proof that likelihood
alignment predicts generated-answer improvements. No full selection matrix has launched.

See the [execution estimate](research/selection_execution_estimate_20260914.md),
[fresh-candidate timing](research/candidate_cost_result_20260914.md),
[baseline timing](research/ordinary_baseline_result_20260914.md),
[endpoint timing](research/greedy_endpoint_result_20260914.md), and
[likelihood result](research/choice_probe_result_20260914.md).
Raw responses, checkpoints and run receipts remain outside Git. Earlier synthetic
and LoRA smoke infrastructure is retained for debugging; its results are separate
from the later original-trainer measurements.

Bounded inference calibration and repeated visual-probe diagnostics are available in
`transfer_alignment.calibrate` and `transfer_alignment.stability`. They do not train
the model or access the final test split. The original Praxis optimizer observer and
checkpoint sidecar are documented in [integration status](research/praxis_integration.md).
Their boundary tests do not establish complete distributed trainer replay.
See [measurement precision](research/measurement_precision.md) for paired evaluation,
fixed-repeat uncertainty and the versioned explicit-final-answer parser correction.

## Start here

```bash
cd Praxis-Extension-main
python -m unittest discover -s transfer_alignment/tests -v
python -m transfer_alignment --output runs/synthetic-smoke
```

Use Python 3.11 for the GPU environment. CPU tests also ran with Python 3.9/PyTorch 2.8.
See [pilot instructions](Praxis-Extension-main/transfer_alignment/README.md), the
[master protocol](research/experiment_protocol.md), and the
[submission scope](research/submission_scope.md).

## Layout

- `Praxis-Extension-main/transfer_alignment/`: new experiment code and CPU tests.
- `Praxis-Extension-main/task_0/`: existing measurement code, retained for shared
  prompt, log-probability and answer-parser utilities. Its older gradient-gradient
  estimator is not the headline metric in the new runner.
- `research/`: protocol and implementation decisions.

The original Praxis repository, local PDFs, benchmark images, model weights, run outputs
and credentials are intentionally excluded from Git. To use the released MCQ reward,
obtain the original reference code alongside the extension:

```bash
git clone https://github.com/Derekkk/Praxis-VLM.git Praxis-VLM-main
```

Record and pin its commit before science runs. The source tree supplied with this project
is a reference snapshot, not evidence of a completed reproduction. Preserve upstream
copyright/license notices when integrating upstream code. No project-wide license is
assigned here on behalf of collaborators.

## Git workflow

Keep `main` as reviewed milestones; use focused branches for later changes. Original
training computations are retained; opt-in instrumentation uses isolated source copies. Future integration with collaborators
can use ordinary pull requests or cherry-picks once their target repository is identified.
Never commit SSH passwords, tokens, run checkpoints or benchmark data.
