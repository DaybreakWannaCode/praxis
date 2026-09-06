# Praxis transfer alignment

Independent experiment work for predicting visual-task transfer from text-only RL updates.
Repository: https://github.com/DaybreakWannaCode/praxis

The primary quantity is the inner product between an expected-visual-correctness gradient
and an **actual** text optimizer displacement. Cosine and norms are diagnostics.

## Current status

Engineering milestone: a CPU synthetic backend and a single-GPU Qwen2.5-VL-3B LoRA
backend share an estimator, optimizer/state restoration, sequential candidate branches,
exact dot products, and independent parent/child evaluation. The synthetic backend is
tested locally. The first real Qwen two-branch smoke test also completed on an H100:
both branches replayed exactly and independent visual evaluation completed.
Detailed measurements and run outputs remain in the ignored local runs folder.
Distributed/full-parameter Praxis parity is not yet verified, and
this tiny engineering run does not establish a VLM transfer benefit.

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
Praxis code paths are not edited by this milestone. Future integration with collaborators
can use ordinary pull requests or cherry-picks once their target repository is identified.
Never commit SSH passwords, tokens, run checkpoints or benchmark data.
