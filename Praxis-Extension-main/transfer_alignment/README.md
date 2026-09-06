# Stage 0 and bounded LoRA branch runner

This package implements the first engineering milestone, not the 72-branch paper study.
It does not start long runs, rent GPUs, touch final test outcomes or train on images.

## CPU verification

From `Praxis-Extension-main` in an environment with PyTorch and NumPy:

```bash
python -m unittest discover -s transfer_alignment/tests -v
python -m transfer_alignment --preflight --output runs/unused
python -m transfer_alignment --output runs/synthetic-smoke
```

Output directories must be new. The synthetic model has two actions and a fixed reference
buffer. Its gradients, updates and outcomes are mathematical/plumbing checks, not VLM data.

## GPU smoke run

Use a dedicated Python 3.11 environment. Install the compatible CUDA PyTorch build for
the host, then install `requirements-gpu.txt` and save `pip freeze`. The initial pins are
not yet a verified GPU environment lock. Do not alter the shared system environment.

The repository also provides `scripts/setup_environment.sh CONDA_PATH ENV_PREFIX`.
It uses the dedicated interpreter by absolute path, validates its prefix, and runs
CPU-only checks. This avoids accidentally installing into an inherited active venv.
Run setup in a named tmux session on a remote host; keep its log under ignored `runs/`.

1. Check current GPU utilization. The engineering script is not a scheduler; do not run
   while another job consumes the device.
2. Make a local copy of `configs/qwen_smoke.json`. Set `revision` to the model's pinned
   Hugging Face commit. Default `local_files_only=true` avoids accidental downloads.
3. Prepare a small normalized manifest (below). Use 2+ train examples, 2-4 scoring
   images and 2-4 independent development images for the first smoke run.
4. Run preflight, then the bounded run inside your own tmux session:

```bash
python -m transfer_alignment --backend qwen --config /path/to/local-config.json \
  --manifest /path/to/manifest.json --preflight --output runs/unused
python -m transfer_alignment --backend qwen --config /path/to/local-config.json \
  --manifest /path/to/manifest.json --output runs/qwen-two-branches
```

The example 256-response-token/256-image-token budget is for plumbing. It is not the
paper evaluation policy; inspect truncation before increasing scope. LoRA adapters are
FP32, base weights BF16, decoder attention only, rank 16/alpha 32, dropout zero.
The model is trained with non-reentrant activation checkpointing and sampled in eval mode.
Generation and the visual score-function derivative both use full-softmax temperature 1.

The fixed reference policy is the pinned base with adapters disabled. The full parent
snapshot covers base weights and buffers; compact child checkpoints are only valid
relative to that parent. A different reference checkpoint needs an explicit backend.

## Manifest schema

JSON list, one object per situation. All fields below except `situation` and `image_path`
are required. Paths are relative to the manifest; image files are decoded during validation.

```json
[
  {
    "id": "text-001",
    "group_id": "source-scene-001",
    "split": "train",
    "question": "Which action is appropriate next?",
    "action_list": ["A. Stop safely", "B. Continue"],
    "answer": "A",
    "situation": "A description from the actual text training corpus."
  },
  {
    "id": "visual-001",
    "group_id": "different-scene-001",
    "split": "score",
    "question": "Which action is appropriate next?",
    "action_list": ["A. Stop safely", "B. Continue"],
    "answer": "A",
    "image_path": "images/visual-001.jpg"
  }
]
```

This fragment illustrates the schema, not a complete executable manifest: `dev` records
and enough distinct training examples are also required. `test` records may be listed
but this engineering runner never evaluates them. It rejects cross-split scene IDs,
normalized description duplicates and exact image-file hashes. This is not a semantic
near-duplicate audit; manually audited scenario provenance is still required.

## What is measured

- Visual gradient: RLOO baseline, sequence-summed log probabilities, correctness only;
  no group standardization, token-length division or visual training step.
- Text update: freshly sampled per-prompt groups, standardized GRPO advantages,
  clipped ratios, fixed-reference KL, token-mean loss, AdamW and norm clipping.
- Actual displacement: FP32 difference before/after each optimizer step in named,
  deduplicated trainable coordinates; dot multiplication/reduction in FP64 chunks.
- Branch score: parent visual gradient dotted with the entire candidate displacement.
  For a four-step branch this is a tentative-window score, not a first-step forecast.
- Replay: full parent model, warm Adam state, scheduler, module modes and Python/NumPy/
  CPU/CUDA RNG restoration. The runner requires bitwise replay; a failed GPU replay
  is an instrument failure, not permission to silently loosen tolerances.
- Outcomes: fresh sampled visual correctness and greedy accuracy on `dev`, with
  independent parent/child rollout seeds and per-example records.

All candidate scores are written and sealed before any child visual outcome is evaluated.
The same parent's evaluation is shared across candidates; those outcome errors are correlated.
The small-run summary is descriptive and does not claim statistical significance.

## Artifacts

`manifest.json`, `parent.pt`, `visual_gradient.pt`, per-step delta files, compact child
states, `scores.jsonl`, `scores_sealed.json`, `updates.jsonl`, response JSONL files and
`summary.json`. Full parent checkpoints contain optimizer and RNG objects: load only
locally trusted checkpoints. Large outputs and local inputs stay outside Git.

## Deliberate limits

The single-device trainer uses one candidate batch per optimizer step with global
token-mean aggregation. The released distributed Praxis trainer averages microbatch
losses separately and has its own rollout/minibatch schedule. Therefore this is
GRPO-style engineering support, **not a validated replacement for Praxis training**.
The `correctness_only` default is an ablation profile; `released_code` imports the original
MCQ reward unchanged when the reference checkout and mathruler are installed.

Next production milestone: attach delta/state capture at actual Praxis optimizer-step
boundaries, verify full-parameter restoration, calibrate independent visual outcome noise,
then lock the 72-branch study. No automatic early stopping is implemented at this stage.

Implementation references: [Qwen2.5-VL](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
and [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
