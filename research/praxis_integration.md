# Praxis optimizer integration: implemented boundary, remaining production gates

The extension now supplies `transfer_alignment.praxis_bridge.PraxisStepRecorder`.
It observes the optimizer owned by the actual Praxis actor, rather than using
the standalone LoRA trainer as a substitute for Praxis training.

## Implemented and checked

`DataParallelPPOActor._optimizer_step` clips accumulated gradients, skips the
optimizer when the norm is nonfinite, calls AdamW otherwise, and clears gradients.
The observer wraps that existing boundary and registers PyTorch pre/post optimizer
hooks. It records the actual before/after optimizer coordinates, displacement,
norm, source hash and applied/skipped/failed status. It does not compute a new
training loss, replace clipping, or change optimizer arguments.

CPU tests extract and execute the exact method from the supplied original source.
They check that recording leaves a warm AdamW update and RNG unchanged, that saved
displacements equal the actual parameter differences, and that a nonfinite skip
does not produce a fabricated update. These are boundary tests, not an end-to-end
Ray/vLLM/FSDP reproduction. Set `PRAXIS_ROOT` to run them against another checkout.

The adapter supports two explicit scopes:

- `single_process`: optional fixed visual gradient using matching optimizer keys
  (`group0.param0`, etc.); its reference is fixed at attachment and becomes stale
  after updates. This is not automatically refreshed visual supervision.
- `rank_local`: raw local optimizer coordinates only. FSDP and multi-process use
  require this scope. A supplied visual gradient is rejected to prevent accidentally
  reporting a local dot product as global alignment. Padding and replicated shards
  are not yet removed/reduced for a scientific global score.

The mechanism uses the documented [PyTorch 2.8 optimizer hooks](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Optimizer.register_step_pre_hook.html).
FSDP changes parameter representations and optimizer construction requirements;
see the [PyTorch FSDP documentation](https://docs.pytorch.org/docs/2.8/fsdp.html).

## Reviewable upstream patch

`patches/praxis-optimizer-observer.patch` adds an opt-in attachment in the original
actor constructor and forwards its environment setting to Ray workers. It does
not edit the loss or optimizer-step body. `patches/source_hashes.json` records the
unmodified source files against which the patch was checked. The supplied original
Praxis directory remains unchanged.

In an isolated original-Praxis checkout with its dependencies installed:

```bash
# Run from the extension repository root; replace paths with actual checkouts.
git -C /path/to/Praxis-VLM apply --check /path/to/praxis/research/patches/praxis-optimizer-observer.patch
git -C /path/to/Praxis-VLM apply /path/to/praxis/research/patches/praxis-optimizer-observer.patch
export PYTHONPATH=/path/to/praxis/Praxis-Extension-main:/path/to/Praxis-VLM
export PRAXIS_ALIGNMENT_CAPTURE_DIR=/persistent/new-run/optimizer-capture
```

Then launch the original trainer with reviewed, bounded overrides. Use a **new**
capture directory for each run/restart; existing rank directories are rejected.
If attaching to an existing Ray cluster, configure the same environment on its
workers explicitly. The driver's environment alone is not an installation on
other machines. Keep capture outputs outside Git.

An additional opt-in patch, `patches/praxis-rollout-state.patch`, writes/loads a
sidecar at the original worker checkpoint boundary. `praxis_state.py` captures
module modes, process RNG and the rollout manager's persistent `torch_random_states`,
`gen_random_states` and `freed_bytes`. These manager attributes are distinct from
the process RNG already saved by the original checkpoint manager. Tests verify
restoration and reject a changed rank/world-size topology. Apply this patch in
addition to the optimizer patch when testing checkpoint replay. An enabled load
requires its matching sidecar; do not silently fall back to incomplete state.
It still does not serialize vLLM requests/engine internals or the driver data
iterator. End-to-end rollout replay remains a required production check.

## Findings that affect baseline reproduction

1. The current public text dataset uses `problem`; the supplied MCQ launcher
   specifies `question`. Prepare and validate local train/validation files rather
   than assuming its referenced `test` split exists.
2. The MCQ launcher is named for 7B and points to a 7B path, although its experiment
   name mentions 3B. Pin the actual 3B model explicitly.
3. Original defaults include long responses, large batches, full-parameter AdamW,
   actor/reference offload and vLLM. Our short LoRA memory result does not size this
   workload. A separate environment is required; do not install its unconstrained
   requirements over the working calibration environment.
4. The original FSDP checkpoint manager saves model, optimizer, scheduler and
   per-rank CPU/CUDA/NumPy/Python RNG. This alone does not establish exact replay
   of the trainer data iterator, Ray workers, rollout engine state or response pool.

## Remaining gates before claiming production integration

### Runtime preflight and its limits

The original source's minimum vLLM version is 0.7.3. Its published
[package metadata](https://pypi.org/pypi/vllm/0.7.3/json) requires PyTorch 2.5.1,
torchvision 0.20.1, torchaudio 2.5.1, NumPy below 2 and, on Linux x86-64,
xformers 0.0.28.post3. Transformers 4.49.0 is a candidate matching the original
minimum: its [tagged Qwen source](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
contains the older attention implementation referenced by Praxis.

`praxis-runtime-candidate.txt` pins a candidate stack. The first resolver attempt
showed that vLLM 0.7.3 specifically requires Ray 2.40.0 through its `adag` extra;
the candidate now uses that version. These are compatibility starting points,
not a validated end-to-end environment.
The isolated Python 3.11 runtime passed original trainer imports, the Qwen
attention patch, bounded config/data loading and a BF16 FlashAttention GPU
forward/backward check on the H100. It uses PyTorch 2.5.1, vLLM 0.7.3,
Transformers 4.49.0 and FlashAttention 2.7.4.post1. The supplied requirements
omitted math-verify, latex2sympy2-extended and TensorBoard despite unconditional
imports. The candidate pins also prevent the math parser from forcing an
OmegaConf downgrade that removes `to_object`. Detailed locks and failed/passed
preflight records remain in ignored run archives. The subsequent one-step
uninstrumented baseline completed original vLLM generation, finite nonzero
gradients, validation and checkpoint saving on one H100 80 GB. A full saved-weight
comparison then found **zero parameter displacement**: the original constant
scheduler initializes LR to zero even when warmup is zero. Its reported LR is
logged after advancing the scheduler. The first step initializes Adam moments but
does not move weights. A second step and another displacement check are required;
successful execution and nonzero gradients alone do not validate an update.
Do not downgrade the calibration environment.

For the current pod, the original runtime lives in the isolated container path
`/opt/praxis-original`: the persistent volume was very slow when installing
thousands of package files. Models, data, checkpoints and the resolved environment
record remain persistent. Rebuild this environment after replacing the container.

`scripts/check_praxis_runtime.py` checks original trainer imports, the original
Qwen attention monkey patch, optional bounded config/data loading, and an optional
small FlashAttention GPU forward/backward pass. It writes failures as well as
successes to JSON. Passing these checks does not establish vLLM/FSDP execution.

`praxis-baseline-bounded.yaml` is a two-step engineering configuration: four text
prompts, five completions per prompt, unchanged original MCQ reward, full-parameter
AdamW, one GPU and a 512-token response cap. It disables compilation/CUDA graphs
for debugging and retains one checkpoint. These explicit departures from the
paper's scale make it a runtime baseline, not a reproduction of reported scores.
`scripts/prepare_praxis_baseline.py` prepares eight fixed distinct text prompts
with source-row and file hashes, split four/four for train/validation. This text
validation checks the pipeline; it is not the independent visual endpoint.

The original trainer always saves a final checkpoint, even with `save_freq=-1`.
Budget for model weights and full Adam state before launch. Do not interpret the
network filesystem's shared backing capacity as the purchased volume quota.

### Execution gates

- The initial uninstrumented text-only run completed against the immutable supplied
  source snapshot, pinned base model and hashed text split, but its first step had
  zero LR and zero displacement. Verify an actual nonzero second step before
  instrumentation parity; paper-scale baseline reproduction remains outstanding.
- Repeat identical fixed-rollout optimizer work with instrumentation enabled and
  verify weights/state match. Keep rollout generation separate when isolating
  optimizer correctness.
- Validate local shard identity, padding masks and process-group reduction against
  a small full-coordinate reference on at least two ranks before reporting global
  gradient/update alignment. Replication must not double-count norms or dot products.
- Extend complete branch restoration to all state influencing rollouts and batches;
  test it with warm optimizer state in the production runtime.
- Compute the visual expected-correctness gradient at the same parent and in the
  same measured coordinates. Use independent scoring/outcome splits.

No full-parameter Praxis run or distributed global alignment result has been
obtained merely by adding this observer. The separate patch makes the next
production step concrete and reviewable without claiming those remaining gates
are complete.
