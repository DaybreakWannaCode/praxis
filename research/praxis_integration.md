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

- Pin the original source commit or immutable snapshot, reward, model and data;
  validate a short uninstrumented text-only baseline with the original trainer.
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
