# RunPod: first GPU smoke test

Status: the user provisioned one H100 PCIe with a 100GB network volume.
Python 3.11, all 15 unit tests, synthetic branch replay and a BF16 CUDA computation
have passed on that host. The real Qwen two-branch run also completed; detailed
measurements remain in the ignored runs folder. Environment checks and a tiny
branch run alone are not transfer evidence.

Use one on-demand GPU Pod with SSH access. Preferred starting hardware is an
A100 80GB; an A40/A6000 48GB is a lower-cost alternative for the current 3B LoRA
engineering runner. Full-parameter/distributed training is not implemented here.
Use at least 48GB host RAM and budget 100GB of persistent workspace storage for
the environment, model cache, small input sample and checkpoints.

RunPod's public pricing page checked 2026-09-07 advertised A100 80GB at $1.59/hour,
A40 at $0.49/hour and RTX A6000 at $0.53/hour. These are planning references,
not a quote or availability guarantee; verify the console before deploying.
Storage is billed separately. Sources:
https://www.runpod.io/pricing and https://docs.runpod.io/pods/storage/types.

## Setup

Choose a Linux CUDA-compatible PyTorch template with SSH and a Python 3.11
environment option. Inspect its interpreter, conda path and mounted storage
before installing. Keep the repository, environment and HF cache under the
persistent workspace (usually `/workspace`). Do not copy BruinML credentials.

```bash
git clone https://github.com/DaybreakWannaCode/praxis.git /workspace/praxis
cd /workspace/praxis
command -v conda
nvidia-smi
df -h /workspace
```

If conda is available, use its actual absolute path with the tested installer:

```bash
bash scripts/setup_environment.sh /actual/conda/path /workspace/envs/praxis-alignment
```

The setup script isolates dependencies and runs CPU checks. It does not download
model weights or launch training. The H100 template inspected on 2026-09-07 had
`/usr/bin/uv` and `/usr/bin/python3.11`, while its default Python was 3.12. Use
the alternative installer on a template with those tools:

```bash
export UV_CACHE_DIR=/workspace/cache/uv
export HF_HOME=/workspace/cache/huggingface
bash scripts/setup_uv_environment.sh /usr/bin/uv /usr/bin/python3.11 /workspace/envs/praxis-alignment
```

Always inspect executable paths on a new template. Store dependency manifests
with each run. Initial downloads and environment creation on the network volume
took several minutes; preserve the environment for subsequent runs.

## Execution gate

Before the real smoke test:

1. Record pod GPU, driver, RAM, disk, git commit and dependency versions.
2. Run all unit tests and the synthetic two-branch test.
3. Download Qwen2.5-VL-3B-Instruct at an explicit commit and save the resolved
   revision in an ignored local config. The checked-in revision is a placeholder.
4. Prepare the pinned text source plus actual VIVA images using the converter
   described in the pilot README. Confirm train/score/dev separation.
5. Run input preflight, then two real candidate branches with the bounded config.
   Use tmux for setup/downloads/runs and keep logs in the ignored `runs/` folder.
6. Inspect replay equality, finite/nonzero gradients, reward degeneracy, truncated
   outputs, peak GPU memory and elapsed time. These are engineering diagnostics,
   not evidence of population-level transfer.

Use the measured memory and timing to plan the next rental. Do not launch the
72-branch study or assume full-parameter Praxis parity from this smoke test.
Export results and environment manifests before terminating a pod. Container
disk data is temporary; check the selected volume's lifecycle before relying
on it to preserve checkpoints after termination.
