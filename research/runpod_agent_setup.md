# RunPod agent setup — 2026-09-12

Followed https://docs.runpod.io/agent-setup.md for Codex.

Verified locally:
- Official marketplace `runpod` registered from
  https://github.com/runpod/runpod-plugins-official.git.
- Installed eight currently published skills: runpod, runpod-mcp, runpodctl,
  flash, runpod-usage, companion-clis, runpod-migrate, runpod-templates.
- Marketplace plugin version 1.2.0 bundles the hosted MCP endpoint
  https://mcp.getrunpod.io/.

Pending user UI step from the official guide: open Codex Plugins, select the
Runpod marketplace, install Runpod (Official), and reload if prompted. Complete
browser OAuth when requested. No account connection or pod listing has yet
been verified. No API key, CLI, or Flash SDK was installed by this setup.

If plugin installation does not register MCP, this installed Codex CLI uses
`codex mcp add runpod --url https://mcp.getrunpod.io/` (the guide's `--transport`
syntax does not match local help). Do not duplicate a working bundled server.

After authentication, list existing pods and inspect the existing experiment
and persistent volume before launching anything. This setup made no pod,
volume, or experiment changes.

## Verified connection — 2026-09-12

MCP pod and volume listing succeeded after user installation/sign-in. Running
pod `e8lcnuph2r1z3r` uses A100-SXM4-80GB at reported compute cost $1.59/hour,
with volume `86u1nnngpl` (`praxis_volume`, 150 GB, US-KS-2) at `/workspace`.
SSH `root@216.81.245.126` port `19499` succeeded with the existing local key.
Project files are present. Candidate `production-h4-001/completion.json`
reports complete=true, train_exit=0, 2344.5008902549744 seconds, horizon=4,
completed_steps=4, status=passed. Tensor checksums still need independent
verification before accepting this export. `/opt/praxis-original/bin/python`
is not executable/present on this replacement container; restore the pinned
runtime before further experimental work. No runs were launched by this check.
