# Choice-probe execution ledger — 2026-09-14

The H1/H4 study remains closed and inconclusive. New protocol: choice_probe_development_lock.md; future study: selection_redesign_20260914.md.

Implementation committed as 469b288; portable launcher fix 1e5301f. Thirteen local objective/contract tests passed. No new text training or generated reasoning responses are part of this check.

Existing A100 pod was live and idle when checked. Its Python environment reports torch 2.5.1+cu124 / transformers 4.49.0. Attempt choice-probe-dev-20260914-001 exited 127 because /usr/bin/time was absent, before any model work; retained. Attempt choice-probe-dev-20260914-002 uses Bash built-in timing, same experiment settings, two-hour cap, and started loading checkpoint shards. The parent score/development panel subsequently completed and passed exact replay; parent.json was copied locally. At the latest check, 12/16 score-image gradients were accumulated without error. Observed GPU usage during gradient work was about 39 GB, not the final peak. No candidate result or completed cost report exists yet.

Remote output: /workspace/praxis/runs/choice-probe-dev-20260914-002. The source and protocol are on persistent storage. Run exits are recorded in run.exit. A complete measurement requires manifest status complete, summary.json, four candidate outputs and final parent replay. Do not read partial outputs as scientific success. Full measured cost and rankings are pending completion.

If the pod is paused mid-gradient, that gradient phase is not resumable in this first implementation. Previously completed phase outputs survive on /workspace. Preserve the interrupted directory; any engineering restart must use a new directory and retain its cost. No automatic sweep, new prompt, horizon or training run.

Latest planning commit: e9e08fc. Proposed baseline: 1,024 audited text prompts, five rollouts each, one pass; 256 audited visual development scenes with greedy evaluation. These are proposals pending audit and cost gates, not launched work.
