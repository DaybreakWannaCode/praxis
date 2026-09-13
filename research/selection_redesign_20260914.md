# Revised transfer study and decision plan

H1/H4 is closed: 0/6 jointly resolved comparisons under the predeclared rule. Both predictor and outcome uncertainty matter. The next inexpensive check uses archived updates and a fixed answer-likelihood surrogate; see choice_probe_development_lock.md. Do not expand H or sampled-response counts.

## Ordinary Praxis before/after study — prepared, not launch-authorized configuration

Retain the original one-stage decision-training entry point and composite text reward. The existing four-prompt engineering data must not be called a meaningful training corpus. Audit the actual released text inventory, scene identities, images and overlap before freezing the run. The source Parquet metadata was inspected on the pod on September 14: 10,000 rows, with problem and answer fields. This is a raw inventory, not 10,000 independently audited eligible scenes. Use a single declared pass over the eligible training corpus, with an explicit numeric optimizer-step count computed from that inventory and the original batch/rollout settings. Do not substitute a guessed step count. A runtime forecast must precede launch; if unaffordable, revise and record the budget before any endpoint outcomes are inspected.

Evaluate the same starting and final models with one greedy answer per eligible development image. Freeze the largest audited affordable distinct-scene panel, image hashes, prompt, maximum output length and explicit_final_v3 parser before training. Report paired scene correctness changes (unparsed = incorrect), parsing and truncation, plus scene-cluster uncertainty. Do not compare these greedy results directly to the old sampled rates. Count repeated questions in a scene as dependent. This one-seed baseline is feasibility evidence, not a seed-robust reproduction.

Prespecify a 32-scene image-dependence subset using a stable hash of scene ID (or all scenes if fewer). Use a fixed derangement of images from different scenes at both endpoints; keep question/options unchanged. Report correct-image gain, shuffled-image gain, and their difference on the same subset. Shuffling can create distribution shift: it tests reliance on image information, not a localized reasoning mechanism. Keep final test untouched.

Checkpoint policy: initial, halfway and final; full optimizer state only where continuation requires it. Record disk estimates before launching; remote filesystem df reports backing-cluster capacity, not the user's network-volume quota. Keep logs/configs and per-image results on /workspace and back them up locally. No automatic early-stopping analysis or horizon extension.

## Prospective selection study — conditional, no sweep yet

Three paired seeds; same initial model, optimizer state, hardware, candidate pool, selected prompt count and actual training budget per arm. Arms: random, realized-update likelihood alignment, direct likelihood lookahead. Score candidate batches from the same parent; discard tentative optimizer/model changes after scoring; train selected data with fresh rollouts. A static selector must be labeled static; do not describe it as an online curriculum. Cosine and norm are diagnostics, not assumed inferior selectors.

Primary outcome: paired-seed difference in final greedy visual correctness versus random on untouched test scenes. Secondary: direct lookahead comparison, parse/truncation, development learning curves. Report each seed, scene-cluster uncertainty conditional on seeds, and between-seed variation separately. Three seeds cannot guarantee narrow uncertainty. Fix candidate-pool size, selected fraction, training updates, test inventory and practically worthwhile improvement after inventory/cost assessment but before arm outcomes; until those fields are numeric this is a proposed design, not a preregistered launch plan. Include all scoring, tentative updates, training and evaluation in the cost comparison. Charge random only for work it actually needs and also compare equal total compute where feasible.

Visual calibration affects selection: describe this honestly as text-only parameter updates with labeled visual calibration, not training entirely without visual supervision. The strongest eventual specificity control is image-probe versus matched caption-probe selection; score-only comparison first, no extra training arm now.

## Verified related work and novelty boundary

- [GradAlign](https://arxiv.org/abs/2602.21492): February 2026 first submission; revised July 2026; abstract describes selection of RL training problems using candidate policy gradients aligned with a small validation set, forming an adaptive curriculum. Generic RL gradient-selection novelty is therefore unavailable. An implementation-level algorithm audit is required before claiming a reproduced GradAlign baseline. A generic cosine or direct-lookahead arm is not that reproduction.
- [LESS](https://arxiv.org/abs/2402.04333): optimizer-aware instruction-data selection, Adam adaptation and low-dimensional gradient features. Distinguish its instruction-tuning setting from our original-Praxis RL updates.
- [My Answer is C](https://arxiv.org/abs/2402.14499): evidence that option-token probabilities and generated answers can disagree. Smooth likelihood improvements cannot substitute for generated-answer evaluation.

The defensible research question is whether labeled visual calibration can select text RL data that improves actual vision-grounded answers at useful total cost. The Taylor identity is mathematical motivation; surrogate prediction alone is an engineering result. Report a negative outcome without escalating experimental complexity to manufacture a positive one.
