# Transfer alignment for text-only RL in VLMs

Experiment protocol, 6 September 2026. Proposed design; no new model runs were executed while preparing it.

## 1. Paper objective and scope

Test whether the **actual parameter displacement produced by text-only GRPO**, evaluated against an independently estimated visual reward gradient, predicts and helps improve visual-task transfer.

The contribution should have three layers:

1. **Prediction:** rank alternative text updates from the same model and optimizer state by their subsequent effect on unseen visual examples.
2. **Dynamics:** establish how useful that prediction remains across training checkpoints and prediction horizons.
3. **Intervention:** use alignment to select text data, then test whether the resulting training policy improves visual performance under matched budgets.

Efficiency and early stopping are additional claims with their own tests. A successful selection experiment does not automatically establish net compute savings. Initial alignment predicting a complete run is a separate, stronger hypothesis than local prediction.

Limit the initial scope to vision-grounded multiple-choice decision making, matching Praxis. These benchmarks alone do not establish general visual reasoning, better perception, or the location of shared reasoning mechanisms.

## 2. What the supplied material establishes

The supplied original paper, `2503.16965v3.pdf`, reports Qwen2.5-VL 3B/7B, full-parameter training, GRPO group size 5, KL coefficient 0.01, learning rate 1e-6, and greedy benchmark evaluation. Its one-stage ablation is the clean starting point for isolating text-only decision training. The math cold-start stage is an additional treatment and should not be mixed into the primary comparison. Appendix A explicitly proposes identifying effective subsets of textual scenarios.

The supplied repository uses AdamW, gradient clipping, KL loss, accumulation, and multiple minibatch optimizer steps inside an actor update. Its MCQ reward is:

`accuracy + 0.8 * format + 0.4 * tag_count + 0.5 * length`.

This differs from the draft pilot's correctness-only reward, and the paper's Stage 2 prose does not describe precisely the same component list. Preserve a named **released-code profile** and a separate **correctness-only ablation**. Do not silently substitute one for the other.

`Praxis-Extension-main/task_0` is an apparatus for fixed-checkpoint gradient measurements, with no optimizer step. It uses Qwen2.5-VL-3B, a VIVA image/description pool, sketched score gradients, and a noise study. Its default group standardization and token aggregation produce a GRPO-style direction, not generally the derivative of unweighted expected correctness. Its text gradient is obtained from paired VIVA descriptions, not necessarily the actual Praxis training corpus. Those are useful diagnostic conditions, but they do not yet implement the proposed main predictor.

The `praxis_theory` figures are synthetic demonstrations. The supplied Task 0 notebook contains no saved execution outputs, and no saved real-VLM measurement tables were found in the extension tree. Runtime estimates in its README are planning statements, not verified timings on your hardware.

Dataset accounting needs an audit: the original paper reports 1,240 VIVA test instances; the pilot README describes 1,062 images and warns of unavailable URLs. These may reflect different releases, counting units, or availability. Establish the actual manifest before setting exact splits or comparing published scores.

## 3. Define the quantity before measuring it

Let z = (image, question, options, correct answer) and let y be the entire generated response, including reasoning and final answer. Define

\[
J_V(\theta)=\mathbb E_{z\sim D_V}\mathbb E_{y\sim\pi_\theta(\cdot|z)}[r(z,y)],
\qquad r\in\{0,1\}.
\]

Use a locked parser, score invalid answers as incorrect, and report parsing and truncation rates separately. The policy includes its temperature, stopping conditions and response cap. For the primary differentiable objective, use temperature 1, full categorical sampling, no top-k/top-p truncation, and sequence-summed response log probability. Keep these settings fixed across visual gradient estimation and stochastic outcome evaluation. Any policy alteration must be reflected in the differentiated log probability.

For one actual optimizer step:

\[
\Delta\theta_{t,b}=\theta'_{t,b}-\theta_t,
\quad A_{t,b}=\langle g_V(\theta_t),\Delta\theta_{t,b}\rangle.
\]

Delta already includes the learning rate. Do not multiply A by it again. Use the ascent derivative of visual reward even though the trainer minimizes a loss.

Under local L-smoothness,

\[
\left|J_V(\theta+\Delta)-J_V(\theta)-\langle g_V,\Delta\rangle\right|
\le \frac L2\|\Delta\|^2.
\]

Positive alignment is therefore a first-order criterion, not an unconditional guarantee for a finite AdamW step. Greedy accuracy is a useful deployment endpoint, but its discontinuous decoder does not have this smooth reward-gradient interpretation. Report stochastic expected correctness as the primary local endpoint and greedy accuracy as a secondary endpoint and primary conventional benchmark summary.

### Visual estimator

Use independent on-policy responses per visual item and a leave-one-out reward baseline:

\[
\hat g_V=\frac1{nG}\sum_{i=1}^n\sum_{j=1}^G
(r_{ij}-\bar r_{i,-j})\nabla_\theta\log\pi_\theta(y_{ij}|z_i),\quad G\ge2.
\]

The baseline excludes the current response and is detached. Under independent draws and matched sampling/scoring, this estimates the gradient of expected correctness. Do not divide by group standard deviation or response length. Centering on the full group without the leave-one-out correction shrinks the expected gradient by (G-1)/G; this matters for calibration. A separate visual GRPO-style gradient is a useful ablation, explicitly labeled as a surrogate.

No visual KL, format reward, or length bonus belongs in this primary visual gradient. The text trainer can retain all released Praxis components: the scientific question is whether that composite text update improves visual correctness.

### Parameters and optimizer state

Compare identical, deduplicated parameter coordinates. For LoRA, both g_V and Delta must be in the same adapter parameterization; do not dot base-weight gradients with adapter updates. Report LoRA rank, scaling, initialization, target modules, and whether embeddings/head are frozen. Changing this coordinate system changes the restricted training problem.

For full-parameter confirmation, include every coordinate actually displaced. Text bypassing the vision tower is not sufficient proof of zero displacement if optimizer moments, weight decay, or existing zero gradients can move parameters. Audit actual changes. Frozen components can be omitted only after checking that their delta is zero.

Capture before/after inference-effective parameters, compute differences in FP32, and accumulate dot products robustly by tensor/shard. Casting an already low-precision dot result to FP64 does not recover precision. Save parameter hashes and optimizer/scheduler/RNG state. Candidate branches restore all of these, including optimizer moments and the reference policy, before each alternative. Gradient-only snapshots are insufficient.

## 4. Splits and information access

Use the released Praxis text training corpus as the primary candidate pool. Audit exact and near-duplicate situations across training data and visual evaluation using text similarity plus scene/source provenance and image hashes where available. Split by scene, source video, or scenario family; keep descriptions and variants of the same scene together. Do not put captions of final-test scenes into text training.

| Partition | Role | Allowed decisions |
|---|---|---|
| T_train | Text corpus and candidate batches | Text updates and selection |
| T_dev | Text-only validation | Text-based baselines, pilot settings |
| V_score | Labeled visual calibration examples | Gradient scoring and selected training/stopping policies |
| V_dev | Disjoint visual development examples | Noise/power calibration, horizon and threshold development |
| V_test | Sealed visual test examples | Final locked prediction and policy evaluation |

Starting allocation for a sufficiently large audited VIVA pool: 128 scene groups for V_score, 128 for V_dev, all remaining groups for V_test. Adjust this once from availability and pilot precision, then freeze it. If official train/development partitions provide enough calibration scenes, prefer those and preserve the official test partition. If a benchmark test set must be subdivided, call the outcome a custom held-out split and do not compare its score directly with the published full-test score.

V_test supports the locked local prediction evaluation and the final policy comparison. Once inspected, it cannot be used to tune the selector; later changes require a new test partition. Save all branch and policy checkpoints before unsealing outcomes where feasible.

Keep PCA-Bench and EgoNormia untouched as external transfer tests when selecting on VIVA. This tests whether VIVA-guided choices generalize; it does not establish alignment measurement within those other tasks. A later within-task alignment replication needs its own calibration/development partition. Preserve the source paper's frame-processing protocol for EgoNormia.

Two scientific settings must be labeled separately:

- **Observational text-only training:** visual measurements never influence the main training path. Diagnostic counterfactual branches are discarded.
- **Visual-probe-guided text-only updates:** visual labels affect which text is selected or when training stops. This is target adaptation through a labeled visual probe, even though images never provide training gradients.

Give all practical selectors the same allowed calibration data. Include image count, labels, model calls, and backward passes in their resource ledger. Do not describe the second setting as using no visual supervision.

## 5. Stage 0: measurement and hardware pilot

Your current GPU is described as an Ada 4000. Assuming RTX 4000 Ada or its SFF variant, NVIDIA specifies 20 GB VRAM. Verify the actual device and available CPU RAM before running. Start with **3B LoRA**, not full AdamW training. Use BF16 base weights initially; if quantization is needed, treat QLoRA as a separate condition and confirm key findings without it.

Provisional local settings: rank 16, alpha 32, zero LoRA dropout, adapters on language attention projections; one response per generation/backward microbatch; G=4 initially; gradient checkpointing; text batch of 8 prompts accumulated across responses. Select the pilot learning rate once on T_dev for stable KL/reward (a small predeclared grid such as 1e-6, 1e-5, 5e-5); do not assume the full-tuning rate is optimal for LoRA. Keep all arms identical afterward.

Start with the existing 256-token/256-image-token cap for plumbing only. Measure truncation and visual accuracy sensitivity on V_dev; expand the response cap, provisionally to 512 or 1024, and image resolution as needed before science runs. A short cap that truncates reasoning or omits final answers can change both alignment and transfer. None of these settings is yet verified to fit or run efficiently on the current card. Do not use the existing gen_batch=32/cache plan unchanged on 20 GB.

Required apparatus checks:

1. On a tiny enumerable policy, compare the expected-correctness estimator with an exact derivative; separately verify sequence masks, EOS, temperature, baseline and loss signs.
2. Verify a candidate branch restores weights, optimizer moments, scheduler, RNG, and reference state. Replaying the same branch must reproduce its delta within stated numerical tolerance. A skipped optimizer step must record delta zero.
3. Verify the dot product against a finite difference of a fixed differentiable objective. This checks implementation, not generalization. For actual expected reward, fresh rollout outcomes remain necessary.
4. Compare repeated independent visual probes at n=16/32/64 and G=4/8, expanding toward n=128 only if needed. Measure directional scalar A for fixed candidate deltas, not only gradient-gradient cosine.
5. Include zero-update and shuffled-score controls. A small direct visual-gradient step on calibration examples is an instrument control only, never a text-only baseline result. Its calibration improvement and its independent generalization are separate checks.
6. Compare exact adapter dot products against any compression over multiple sketch seeds and candidate directions. Avoid compression initially when adapter vectors are manageable.

Record measurement noise, candidate ranking reproducibility, rollout noise, truncation, memory and seconds per operation. For small true alignment, use an absolute error tolerance in reward units or ranking resolution; a CI excluding zero with relative half-width below 25% is not a universal go/no-go gate. A near-zero true signal should be allowed to remain unresolved.

Failure of the cross-modal sign test with a successful within-modality control is **not proof of zero alignment**. Use equivalence intervals tied to a practical effect threshold, or report inconclusive. Repeated generations reduce response noise; they do not create additional independent situations.

## 6. Experiment 1: independent prediction from controlled branches

### Design

First run an observational, random-text 3B LoRA trajectory. Development version: 1 seed, 3 checkpoint states (initial, middle, late), 8 randomly sampled candidate text batches per state. This gives 24 branches to size measurement and evaluation costs, not a confirmatory result.

Use 100 actual optimizer steps for the initial development trajectory (checkpoints 0/50/100). A provisional core parent budget is 200 steps (0/50/100/150/200); expand it only if development shows this does not cover meaningful learning. Lock the final budget before confirmatory runs. These are local LoRA study budgets, not claims of equivalence to the original full-training schedule.

Core confirmatory version: 3 independent parent training seeds, 5 checkpoint states at 0/25/50/75/100% of a fixed text budget, and 12 random candidate batches per state: **180 candidate branches**. Three seeds are a minimum replication design, not a promise of adequate power. Add seeds if the locked pilot power analysis requires them. Candidate batch sizes, sampling rules, horizons and full parent budget are fixed before testing.

At each checkpoint:

1. Estimate g_V on V_score before seeing branch outcomes.
2. Snapshot the full training state. For each candidate batch, restore the snapshot, generate its text rollouts and run one actual GRPO optimizer step. Record Delta and A. The candidate score is available before committing that branch to a continuing run, although computing it costs a tentative update.
3. Freeze branch IDs, scores, baseline predictors and checkpoints. Evaluate the common parent and every candidate child on the independent visual outcome split, first V_dev for development, then V_test for the locked test.
4. Use fresh visual responses independent of those used for g_V. Reuse visual items across parent/child for paired item analysis. Common random numbers may reduce variance if explicitly implemented, but matched seed integers alone do not guarantee useful coupling; retain an independent-rollout audit.
5. Discard diagnostic branches and continue the parent trajectory according to its original text schedule.

Primary outcome: paired change in sampled visual correctness. Start evaluation calibration at 16 responses per item and consider 32/64 if precision is inadequate. Reuse the common parent's estimate across candidates only with the induced correlation represented in analysis. Also evaluate greedy accuracy and answer transition counts.

The key scatterplot contains candidate scores from V_score against outcomes on V_test, faceted or centered within checkpoint. A plot using a frozen visual surrogate for both axes only verifies differentiation.

### Prediction horizons and local linearity

Keep three questions separate:

- One step: A predicts the effect of the exact candidate update.
- H steps: initial A predicts a short continuation. Candidate b defines a small text pool/curriculum from which H updates are drawn; the first update is scored. This is a prospective horizon test, not the one-step Taylor identity.
- Complete run: an initial or warm-up score predicts final transfer. This requires many independent runs/interventions; six published task/model values are descriptive context, not adequate validation.

If one-step fresh accuracy changes are below the noise floor, do not increase the learning rate until correlation appears. Either fund more evaluation or choose H from a predeclared development grid, such as 1/4/16 steps. Keep a smaller one-step estimator/linearity study and identify the primary H-step test as empirical short-horizon prediction.

On a small preselected subset, evaluate theta + c*Delta for c=0, 0.25, 0.5, 1, 2, with measured parameter changes. These are controlled perturbations, not ordinary resumed AdamW steps. Compare Delta J(c) with c*A and curvature residuals. Re-evaluation variance can overwhelm the smaller perturbations; report that rather than manufacturing a local linearity result.

### Baselines and inference

Compare A with: cosine(g_V, Delta); update norm; visual gradient norm; their norm product; raw text gradient inner product; visual-GRPO-surrogate alignment; text correctness, training loss/reward and advantage variance; prompt/response length; and semantic relevance to visual-calibration descriptions. Include a zero-change predictor and a checkpoint-only predictor.

Fit simple prediction/calibration models on development data and test them on held-out runs/checkpoints. Do not randomly split adjacent branches from the same state between training and test. Primary analysis: within-checkpoint association and held-out predictive error improvement over the locked strongest non-alignment baseline. Report Spearman correlation, slope/calibration, MAE, incremental held-out R-squared, and sign discrimination with uncertain outcomes handled explicitly. Do not count every sampled completion as an independent experimental replicate.

Use paired, hierarchical resampling over parent runs and scenario groups, preserving all shared-parent/candidate evaluations and checkpoint dependencies. Show per-seed results because uncertainty over only three runs is weakly estimated. Correct the small family of primary predictor comparisons, e.g. with Holm correction; label the remaining diagnostics exploratory.

For a per-item estimated parent/child change d_i, the item-level standard error is approximately sd(d_i)/sqrt(n_items). Estimate it on V_dev; simulate the planned paired/hierarchical analysis to choose sample counts for a predeclared meaningful effect (provisionally 1 percentage point for final selection). Branches and generations cannot repair inadequate scene diversity. VIVA alone may be underpowered for small final gains; report the detectable effect and use larger external data or more independently sourced scenes if necessary.

## 7. Experiment 2: training dynamics without adaptive selection

Reuse the independent random-text parent runs. Record text correctness and composite reward separately, visual sampled correctness, greedy accuracy, A, cosine, update norm, KL, clipping, response length and parsing/truncation. Use at least 10 evenly spaced measurement windows after development locks cadence; the five branching checkpoints are a subset.

Primary dynamics test: does the current score predict the **next** visual change after accounting for training progress and update size? Show raw per-run curves and a prospectively specified smoothing window. Do not search for an attractive zero crossing.

For selected contiguous short windows, recompute g_V at each actual step and compare sum_t A_t with observed visual change. Report residuals and sum_t ||Delta_t||^2. Sparse checkpoints using one stale g_V against a long displacement provide a coarse window predictor, not the gradient-flow identity. Do not multiply sampled scores by checkpoint spacing and call the result an exact integral.

An early-positive / late-zero / negative pattern is a hypothesis. Positive alignment can persist, greedy accuracy can plateau while sampled correctness improves, and noisy observations can remain inconclusive. A safe-horizon lower bound expiring does not imply subsequent harm. Estimated curvature and checkpoint secants are exploratory diagnostics, not certified stopping times.

## 8. Experiment 3: intervention through text selection

This is the strongest additional contribution. Use **batches as the primary selection unit**: actual AdamW/GRPO updates are state-dependent and are not sums of independently computed example updates. Per-example influence is an optional approximation that must be validated against batch effects.

### Controlled static selection

At a common checkpoint and optimizer state, score a locked pool of candidate text minibatches using V_score. A practical starting pool is 64 batches, each containing the same number of prompts. Use 16-batch subsets (25%) for each arm. Increase the pool or subset size only from development evidence that repeated training would otherwise be dominated by tiny-data memorization.

Core arms:

| Arm | Selection rule | Purpose |
|---|---|---|
| High-A | Highest scores | Proposed selection policy |
| Low-A | Lowest scores | Directional contrast; do not label negative unless measured negative |
| Random | Stratified random subset | Primary budget-matched baseline |
| Text-quality | Locked correctness/advantage-variance rule | Tests generic learnability as an explanation |
| Cosine | Highest normalized scores | Tests whether magnitude matters in selection |

Also include semantic relevance if affordable, and a shuffled-score selector as a calibration control. Match or stratify candidate composition on source/topic, text length and baseline difficulty for a mechanistic contrast. Report balance and support; do not force matches where high and low score distributions have no overlap. Run the unrestricted high-A vs random comparison as the practical policy test.

Use matched initialization, optimizer state, KL reference, prompt format and text update budget. Run each arm with at least three paired selection/training seeds. Average or replicate scores for the initial pool to limit selection of measurement noise; quantify independent probe split ranking and selected-score uncertainty. Do not optimize the selector on V_test.

Report both a short endpoint, where static rankings should be useful, and a longer fixed budget, where stale rankings may fail. Use fresh text rollouts during training. Log repetition counts and unique text coverage. Reusing cached rollouts for selection does not authorize treating those rollouts as perpetually on-policy during later training.

Provisional selection endpoints are 50 and 200 actual optimizer steps, locked after development and identical across arms. Measure how often each selected prompt repeats; if the small subset makes this a repeated-data experiment, label it accordingly and add a broader-pool confirmation before claiming general data efficiency.

Primary treatment contrast: high-A minus random final visual accuracy on V_test, with paired confidence intervals. High-A minus low-A establishes ordering; it is insufficient on its own to establish usefulness. High-A exceeding cosine and text-quality would support the specificity of the proposed score. A high/low contrast is evidence for the selection intervention; it does not alone prove alignment is the unique causal mediator.

### Online variant and baselines

After static selection passes, test a selector that periodically refreshes g_V and chooses among a fixed number of candidate batches from the current optimizer state. A provisional refresh every 10 optimizer steps and 4 candidates per decision must be tuned only on V_dev. Cache/checkpoint optimizer state sequentially; never maintain many full model copies on the local GPU.

Distinguish resource comparisons:

1. **Matched updates/tokens:** same training opportunity, explicitly showing all scoring overhead.
2. **Matched total compute:** allow random training to use the time/compute saved by avoiding probes and rejected candidates.

Actual generated lengths vary, so equal steps do not imply equal tokens. Use matched-step experiments for update causality and a token/cost curve for efficiency, with a declared budget stopping convention. Include scorer construction, captioning, backward passes, state restores, rejected candidates and development overhead. Separate one-time research diagnostics from deployment costs, and state any amortization assumption.

Include a validation-based candidate selector that evaluates each tentative child directly on V_score, at a comparable total scoring budget. This asks whether gradient scoring offers value beyond simply using visual validation. A conventional optimizer-aware gradient-influence baseline, adapted and labeled for RL batches, further strengthens novelty positioning.

## 9. Experiment 4: practical stopping and efficiency

Treat early stopping as secondary until dynamics demonstrate a useful prospective signal. It is cheaper to test initially by replaying locked stopping rules against saved observational trajectories. Charge each rule for the measurements it would actually request.

Baselines: full fixed budget; a shorter budget selected on V_dev; text-validation plateau; visual-validation plateau using the same labeled visual access; alignment stopping. An oracle best checkpoint on V_test is an analysis ceiling, never a deployable baseline.

A candidate rule: at each fixed monitoring time, score several independent candidate text batches; stop if the upper confidence bound for their mean predicted next-window gain falls below a locked practical threshold for two consecutive checks. Set the threshold in expected correctness units, provisionally 0.1 percentage points per chosen window, and calibrate it on development data. The average over random candidates estimates the continuation policy's next-step usefulness; one unlucky batch should not stop the run.

This is an empirical rule, not a theorem-backed guarantee. Repeated checks require a declared monitoring limit and appropriate sequential error control if inferential claims are made. If sign uncertainty remains high, do not silently interpret it as negative alignment. Predeclare maximum budget and fallback behavior.

Report final accuracy versus total GPU-hours, visual-label budget and generated tokens, plus the frequency of premature stopping. Claim efficiency only if alignment reaches matched quality with lower **total** cost or improves quality at matched total cost.

## 10. Secondary analyses

**Caption content:** use paired scenes from a dedicated training/development pool, never V_test. Cross task-critical information present/omitted with irrelevant detail short/long. Hold question/options and answer validity fixed; include length-matched relevant/irrelevant variants. Blindly audit factuality and answer leakage. This separates relevance from raw length better than sparse/medium/verbose alone. Keep caption generator/version/prompts fixed, and propagate all scene variants into one split. Test description -> A and description -> visual gain; do not assert statistical mediation without the additional assumptions/design it requires.

**Grounding dependence:** include question/options-only and shuffled-image evaluation on a prespecified subset. These reveal whether the gain depends on the actual image rather than answer priors. They do not alone localize a reasoning circuit.

**Modality diagnostics:** image/caption gradient similarity is a comparison of optimization directions. Reserve the term representation alignment for hidden-state or embedding measurements. Compare both with transfer alignment as alternative predictors rather than identifying them by definition.

**Reward and optimizer robustness:** after core results, repeat key contrasts with correctness-only text reward and compare raw gradient vs actual update across fresh/warm optimizer states. Report this as an extension to, not replacement for, the released-code profile.

**Scale and training regime:** add 7B only after 3B findings survive full-parameter confirmation. Agreement within Qwen 3B/7B supports limited scale robustness, not cross-architecture universality.

## 11. Resource plan and execution order

| Phase | Hardware and scope | Decision before expansion |
|---|---|---|
| A | Current card: data audit, parser/estimator/state checks, 3B LoRA memory test | Correct implementation and usable output lengths |
| B | Current card: 24-branch development pilot, n/G/horizon calibration | Measurable reward changes or an explicitly justified longer horizon |
| C | Current card if timings permit: 3B LoRA prediction and 5-arm selection | Independent predictive/selection results, including null results |
| D | Rented GPU(s): 3B full-parameter Praxis one-stage reproduction, then core branch and high/random/low confirmation | Effect survives actual Praxis training regime |
| E | Optional rental: 7B confirmation, external benchmarks, online efficiency | Materially broadens a supported claim |

An 80 GB GPU is a candidate for a full-parameter 3B feasibility run, not a guaranteed fit for released sequence lengths, rollout engine and reference-model memory. Multi-GPU/sharding or sequential rollout may still be necessary. Full AdamW training has weights, gradients, moments and potentially master weights before activations; the existing 20 GB card should not be the assumed full-training platform. CPU offloading also needs measured RAM and throughput.

Do not book a complete run matrix from README runtime estimates. Benchmark a representative generation group, visual gradient batch, actual GRPO update, state restore and evaluation batch. Estimate total time from measured components and add an explicit contingency.

For E1, with S seeds, C checkpoints, B candidates, N outcome items and K_eval responses, fresh evaluation costs approximately

`S * C * (B + 1) * N * K_eval` completions,

before perturbations or extra horizons. The core 3*5*12 design with N=512 and K_eval=16 requires about **1.60 million evaluation completions**. This is why the small pilot comes first and why three horizons should not all be expanded blindly. Parent reuse reduces cost but creates correlated outcome errors. Local adapter training fitting in memory does not make this evaluation workload cheap.

Rental priority: reliable full-parameter 3B confirmation over a broad 7B sweep. If funds are tight, shrink secondary analyses and the number of prediction horizons before sacrificing independent outcomes or the random-selection baseline. No rental was initiated for this design.

## 12. Paper figures, claims and failure criteria

| Figure/table | Evidence | Claim it can support |
|---|---|---|
| 1 | Held-out branch scatter, error bars, predictor comparison | Independent short-horizon transfer prediction |
| 2 | Per-run dynamics and forward-window prediction | Time dependence and staleness of the metric |
| 3 | High/random/low/text-quality/cosine outcomes | Utility of deliberate text selection |
| 4 | Accuracy vs total compute and probe labels | Net practical benefit, if present |
| Appendix | Measurement reliability, perturbation curves, captions, image controls | Estimator validity, limits and alternative explanations |

Negative findings remain informative: A may explain a tiny local surrogate but fail on fresh task reward; cosine may outperform a noisy dot product; selection may fail beyond a short horizon; high-A may beat low-A but not random; scoring costs may exceed training savings; alignment may never cross zero. Each rules out a distinct claim without disproving the directional derivative identity.

Success should be stated by effect size and uncertainty, not a desired-looking scatterplot. The main empirical claim requires independent predictive value beyond simple controls **and** a useful selection intervention. A paper limited to calibrated local prediction can still be honest, but its scope must be narrower.

## 13. Changes needed in the writeup and pilot

These are proposed changes; existing code and PDFs were not edited.

1. Make realized-update A the main metric. Keep gradient-gradient Lambda and cosine as explicitly different quantities; rankings are not equivalent across changing norms.
2. Separate expected stochastic reward from greedy benchmark accuracy. Use the former for the local Taylor prediction.
3. Implement a visual expected-reward estimator without group standardization and token-length normalization. Preserve the existing GRPO-style estimator as an ablation.
4. Remove any claim that positive correlation with the text gradient guarantees the same sign against an arbitrary visual gradient. Prompt reweighting can change that sign even before clipping.
5. Replace whole-run prediction from six published model/task numbers with controlled replicated branches and prospective interventions. Match the actual text corpus used for candidate updates.
6. Correct the safe-horizon interpretation: a lower bound expiring does not predict a peak, monotonic decay, or harm after the horizon. Likewise, a first-order approximation is not guaranteed to over- or undershoot; only the curvature-adjusted lower bound is one-sided under its assumptions.
7. Finite checkpoint secants do not certify a global/pathwise Lipschitz upper bound. Noise does not make the resulting estimated horizon automatically conservative.
8. An unbiased inner-product estimator does not imply an unbiased sign decision. Inflation of norm second moments alone does not prove a universal direction of bias for the expected cosine ratio.
9. Paired item noise has a covariance term of either sign, not necessarily positive. Cross-fitting addresses dependence under stated sampling assumptions; it does not remove GRPO estimator bias. A fixed sketch is not guaranteed to conservatively inflate uncertainty, and contiguous-block sketches do not automatically inherit an isotropic random-projection error formula. Validate against exact measurements and across independent sketches.
10. Replace sign-only noise gates with precision and practical-effect criteria; treat control failures and near-zero estimates with appropriate uncertainty.
11. Add actual optimizer delta logging, branch restoration and independent-outcome evaluation. Attach probes at real optimizer-step boundaries; one rollout iteration can contain several such steps. Preserve the actual microbatch loss weighting rather than assuming a single globally token-normalized loss.
12. Demote sufficiency, Fisher geometry, quadratic-horizon fits and representation mechanisms until data justify them. Keep useful mathematical material in an appendix with assumptions stated precisely.

## 14. Required records

For every run: model/repository revision or source-tree digest, dependency versions, hardware, full config, trainable-coordinate manifest, dataset hashes and exclusions, split IDs, training/rollout/probe/evaluation seeds, parser/reward versions, and checkpoint/optimizer/reference hashes.

For each branch: parent/run/checkpoint ID, candidate scene IDs, text tokens/rollouts/rewards, optimizer-step count, raw and applied gradient diagnostics, actual delta norm, exact or compressed A with uncertainty, cosine and norms, KL/clipping, parameter precision, wall time and peak memory.

For every outcome: scene ID, parent/child ID, sample seed, parsed answer, correctness, parse/truncation status and response length. Save selection memberships and selector version before final evaluation. Use machine-readable tables/JSONL plus a locked analysis plan; terminal logs and synthetic figures are insufficient evidence of VLM transfer.

## Sources inspected

- Local original: `2503.16965v3.pdf`, especially Table 1, Section 3.4, Appendix A and B.
- Local draft: `praxis_VLM (2).pdf`, especially Sections 3.4, 4 and 9.
- Local code: `Praxis-VLM-main/verl/workers/actor/dp_actor.py`, `verl/trainer/core_algos.py`, `verl/utils/reward_score/mcq.py`, `examples/config.yaml`, and the text-training launch script.
- Local pilot: `Praxis-Extension-main/task_0/README.md`, `config.yaml`, `src/gradient.py`, `src/rewards.py`, `src/alignment.py`, `src/sketch.py`, notebook metadata; synthetic context in `praxis_theory/README.md`.
- [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333): existing optimizer-aware gradient influence and data selection. The proposed novelty must be grounded in cross-modal RL updates, independent reward prediction and interventions, not gradient selection alone.
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783): related analysis of GRPO optimization bias. The estimator distinction here also follows directly from the score-function definition and inspected local normalization code.
- [NVIDIA RTX 4000 Ada specifications](https://www.nvidia.com/en-us/products/workstations/rtx-4000/): 20 GB GPU memory; the actual installed device and available memory remain to be checked.
