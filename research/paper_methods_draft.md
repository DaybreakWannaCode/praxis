# Working methods text — development draft

Updated September 15. The sections beginning “Original expected-correctness methods”
retain the earlier protocol for provenance; they are not a frozen specification of
the next experiment. H1/H4 development is closed as inconclusive. Independent
prediction and beneficial data selection have not been established. The next
study's endpoint, complete compute allocation and storage plan remain unresolved.

## Current estimands and implementation

Three quantities must remain distinct in the paper:

1. **Expected generated-answer correctness**, $J_R(\theta)$, is the expectation
   of binary parsed correctness under a specified sampled response policy. This
   was the target of the original leave-one-out gradient estimator below.
2. **Normalized label-prefix log likelihood**, $J_L(\theta)$, is the deterministic
   differentiable surrogate implemented by `choice_probe.py`. It is measured with
   an image and question/options under a label-only system prompt, without a
   generated reasoning completion. For permitted, prefix-free tokenized labels
   $c\in C(z)$, define
   $$
   \ell_\theta(c;z)=\sum_{k=1}^{|c|}\log\pi_\theta(c_k\mid z,c_{<k}),\qquad
   q_\theta(c\mid z)=\frac{\exp\ell_\theta(c;z)}{
     \sum_{b\in C(z)}\exp\ell_\theta(b;z)},\qquad
   J_L(\theta)=\frac1n\sum_i\log q_\theta(a_i^\star\mid z_i).
   $$
   EOS is not appended. This normalizes label-prefix likelihoods over permitted
   choices; it is not the probability of a correct complete reasoning response.
   For single-token labels the full-vocabulary softmax normalizer cancels. The
   implementation sums multi-token prefix log probabilities without token-length
   averaging and performs the choice normalization in float64.
3. **Greedy extracted accuracy**, $G(\theta)$, is the empirical binary correctness
   of one deterministic completion per image under the frozen reasoning/answer
   prompt and parser. It is the endpoint of the completed ordinary baseline.
   It is neither the sampled expectation $J_R$ nor the surrogate $J_L$.

For the realized text displacement $u=\theta_{\mathrm{child}}-\theta_{\mathrm{parent}}$,
the current likelihood score is $A_L=\nabla J_L(\theta)^\top u$. The local expansion
of $J_L$ motivates prediction of a change in that *same* surrogate. It does not
mathematically imply a gain in $J_R$ or $G$, particularly because the prompt and
response policy differ. Greedy decoded correctness is discontinuous at decision
boundaries; do not assign it the likelihood gradient or apply the smooth Taylor
argument directly to its observed accuracy.

Independent-panel likelihood prediction would additionally test generalization
from the visual scoring panel to the outcome panel. Independent generated-answer
prediction would test a further cross-objective relationship. The latter was not
included in the initial intervention-only cost forecast; see
[corrected scope and cost](selection_reassessment_20260915.md). Choosing the
surrogate endpoint would need an explicit change in the paper's claimed estimand,
not merely different notation for the original expected-correctness claim.

## Completed evidence and claim limits

The adapted ordinary baseline used 1,024 text prompts, five rollouts per prompt,
32 updates and one seed. Its independent development accuracy changed from
201/256 to 203/256: +0.78 percentage points, paired image-bootstrap 95% interval
[−1.95,+3.91]. This is unresolved and conditional on the panel/trajectory, not a
seed-robust transfer result. See [baseline result](independent_baseline_result_20260915.md)
and [figure caption](independent_baseline_figure_20260915.md).

Two original-worker candidate lifecycles verified displacement construction,
likelihood scoring and bounded temporary retention. Their same-probe local
surrogate agreement is numerical calibration, not independent prediction or a
selection intervention. Original-worker loaded model/Adam/scheduler/RNG equality
and separate dataloader continuation passed; these checks do not prove arbitrary
post-restore rollout replay. See [restoration evidence](restore_only_execution_20260915.md).

The proposed smaller selection comparison is Alignment versus Random with three
paired training seeds, a shared static pool of 64 candidate batches and 32 selected
batches per arm. It has not launched or been frozen as the complete paper design.
A two-arm result cannot establish superiority over cosine selection. Visual
calibration is supervised adaptation data even though subsequent optimizer
updates use text only. Equal-update gains are not equal-total-compute gains.

## Original expected-correctness methods (historical protocol)

The following text describes the original RLOO/precision protocol and preserves
its mathematical estimator and branch controls. References to queued precision
checks, an H4 fallback, or three selection arms below are historical proposals;
the current status and unresolved choices above take precedence.

## Question and objectives

We study when an update induced by text-only reinforcement learning improves a
vision-language model's expected correctness on image-grounded questions. Text
updates follow the pinned released Praxis implementation, including its composite
reward, group-relative advantages, clipping, KL term, optimizer and scheduler.
Visual correctness is a separate objective: a completion receives one if a fixed
parser extracts the correct benchmark choice, and zero otherwise. The visual
probe and independent visual outcome evaluation use the same extraction rule and
generation configuration. Format and length bonuses are excluded from both.

For a parent parameter state θ and a realized text update u, define transfer
alignment A = ∇J_V(θ)ᵀu. Under differentiability and a sufficiently small update,
Taylor expansion gives J_V(θ+u) − J_V(θ) = A + a higher-order remainder. The sign of
A alone does not guarantee a finite-step gain when that remainder is material.
For a multi-step window, u denotes the complete displacement from the same parent
to the window endpoint. We do not substitute the first step's displacement.

The inner product is the primary predictor. Cosine, update norm, text reward and
response length are comparison measurements. Alignment is local and dependent on
the chosen parameter coordinates and visual policy. Numerical precision and
sampling implementation must therefore be recorded; an automatic-differentiation
gradient through finite-precision computation is an approximation, not a proof
that tiny master-weight updates change sampled outputs.

## Visual gradient estimator

For each probe image, draw K ≥ 2 independent completions at the frozen parent.
The weight for completion j is its binary reward minus the mean reward of the
other K−1 completions for that image. Multiply that detached weight by the
gradient of the sequence log probability, summed over generated response tokens
including EOS when emitted. Average over completions and images. There is no
reward-standard-deviation normalization or response-length averaging. Images
with constant sampled reward produce a zero estimated contribution and remain
in the denominator. Record this frequency rather than silently dropping them.

Writing z_i for a probe question and image, the estimator is

$$
\widehat g_V = \frac{1}{nK}\sum_{i=1}^{n}\sum_{j=1}^{K}
\left(r_{ij}-\frac{1}{K-1}\sum_{\ell\ne j}r_{i\ell}\right)
\nabla_\theta\log\pi_\theta(y_{ij}\mid z_i).
$$

Conditional on the image and frozen policy, the leave-one-out baseline is
independent of response j. Its expected product with the response score gradient
is zero because the expected score gradient is zero. Under the usual
interchange-of-gradient-and-expectation conditions, this gives an unbiased
estimator of the finite-panel expected-correctness gradient. This statement is
about the ideal policy-gradient estimator; finite-precision automatic
differentiation and limited Monte Carlo draws remain practical approximations.
No derivative is taken through the binary parser or reward.

Independent response repeats estimate Monte Carlo uncertainty for fixed probe
images and fixed candidate displacements. They do not by themselves estimate
generalization uncertainty across images, parents or tasks. Preserve per-image
projections so those sources of variation can be assessed separately.

The gradient of a finite probe panel is not exactly the gradient of a different
held-out panel. Transfer prediction therefore also tests generalization across
images, beyond the local Taylor identity. Repeating response draws reduces
sampling error conditional on the probe images; it cannot eliminate a systematic
probe-to-outcome mismatch. The development image pool is availability-limited
and does not establish performance across the entire benchmark.

## Controlled branches and intervention

Compare candidate text updates from the same warm parent, including its Adam
moments, scheduler and captured worker RNG/state. Start each production branch in
a fresh trainer/rollout process. A fixed candidate was reconstructed identically
across fresh processes; this is narrower than a general guarantee of arbitrary
driver or vLLM engine serialization. Ensure the next learning rate is nonzero.
Audit fixed-input observer parity
against the uninstrumented original update. Map visual gradients and realized
updates into the same canonical parameters, excluding padding and duplicate tied
aliases. No image-gradient optimizer step is applied.

Predictive outcomes use independent development images; final-test data remain
untouched during development. Parent/null controls quantify sampling noise and
shared-parent effects. Within-parent comparisons are primary: branches sharing
a parent and images are not independent observations. The bounded development
check will lock the horizon and affordable response budget before confirmatory
runs, even if the result is unresolved or null.

For the development precision check, independent response repeats produce paired
alignment contrasts for each candidate pair. Outcome comparisons share random
seeds within an image/completion pair and reset the seed for each completion.
This coupling can reduce variance without changing either policy's marginal
expected correctness. A separate unchanged-parent control uses independent
seeds. Conservative fixed-panel outcome intervals and approximate score-repeat
intervals are reported separately from image variation. The frozen development
decision uses a two-percentage-point practical resolution and does not require
positive transfer or agreement between alignment and outcomes. Its exact budget,
limitations and H=4 fallback are specified in `production_precision_lock.md`.

The subsequent selection experiment compares alignment, random and cosine
selection with matched training-update budgets and recorded token/scoring costs.
Because selection uses a labeled visual probe, this is **visual-guided selection
of text-only updates**. It must not be described as learning without any visual
supervision. Higher final accuracy alone does not establish compute efficiency;
candidate-update and visual-scoring costs belong in that comparison.

## Reporting limits and supporting analyses

Unconditional extracted correctness is primary. Parse success, correctness
conditional on parsing and truncation are separate diagnostics. An improvement
in producing usable final answers does not alone demonstrate stronger semantic
reasoning. Report all released text-reward components and group variation to
distinguish correctness-driven from formatting-driven signals.

Training dynamics and retrospective stopping diagnostics are supporting analyses
using preserved checkpoints. An early-stopping method would need its own frozen
rule and independent validation; the current study does not yet make that claim.
Related-work attribution and the exact final sample sizes remain to be completed
before submission. No result placeholders in this draft constitute evidence.
