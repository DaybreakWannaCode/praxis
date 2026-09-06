# Task 0 — the alignment measurement apparatus

Measure the cross-modal gradient alignment

$$\Lambda = \langle \nabla J_v,\ \nabla J_t\rangle, \qquad \alpha = \frac{\Lambda}{\|\nabla J_v\|\,\|\nabla J_t\|}$$

at a **fixed** Qwen2.5-VL checkpoint, over the language-backbone parameters only, and
characterise the measurement noise well enough to know whether Task 1 could ever locate a
sign or a zero-crossing.

**No training. No optimizer step.** Load a frozen model, compute two gradient vectors on
small data, dot them, and find out how noisy that number is.

---

## Go / no-go

Task 0 **passes** if some $(n, G)$ feasible in Colab gives all three:

| Criterion | Threshold | Where |
|---|---|---|
| sign of $\hat\Lambda$ stable across resamples | ≥ 95% | `sign_stability.csv` |
| bootstrap CI on $\hat\Lambda$ excludes zero | — | `noise_floor.csv` |
| CI half-width as a fraction of $\lvert\hat\Lambda\rvert$ | < 25% | `noise_floor.csv` |

It **fails** if the sign flips randomly even at large $n, G$ — then Task 1's zero-crossing
is unlocatable and the remedies are more samples, more minibatch averaging, or measuring
the cosine on independent halves (the last is already built in, see *Cross-fitting*).

### Which interval the criterion is judged on

Two intervals get reported and they are not interchangeable.

- **`ci_lo` / `ci_hi`** — the 2.5/97.5 percentiles of the *sampling distribution* of
  $\hat\Lambda$. This is the interval one measurement at that $(n,G)$ lands in. Its
  half-width is ≈ 1.96 std and does **not** shrink with more draws. **This is the gate.**
- **`se_of_mean`, `mean_ci_*`** — describes the *mean over draws* and does shrink like
  $1/\sqrt{R}$.

Judging acceptance on the second would be gameable: take more draws, pass the test, learn
nothing about a single measurement. The criterion is about whether one measurement at a
checkpoint pins $\Lambda$ down, so it is judged on the first.

---

## Running it

```bash
cd task_0
pip install -r requirements.txt

python -m src.run_task0 all --smoke     # full pipeline, ~15 min, validates the plumbing
python -m src.run_task0 all --auto-m    # the real study, ~3-4 GPU-hours on an A100
```

Or open `run_task0.ipynb` in Colab (A100 + high-RAM runtime).

Individual stages, in dependency order:

| Stage | What it does | Cost |
|---|---|---|
| `sanity` | correctness gate — finite differences, masking, sampling distribution | ~2 min |
| `data` | download VIVA, verify images decode, build the paired pool | ~10 min |
| `anchor` | exact full-backbone $\Lambda$; measures true $\lvert\alpha\rvert$ | ~40 min |
| `cache` | K generations + K sketched gradients per (item, modality) — **resumable** | ~1.5 h |
| `acid` | bootstrap-vs-fresh variance test | ~30 min |
| `sweep` | the $(n,G)$ grid, CIs, sign stability, figures, verdict | seconds |

`sweep` needs no GPU model, so it is the cheap one to iterate on. `cache` writes progress
per (modality, item) and resumes after a Colab disconnect.

---

## What is actually built

```
task_0/
  config.yaml            every knob; results are a function of this file plus the seeds
  src/
    load_model.py        frozen Qwen2.5-VL, backbone isolation (tied-weight dedupe)
    gradient.py          the policy-gradient extractor — image OR text, one code path
    sketch.py            unbiased O(d) gradient compression
    cache.py             per-(item, sample) gradient cache, resumable
    alignment.py         Lambda, alpha, cross-fitting, split-half cosine
    noise_study.py       resample loop, sweeps, CIs, acid test, verdict
    sanity_checks.py     the correctness gate
    rewards.py           answer parsing + verl-faithful GRPO advantage
    data_prep.py         VIVA download and the paired calibration slice
  results/
    noise_floor.csv      std(Lambda), CIs, controls and reward diagnostics vs (n, G)
    sign_stability.csv   sign-agreement fraction vs (n, G)
    task0_results.json   everything, including the config hash and the verdict
    raw_draws.npz        every individual draw, for re-analysis without a rebuild
    sanity_checks.json   the correctness gate's result, per check
    exact_anchor.json    full-backbone Lambda + sketch fidelity + cache trade table
    acid_test.json       bootstrap-vs-fresh variance comparison
    timings.json         measured wall-clock per stage, for budgeting Task 1
    figures/             noise_floor.png, anchor_distribution.png
  run_task0.ipynb        Colab entry point
```

## Reading the result: the interpretive fork

A failed $\Lambda$ means two opposite things, and telling them apart is what the
**within-modality positive control** is for. $\langle \hat g_v^A, \hat g_v^B\rangle$ over
disjoint item halves is an unbiased estimate of $\|\nabla J_v\|^2$, so its true sign is
*known* to be positive. If the apparatus cannot resolve that, it certainly cannot resolve
a cross-modal inner product.

| $\Lambda$ | control | reading |
|---|---|---|
| passes | passes | measurement works; use the cheapest passing $(n,G)$ in Task 1 |
| fails | **passes** | alignment is genuinely ≈ 0. More samples will not rescue it — this is a result, not a failure |
| fails | fails | instrument is blind; the $\Lambda$ null is uninformative. Raise $G$/$n$ first |
| passes | fails | internally inconsistent (the control is the *easier* measurement) — suspect a bug |

The verdict also prints a **variance decomposition**. Fitting
$\mathrm{Var}(\hat\Lambda) \approx (A + B/G)/n$ over the grid gives the crossover
$G^\* = B/A$: below it, more samples per item buy more than more items; above it, the
reverse. That turns "it's too noisy" into a specific next action.

Supporting diagnostics per cell, which explain *why* a $\Lambda$ is small:
`*_mean_reward` (is the model at ceiling or floor?), `*_degenerate_frac` (fraction of
groups where every answer scored the same, contributing zero gradient),
`*_mean_abs_adv` and `*_nonzero_adv_frac` (how much signal each group carries), and
`*_parse_ok_frac` (are completions even being scored?).

---

## Three design decisions you should know about

### 1. The literal sweep is computationally impossible; the cache makes it exact anyway

The grid is 4 values of $n$ × 3 of $G$ × 20–50 resamples × 2 modalities. Summing $n\cdot G$
over the grid gives 21,000 (item, sample) pairs *per resample set*; at 30 resamples that is
**~1.26M generate-plus-backward passes** — hundreds of GPU-hours, not the few the budget
assumes.

The way out is that the advantage $\hat A_{ij}$ is a **scalar**. Every estimate in the
entire sweep is a linear reweighting of the same per-sample gradients:

$$\hat g = \text{scale}\cdot\sum_{ij}\ \text{coeff}(\hat A_{ij}, L_{ij})\ v_{ij},
\qquad v_{ij}=\nabla_\theta \sum_t \log \pi_\theta(a_{ijt})$$

So each $v_{ij}$ is computed **once** (2 × `n_pool` × K ≈ 25,600 backward passes, ~1.5 h)
and every $(n, G)$, every resample, every bootstrap replicate, and *both* GRPO conventions
become arithmetic on the cache. `sanity_checks.check_cache_linearity` asserts that the
cached recombination equals direct backprop to float precision — if that ever fails, the
sweep is measuring something the anchor is not.

$v_{ij}$ is 3.09e9 floats, so caching 25,600 of them raw is 200 TB. They are compressed by
a **block-sign sketch**: fix random signs $\sigma_k \in \{\pm1\}$ and sum contiguous blocks,
$(Sx)_b = \sum_{k \in b} \sigma_k x_k$. Then $\mathbb{E}\langle Sx, Sy\rangle = \langle x,y\rangle$
*exactly*, with variance $\sim \|x\|^2\|y\|^2/m$ — so the relative sketch error on $\Lambda$
is $\approx 1/(|\alpha|\sqrt{m})$. That is why the exact anchor runs **first**: it measures
the true $|\alpha|$, and `--auto-m` resizes $m$ from it. At $m = 2^{18}$ the cache is 13.4 GB.

The sketch is validated two ways: `check_sketch` verifies unbiasedness synthetically, and
the exact anchor projects *the same* full-precision vectors through the sketch, so the
error it reports is pure sketch error with no sampling noise mixed in.

Because the signs are fixed, sketch error is not fresh noise per resample — but different
resamples give different $\hat g$, so it shows up as extra spread. It therefore **inflates**
the reported noise floor rather than hiding it, which is the safe direction to be wrong in.

### 2. Cross-fitting, because the plug-in numbers are biased

The brief's $\hat\Lambda = \langle\hat g_v, \hat g_t\rangle$ uses the same items for both
modalities. Per-item noise in the two channels is correlated (same situation drives both),
so

$$\mathbb{E}\langle \hat g_v, \hat g_t\rangle = \langle \nabla J_v, \nabla J_t\rangle + \tfrac{1}{n}\mathbb{E}\langle \delta_v, \delta_t\rangle$$

— a positive $O(1/n)$ inflation. The pool is split into **disjoint halves** A and B and the
estimate symmetrised, $\hat\Lambda_{cf} = \tfrac12(\langle \hat g_v^A, \hat g_t^B\rangle + \langle \hat g_v^B, \hat g_t^A\rangle)$,
which is unbiased. Both are reported; `study.estimator` picks the headline (default
`crossfit`) and their gap *is* the measurement of the bias.

The same problem is worse for the cosine: $\mathbb{E}\|\hat g\|^2 = \|\nabla J\|^2 + \mathbb{E}\|\text{noise}\|^2$,
and in 3.09e9 dimensions the noise term dominates, so the plug-in cosine is biased hard
**toward zero**. The independent halves fix that too:
$\widehat{\|\nabla J_v\|^2} = \langle \hat g_v^A, \hat g_v^B\rangle$, giving
`alpha_shc`. Expect `alpha_plugin` $\ll$ `alpha_shc`.

Note that at cross-fit "n", each gradient uses $n/2$ items and $n$ items are touched in total.

### 3. Group bootstrap, and the acid test that guards it

Answers are drawn **with replacement** from a pool of K=32 cached generations per (item,
modality). Drawing G of K *without* replacement would understate answer-sampling variance
by $1-\frac{G-1}{K-1}$ — about 2× at G=16, K=32. Bootstrapping matches infinite-population
variance to first order.

That is a claim, so it is tested. `run_acid_test` compares the bootstrap's std against
**genuinely fresh generations** at the same $(n, G)$ with the **items held fixed** on both
sides, so only the answer source varies. It reports the variance ratio with a 95% CI. If
that CI excludes 1.0, set `study.group_resample=subsample_fpc` and re-run the sweep — the
finite-population fallback is implemented.

Caveat worth stating plainly: with `n_fresh=10` the ratio CI is roughly $[0.47F, 3.3F]$.
That is wide. It will catch a 2× discrepancy, not a 20% one. Raise `acid_test.n_fresh` if
you want a tighter bound, at ~4 min per extra rep.

On the epsilon floor: `adv_eps` (1e-6) is shared with Task 1 training. Worth knowing that
with 0/1 rewards it is **inert** — the ddof=1 group std of a binary group with $k$ of $G$
correct is $\sqrt{k(G-k)/(G(G-1))}$, so the smallest *nonzero* std is $1/\sqrt{G}$ (0.25 at
G=16). Near-degenerate groups do not exist here. The only degenerate case is $k \in \{0, G\}$,
where every $r - \bar r$ is exactly zero and so is every advantage. So bootstrap-induced
degeneracy contributes zero gradient rather than a spurious huge one. `adv_std_floor` is
exposed only so sensitivity to a larger floor can be measured; leave it at 0.0.

---

## The four traps in the gradient extractor

The score-function extraction is where this goes silently wrong. Each trap has a check.

| Trap | Consequence | Guard |
|---|---|---|
| Sampling from a truncated distribution while differentiating the full softmax. Qwen ships `top_k=20, top_p=0.8, temperature=0.7` in its generation config. | $\hat g$ **biased**, not just noisy | `_generation_config` builds from scratch and pins top_k=0/top_p=1.0/rep_penalty=1.0; `check_sampling_matches_policy` compares the realised next-token histogram to `softmax(logits)`, and specifically checks the mass outside the model's top-k, which a stray `top_k` would drive to exactly zero |
| Prompt tokens leaking into the gradient | measures the wrong objective | logits sliced at `prompt_len-1 : -1`; `check_logprob_alignment` compares against a naive position-by-position reference *and* verifies an off-by-one would be detected |
| Padding counted as generated tokens | length-dependent junk in every $\Lambda$ | sequences are **trimmed** at the first EOS (EOS kept — emitting it is part of the action), so nothing downstream must remember to mask; `check_response_lengths` tests hand-built cases |
| Advantage applied with the wrong sign or granularity | every $\Lambda$ silently corrupted | `check_finite_difference`: step $\theta \to \theta + \epsilon g$ and verify $\log\pi$ rises by $\epsilon\|g\|^2$; `check_advantage_sign`: A=−1 must lower it |

The finite-difference check auto-sizes $\epsilon$ to the parameter dtype and **asserts the
weights actually moved** — in bf16 a perturbation under ~0.4% of a weight rounds straight
back and the test would otherwise pass vacuously. Run `sanity --fp32` for the strict form.

One more: Qwen2.5-3B has `tie_word_embeddings=True`, so `lm_head.weight` **is**
`embed_tokens.weight`. Walking `named_parameters()` naively would put ~311M coordinates
into the flat vector twice, double-weighting 10% of every inner product. `select_backbone`
dedupes by identity and reports what was tied.

---

## Data

VIVA (Hu et al., EMNLP 2024): 1,062 images of real situations with human-annotated correct
actions. Annotations come from HuggingFace (`zhehuderek/VIVA_Benchmark_EMNLP24`); **images
are not hosted** and are fetched from their original third-party `image_url`s, which rot.
`build_calibration` downloads what it can, verifies each file actually decodes (dead links
return HTML error pages and 1×1 placeholders with HTTP 200), and builds the pool only from
survivors, reporting the yield. If too few survive, the repo's
[Google Drive mirror](https://drive.google.com/drive/folders/1eFLdVoRdw3kNXdi-zkmf_NDyxEbbpGY5)
can be unpacked into `data/VIVA_images/` and the stage re-run with `--no-download`.

**The text channel is VIVA's own `situation_description`** — e.g. *"A person improperly
disposes of a plastic bottle by throwing it out of a car window onto a scenic rural road."*
This is a change from the self-captioning plan, and a strict improvement: it is fixed
benchmark data, so it is independent of whichever checkpoint is being measured. That
matters in Task 1, where the model moves between checkpoints and the text channel must not
move with it. It also does not name or hint at the correct action (VIVA's `reason` field
does, and is deliberately unused).

`text_channel: self_caption` is still available and regenerates captions with
`data.caption_model` (greedy, cached to disk). Comparing the two answers a real question —
does $\Lambda$ depend on where the text channel came from? — but it is not the default,
because a caption written by the model under test shares that model's visual encoding and
can inflate alignment.

---

## Known limitations

- **Sketch fidelity depends on the true cosine**, and this is the most likely thing to bite.
  Relative sketch error is $1/(|\alpha|\sqrt m)$, so at $m = 2^{18}$:

  | measured $\lvert\alpha\rvert$ | sketch error | verdict |
  |---|---|---|
  | 0.20 | 0.8% | fine |
  | 0.05 | 3.9% | fine |
  | 0.01 | 19.5% | marginal — comparable to the 25% acceptance threshold |
  | 0.005 | 39% | sweep unusable as an absolute measurement |

  The $m$ needed for 5% error at $|\alpha| = 0.01$ is 16.8M, which no affordable cache
  reaches. So the anchor stage prints an explicit trade table (`describe_cache_plan`):
  halving `n_pool` or `K` doubles the affordable $m$, at the cost of the top of the $n$
  sweep or of bootstrap pool depth. If even that is not enough, the honest fallback is to
  **read the sweep as a relative measurement** — how noise scales with $n$ and $G$, which
  the sketch error does not distort much — and quote the exact anchor for the absolute
  $\Lambda$. Since sketch error inflates the spread, a sweep that passes despite it has
  passed conservatively.
- **The acid test's power is limited by `n_fresh`** (see above).
- **Bootstrap conditions on the K=32 pool** per item. The group-mean variance it produces is
  right to first order, but it treats each pool's empirical answer distribution as the truth;
  the residual is $O(1/K)$ and is exactly what the acid test is there to bound.
- **bf16 gradients.** Within an item, G terms accumulate in bf16 before the fp32 sink write.
  That is well inside the sampling noise, but it is an approximation. `model.dtype: float32`
  makes it exact at 2× the memory.
- **Rule-based reward only.** No GPT-4o answer-parsing fallback. `parsed_ok` is tracked and
  reported per modality; a high parse-failure rate means many degenerate groups, hence
  small gradients, and would show up as an inflated `*_degenerate_frac`.
- **One checkpoint.** Task 0 deliberately says nothing about how $\Lambda$ moves during
  training. That is Task 1, and the proxy target (CI half-width < 25% of $|\Lambda|$) stands
  in for a between-checkpoint change nobody has measured yet.
