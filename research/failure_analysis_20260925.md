# Failure analysis: the text treatment never trained reasoning

25 September 2026. Diagnosis from existing archives only; no GPU work was run.

## Finding

Every inconclusive result since September 8 (H1, H4, the likelihood probe and the
+0.78-point independent baseline) measured text updates from a GRPO run whose
reasoning reward never switched on. The adapted baseline trained a letter-only
policy, and its visual evaluation then measured a reasoning policy. Null or
unresolved transfer is the expected outcome of that setup. It is not evidence
about alignment.

`research/scripts/audit_training_signal.py` on
`runs/ordinary-baseline-20260914/completed/metrics.jsonl` (exit 1):

| Quantity | Steps 1–8 | Steps 25–32 | Praxis paper, Fig. 7 (one-stage 7B) |
|---|---:|---:|---|
| Format reward | 0.000 | 0.000 (0/32 steps non-zero) | saturates early |
| Tag-count reward | 0.000 | 0.001 | saturates early |
| Length reward | 0.000 | 0.0003 | rises |
| Mean response tokens | 19.0 | 17.9 | rises, then stabilizes |
| Text accuracy | 0.707 | 0.734 | rises after format |

The composite reward `acc + 0.8·format + 0.4·tag + 0.5·length` therefore reduced
to letter accuracy. Within a five-sample group of near-deterministic letters, the
rewards are often identical, so the advantage and the gradient are zero.

## Causes

1. **Choice-only prompt.** Every released `problem` ends with
   `Now answer the question. Just output the choice:`. This held in 1,024/1,024
   training rows, and the suffix contradicts the `<think>/<answer>` system prompt.
   The pinned revision `9724bba` is the only published revision, and its columns
   are `messages, images, problem, answer`. The upstream launcher's
   `prompt_key=question` does not exist in this release, so the adaptation had no
   ready-made reasoning prompt.
2. **No cold start.** The paper's Stage 2 begins from a Stage-1 geometry-GRPO model
   that already emits the format. This project chose the one-stage variant with 3B
   Instruct and 32×5 rollouts per step. That policy never sampled a
   format-compliant response, so the strict format regex could not be learned by
   sampling.
3. **Train/evaluation mode mismatch.** The visual endpoints use a reasoning system
   prompt: `<think>` appears in 96.1% of initial and 99.6% of final responses, with a
   mean of about 120 tokens. Training rollouts averaged about 18 tokens with no tags.
   The text update optimized a behavior that the visual evaluation does not
   exercise.
4. **No treatment-activity gate.** The existing preflights verify hashes, budgets,
   storage and GPU idleness. None checks that training activates the reward terms
   that define the Praxis treatment. The suffix was documented on September 14 as
   a throughput limitation and the run continued, so all later candidates
   (H1/H4 parents, the lifecycle candidates, the step-16 warm parent) inherit it.

A separate, structural problem remains after the fix. 32-image panels give about
±10-point outcome intervals for 1–5-point contrasts, and 256–512 images still give
±2–4 points (see `selection_reassessment_20260915.md`). Fixing the treatment makes
effects plausibly larger. It does not make one-step candidate effects resolvable.

## Fix (implemented, CPU-tested)

- `transfer_alignment/training_signal.py`:
  - `apply_prompt_contract(problem, "reasoning")` replaces the choice-only suffix
    with an instruction showing the exact layout that the original `format_reward`
    accepts. `"released"` is byte-identical to the old prompt.
  - `rollout_signal(samples)` scores sampled rollouts for format rate and
    per-group reward variation.
  - `audit_metrics(rows)` fails a run whose final-window format reward is below 0.1
    or whose mean response is below 64 tokens.
  - Tests check exact parity with the original Praxis `format_reward` and
    `tag_count_reward`.
- `research/scripts/prepare_ordinary_baseline.py --prompt-contract reasoning`
  writes the new contract and records it in `audit.json`. The default `released`
  preserves the completed baseline's provenance.
- `research/scripts/audit_training_signal.py` is the post-run gate. On the
  completed baseline it fails with the table above.

## Next steps, in order

1. Prepare reasoning-contract data. Treat it as a new declared experiment, not an
   amendment of the completed baseline.
2. Run a rollout pre-check before any training run: about 64 prompts × 5 samples
   from the 3B Instruct model, about 10–15 GPU-minutes, scored with
   `rollout_signal`. Proceed only if formatted responses appear in most groups.
   If they do not, restore the paper's Stage-1 cold start rather than weakening the
   released reward.
3. Rerun the 32-step baseline and require `audit_training_signal.py` to pass.
   Longer responses invalidate the September 14 throughput rates, so re-time
   before budgeting.
4. Only then repeat the independent visual baseline. Start candidate or selection
   work only if that baseline shows a transfer effect worth predicting. The
   answer-label likelihood probe uses a choice-only system prompt, so it becomes a
   weaker surrogate under a reasoning policy and should be re-justified.

Existing results remain valid as records of the letter-only regime. They should
not be cited as tests of Praxis-style text reasoning transfer.
