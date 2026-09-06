"""Answer parsing and the GRPO group advantage.

The reward here must be the reward Task 1 trains on, or Lambda measures a different
objective than the one that later moves. Two pieces:

  * `score_completion` — parse a free-form completion down to an option letter and
    compare to the gold label. Binary, 0/1, as the brief specifies.
  * `group_advantages` — verl/trainer/core_algos.py::compute_grpo_outcome_advantage,
    reduced to one group of scalar rewards.

A note on `adv_eps` and degenerate groups, since this is where the bootstrap could have
bitten us. With binary rewards and k correct out of G, the ddof=1 group std is

    sd = sqrt( k (G-k) / (G (G-1)) ),

so the smallest *nonzero* std is 1/sqrt(G) — 0.25 at G=16, 0.35 at G=8. The verl epsilon
(1e-6) is therefore inert for 0/1 rewards: it can never be the thing that rescues a
near-degenerate group, because near-degenerate groups do not exist here. The only
degenerate case is k in {0, G}, where sd is exactly 0 *and* every r - mean is exactly 0,
so every advantage is 0 and the group contributes no gradient at all. That is the correct
behaviour and it is what verl does. `adv_std_floor` exists only so the sensitivity of
Lambda to a larger floor can be measured; leave it at 0.0 to match training.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

# ---------------------------------------------------------------- answer extraction

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_OPEN = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)
_THINK_END = re.compile(r"</think>", re.IGNORECASE)

#  "D."  "(D)"  "**D**"  "D)"  or a bare "D" on its own.  Punctuation after the letter is
#  required unless the letter is the whole span, which is what stops "A person walks..."
#  and "I would call 911" from being read as choices A and I.
_LEADING_LABEL = re.compile(r"^[\s*_`\[(]*([A-Za-z])[\s]*[).:,\-\]*_`]+")
_LONE_LABEL = re.compile(r"^[\s*_`\[(]*([A-Za-z])[\s*_`\])]*$")
_PHRASED = re.compile(
    r"(?:answer|option|choice|action|select(?:ed)?|pick|chose|choose)\b[^A-Za-z0-9]{0,12}"
    r"(?:is\b[^A-Za-z0-9]{0,4})?([A-Za-z])\b",
    re.IGNORECASE,
)
_OPTION_PREFIX = re.compile(r"^\s*[\(\[]?([A-Za-z])[\).:\]]\s*")


def extract_answer_span(text: str) -> str:
    """Narrow a completion down to the part that is supposed to hold the choice.

    Prefers <answer>...</answer>; falls back to whatever follows </think>; falls back to
    the whole completion. An unclosed <answer> still counts — truncation at
    max_new_tokens is common and we would rather score it than throw it away.
    """
    if text is None:
        return ""
    m = _ANSWER_TAG.search(text)
    if m:
        return m.group(1).strip()
    m = _ANSWER_TAG_OPEN.search(text)
    if m:
        return m.group(1).strip()
    parts = _THINK_END.split(text)
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


def option_labels(action_list: Sequence[str]) -> list[str]:
    """Labels for the options, taken from the option text when it carries one.

    VIVA ships options already prefixed ("A. Pick up the items ..."), so we read the
    label off rather than assuming positional A/B/C — that keeps us aligned with the
    `answer` field even if an item's options are out of order.
    """
    labels: list[str] = []
    for i, opt in enumerate(action_list):
        m = _OPTION_PREFIX.match(opt or "")
        labels.append(m.group(1).upper() if m else chr(ord("A") + i))
    return labels


def option_bodies(action_list: Sequence[str]) -> list[str]:
    """Option text with any leading label stripped, for substring matching."""
    return [_OPTION_PREFIX.sub("", opt or "").strip().rstrip(".").strip() for opt in action_list]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _fuzzy_option(span: str, labels: Sequence[str], bodies: Sequence[str]) -> str | None:
    """Match a paraphrase of an option body by token coverage.

    Models routinely restate the chosen action instead of naming its letter ("remind them
    politely not to do so" for "D. Politely remind them not to do so."). Coverage of the
    *option's* tokens by the span is the right direction to measure — the span is usually
    longer — and a margin over the runner-up is required so that options sharing common
    words cannot both look like matches.
    """
    span_tok = _tokens(span)
    if not span_tok:
        return None
    scored: list[tuple[float, str]] = []
    for lab, body in zip(labels, bodies):
        bt = _tokens(body)
        if len(bt) < 3:
            continue
        scored.append((len(bt & span_tok) / len(bt), lab))
    if not scored:
        return None
    scored.sort(reverse=True)
    best, lab = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    return lab if best >= 0.8 and best - runner >= 0.2 else None


def parse_choice(completion: str, action_list: Sequence[str]) -> str | None:
    """Map a completion to one of the option labels, or None if it cannot be read.

    None is a real outcome, not an error: it becomes reward 0 and is counted separately
    (`parsed_ok`) so a high parse-failure rate surfaces as a diagnostic instead of
    silently flattening every group to zero advantage and zero gradient.

    The cascade is ordered most-explicit first, and deliberately refuses to guess at the
    end. The tempting last rule — "if exactly one option letter appears anywhere, take it"
    — reads "A person is littering, so the best response is to remind them politely not to
    do so" as choice A, because the indefinite article is a valid option label. A wrong
    label is far worse than an unparsed one: it flips a reward, which flips an advantage,
    which contributes a gradient term pointing the wrong way. So the bare-letter rule is
    restricted to spans short enough to be an answer rather than a sentence.
    """
    span = extract_answer_span(completion)
    span = span.strip().strip("*").strip()
    if not span:
        return None

    labels = option_labels(action_list)
    label_set = set(labels)
    bodies = option_bodies(action_list)

    m = _LONE_LABEL.match(span)
    if m and m.group(1).upper() in label_set:
        return m.group(1).upper()

    m = _LEADING_LABEL.match(span)
    if m and m.group(1).upper() in label_set:
        return m.group(1).upper()

    m = _PHRASED.search(span)
    if m and m.group(1).upper() in label_set:
        return m.group(1).upper()

    lowered = span.lower()
    hits = [lab for lab, body in zip(labels, bodies)
            if body and len(body) >= 8 and body.lower() in lowered]
    if len(hits) == 1:
        return hits[0]

    fuzzy = _fuzzy_option(span, labels, bodies)
    if fuzzy is not None:
        return fuzzy

    # Bare letter, only in a span too short to be prose (<= 6 words) and only if written
    # UPPERCASE. Both guards are load-bearing: without the length bound "A person is
    # littering" reads as A, and without the case bound a truncated "...without a close"
    # reads as A off the indefinite article.
    if len(span.split()) <= 6:
        found = sorted(set(re.findall(r"\b([A-Z])\b", span)) & label_set)
        if len(found) == 1:
            return found[0]
    return None


def score_completion(completion: str, gold: str, action_list: Sequence[str]) -> tuple[float, bool]:
    """Return (reward in {0,1}, parsed_ok)."""
    pred = parse_choice(completion, action_list)
    if pred is None:
        return 0.0, False
    return (1.0 if pred == str(gold).strip().upper() else 0.0), True


# ---------------------------------------------------------------- GRPO advantage


def group_advantages(
    rewards: Sequence[float],
    *,
    normalize_by_std: bool = True,
    eps: float = 1e-6,
    std_floor: float = 0.0,
) -> list[float]:
    """Group-centred (and optionally group-scaled) advantages for one prompt group.

    Mirrors compute_grpo_outcome_advantage: mean/std over the group, ddof=1 std, and
    (r - mean) / (std + eps). `normalize_by_std=False` is verl's `norm_adv_by_std_in_grpo`
    off-switch, i.e. the Dr.GRPO / brief-literal variant that only centres.

    verl treats a singleton group as mean=0, std=1 (the advantage is the raw reward).
    Reproduced for fidelity; G >= 4 everywhere in this study so it never fires.
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [float(rewards[0])] if normalize_by_std else [0.0]

    mean = sum(rewards) / n
    centred = [float(r) - mean for r in rewards]
    if not normalize_by_std:
        return centred

    var = sum(c * c for c in centred) / (n - 1)
    sd = math.sqrt(max(var, 0.0))
    denom = max(sd, float(std_floor)) + float(eps)
    return [c / denom for c in centred]


def group_is_degenerate(rewards: Sequence[float]) -> bool:
    """True when every reward in the group is identical -> every advantage is 0."""
    if not rewards:
        return True
    first = rewards[0]
    return all(r == first for r in rewards)


def binary_group_std(k: int, G: int) -> float:
    """ddof=1 std of a 0/1 group with k ones. Used to document the 1/sqrt(G) floor."""
    if G < 2:
        return 0.0
    return math.sqrt(k * (G - k) / (G * (G - 1)))


__all__ = [
    "extract_answer_span",
    "option_labels",
    "option_bodies",
    "parse_choice",
    "score_completion",
    "group_advantages",
    "group_is_degenerate",
    "binary_group_std",
]
