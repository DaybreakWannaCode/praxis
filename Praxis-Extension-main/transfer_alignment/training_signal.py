"""Check that text GRPO actually exercises the reasoning reward before measuring transfer.

The released `problem` field ends with a choice-only instruction. Under it, the
adapted 32-step baseline never earned format, tag or length reward: the policy
emitted ~18-token letters and the composite reward reduced to letter accuracy.
Every downstream alignment and transfer measurement inherits that regime.
"""
from __future__ import annotations

import re
from collections import defaultdict

RELEASED_SUFFIX = "Now answer the question. Just output the choice:"
# Mirrors the exact layout required by the original `format_reward` regex.
REASONING_SUFFIX = ("Now answer the question. First reason about the situation and each option, "
                    "then give only the option letter. Use exactly this format:\n"
                    "<think>\nyour reasoning\n</think>\n<answer>\nletter\n</answer>")
CONTRACTS = ("released", "reasoning")
_WINDOW_KEYS = (("accuracy", "reward/accuracy"), ("format", "reward/format"),
                ("tag_count", "reward/tag_count"), ("length_reward", "reward/length"),
                ("response_length", "response_length/mean"))
_FORMAT = re.compile(r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$", re.DOTALL)


def apply_prompt_contract(problem, contract):
    """Return the training prompt under a declared contract; `released` is unchanged."""
    if contract not in CONTRACTS:
        raise ValueError(f"Unknown prompt contract {contract!r}")
    if contract == "released":
        return problem
    stripped = problem.rstrip()
    if not stripped.endswith(RELEASED_SUFFIX):
        raise ValueError("Problem does not end with the released choice-only suffix")
    return stripped[:-len(RELEASED_SUFFIX)] + REASONING_SUFFIX


def format_ok(text):
    return bool(_FORMAT.match(text))


def tag_count(text):
    return 0.25 * sum((text.count("<think>\n") == 1, text.count("\n</think>\n") == 1,
                       text.count("\n<answer>\n") == 1, text.count("\n</answer>") == 1))


def rollout_signal(samples):
    """Summarize sampled rollouts: [{'group_id', 'text', 'correct'}] -> reasoning-signal rates.

    A GRPO group contributes gradient only if its composite rewards differ, so a
    format term never observed within any group cannot be learned by sampling.
    """
    if not samples:
        raise ValueError("No rollouts supplied")
    groups = defaultdict(list)
    for s in samples:
        groups[s["group_id"]].append(s)
    formatted = [format_ok(s["text"]) for s in samples]
    varied = [len({(format_ok(s["text"]), tag_count(s["text"]), bool(s["correct"])) for s in g}) > 1
              for g in groups.values()]
    return {"responses": len(samples), "groups": len(groups),
            "format_rate": sum(formatted) / len(samples),
            "any_tag_rate": sum(tag_count(s["text"]) > 0 for s in samples) / len(samples),
            "groups_with_formatted_response": sum(any(format_ok(s["text"]) for s in g)
                                                  for g in groups.values()) / len(groups),
            "groups_with_reward_variation": sum(varied) / len(groups)}


def audit_metrics(rows, window=8, min_format=0.1, min_length=64.0):
    """Audit trainer metrics.jsonl rows; `passed` requires the reasoning reward to be active.

    Thresholds are engineering gates, not scientific effect sizes: a run that
    never earns format reward and keeps letter-length responses is not the
    Praxis text-reasoning treatment, whatever its accuracy.
    """
    steps = sorted((r for r in rows if "reward/accuracy" in r.get("metrics", {})), key=lambda r: r["step"])
    if len(steps) < window:
        raise ValueError(f"Need at least {window} training steps, found {len(steps)}")
    series = lambda key, part: [r["metrics"][key] for r in part]
    mean = lambda xs: sum(xs) / len(xs)
    first, last = steps[:window], steps[-window:]
    result = {"steps": len(steps), "window": window,
              "format_nonzero_steps": sum(r["metrics"]["reward/format"] > 0 for r in steps),
              "tag_nonzero_steps": sum(r["metrics"]["reward/tag_count"] > 0 for r in steps),
              "length_reward_nonzero_steps": sum(r["metrics"]["reward/length"] > 0 for r in steps)}
    for name, part in (("first", first), ("last", last)):
        result[f"{name}_window"] = {label: mean(series(key, part)) for label, key in _WINDOW_KEYS}
    failures = []
    last = result["last_window"]
    if last["format"] < min_format:
        failures.append(f"final-window format reward {last['format']:.3f} < {min_format}")
    if last["response_length"] < min_length:
        failures.append(f"final-window mean response length {last['response_length']:.1f} "
                        f"< {min_length} tokens")
    result["failures"] = failures
    result["passed"] = not failures
    return result
