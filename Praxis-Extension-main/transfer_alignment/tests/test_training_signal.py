import importlib.util
from pathlib import Path
import sys
import types
import unittest

from transfer_alignment.training_signal import (REASONING_SUFFIX, RELEASED_SUFFIX, apply_prompt_contract,
                                                audit_metrics, format_ok, rollout_signal, tag_count)

PROBLEM = ("You are given a situation and a question.\n\n## Situation: \nA fire alarm sounds.\n\n"
           "## Question:\nWhat should you do?\nA. Leave calmly.\nB. Ignore it.\n\n" + RELEASED_SUFFIX)
GOOD = "<think>\nThe alarm signals danger, so leaving is safest.\n</think>\n<answer>\nA\n</answer>"


def _original_mcq():
    path = Path(__file__).resolve().parents[3] / "Praxis-VLM-main/verl/utils/reward_score/mcq.py"
    if not path.exists():
        return None
    sys.modules.setdefault("mathruler", types.ModuleType("mathruler"))
    grader = types.ModuleType("mathruler.grader")
    grader.extract_boxed_content = grader.grade_answer = lambda *a: None
    sys.modules.setdefault("mathruler.grader", grader)
    spec = importlib.util.spec_from_file_location("original_mcq", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(step, fmt=0.0, length=18.0, acc=0.7):
    return {"step": step, "metrics": {"reward/accuracy": acc, "reward/format": fmt, "reward/tag_count": fmt,
                                      "reward/length": 0.0, "response_length/mean": length}}


class PromptContractTests(unittest.TestCase):
    def test_released_is_unchanged(self):
        self.assertEqual(apply_prompt_contract(PROBLEM, "released"), PROBLEM)

    def test_reasoning_replaces_choice_only_suffix(self):
        prompt = apply_prompt_contract(PROBLEM, "reasoning")
        self.assertNotIn("Just output the choice", prompt)
        self.assertTrue(prompt.endswith(REASONING_SUFFIX))
        self.assertIn("B. Ignore it.", prompt)

    def test_reasoning_rejects_unexpected_template(self):
        with self.assertRaises(ValueError):
            apply_prompt_contract("Pick one: A or B", "reasoning")
        with self.assertRaises(ValueError):
            apply_prompt_contract(PROBLEM, "freeform")

    def test_instructed_layout_earns_original_format_reward(self):
        example = REASONING_SUFFIX.split("format:\n", 1)[1]
        self.assertTrue(format_ok(example))


class RewardParityTests(unittest.TestCase):
    def test_matches_original_format_and_tag_rules(self):
        mcq = _original_mcq()
        if mcq is None:
            self.skipTest("Original Praxis reference tree not present")
        for text in (GOOD, "A", "B. Ignore it.", "<think>x</think><answer>A</answer>",
                     "<think>\nx\n</think>\n<answer>\nA\n</answer>\n", GOOD + " extra"):
            self.assertEqual(float(format_ok(text)), mcq.format_reward(text), text)
            self.assertEqual(tag_count(text), mcq.tag_count_reward(text), text)


class RolloutSignalTests(unittest.TestCase):
    def test_letter_only_rollouts_have_no_format_signal(self):
        samples = [{"group_id": g, "text": "A", "correct": True} for g in range(4) for _ in range(5)]
        result = rollout_signal(samples)
        self.assertEqual(result["format_rate"], 0.0)
        self.assertEqual(result["groups_with_reward_variation"], 0.0)

    def test_mixed_format_creates_group_variation(self):
        samples = [{"group_id": 0, "text": GOOD, "correct": True},
                   {"group_id": 0, "text": "A", "correct": True},
                   {"group_id": 1, "text": "A", "correct": True},
                   {"group_id": 1, "text": "A", "correct": True}]
        result = rollout_signal(samples)
        self.assertEqual(result["format_rate"], 0.25)
        self.assertEqual(result["groups_with_formatted_response"], 0.5)
        self.assertEqual(result["groups_with_reward_variation"], 0.5)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            rollout_signal([])


class MetricsAuditTests(unittest.TestCase):
    def test_letter_regime_fails(self):
        result = audit_metrics([_metrics(i) for i in range(1, 33)])
        self.assertFalse(result["passed"])
        self.assertEqual(result["format_nonzero_steps"], 0)
        self.assertEqual(len(result["failures"]), 2)

    def test_active_reasoning_passes(self):
        rows = [_metrics(i, fmt=min(1.0, i / 10), length=20 + 10 * i) for i in range(1, 33)]
        result = audit_metrics(rows)
        self.assertTrue(result["passed"], result["failures"])
        self.assertGreater(result["last_window"]["response_length"], result["first_window"]["response_length"])

    def test_validation_rows_ignored_and_short_runs_rejected(self):
        rows = [_metrics(i) for i in range(1, 5)] + [{"step": 4, "metrics": {"val/format_reward": 0.0}}]
        with self.assertRaises(ValueError):
            audit_metrics(rows)


if __name__ == "__main__":
    unittest.main()
