import unittest
from transfer_alignment.prepare import parse_text


class PrepareTests(unittest.TestCase):
    def test_current_problem_schema_without_answer_leakage(self):
        row = {"problem": "Instructions\n## Situation:\nA vehicle blocks the crossing.\n"
               "## Question:\nWhat next?\nA. Wait.\nB. Walk around it.\n\nNow answer the question. Just output the choice:",
               "answer": "A", "messages": [{"role": "assistant", "content": "SECRET GOLD RATIONALE"}]}
        item = parse_text(row, 7)
        self.assertEqual(item.question, "What next?")
        self.assertEqual(item.action_list, ["A. Wait.", "B. Walk around it."])
        self.assertEqual(item.situation, "A vehicle blocks the crossing.")
        self.assertNotIn("SECRET", item.question + item.situation + str(item.action_list))

    def test_unrecognized_template_fails_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            parse_text({"question": "Unstructured unknown prompt", "answer": "A"}, 0)


if __name__ == "__main__":
    unittest.main()
