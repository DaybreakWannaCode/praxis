import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from transfer_alignment.prepare import parse_text, prepare
from transfer_alignment.data import load_manifest


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

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow required for image preparation integration")
    def test_local_images_to_disjoint_manifest(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            texts = [{"problem": f"## Situation:\nTraining situation {i}.\n## Question:\n"
                      "What next?\nA. Wait.\nB. Leave.\nNow answer the question.", "answer": "A"}
                     for i in range(2)]
            (root/"text.json").write_text(json.dumps(texts))
            images = root/"images"
            images.mkdir()
            rows = []
            for i in range(4):
                Image.new("RGB", (64, 64), (i*50, 10, 20)).save(images/f"{i}.png")
                rows.append({"index": i, "image_file": f"{i}.png", "image_url": f"https://example.org/{i}",
                             "situation_description": f"Distinct visual situation {i}",
                             "action_list": ["A. Wait.", "B. Leave."], "answer": "A"})
            (root/"viva.json").write_text(json.dumps(rows))
            manifest = prepare(root/"text.json", root/"viva.json", images, root/"out",
                               text_revision="a"*40, viva_revision="b"*40)
            items = load_manifest(manifest)
            self.assertEqual([sum(i.split == s for i in items) for s in ["train", "score", "dev"]], [2, 2, 2])
            self.assertEqual(len({i.group_id for i in items}), 6)
            self.assertTrue((root/"out/provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
