import tempfile
import unittest
from pathlib import Path

from transfer_alignment.backends import TinyBackend
from transfer_alignment.calibrate import run, summarize
from transfer_alignment.data import synthetic_items


class CalibrationTests(unittest.TestCase):
    def test_format_and_accuracy_are_separate(self):
        rows=[dict(parsed=True,correct=1,length=10,truncated=False),
              dict(parsed=True,correct=0,length=20,truncated=False),
              dict(parsed=False,correct=0,length=30,truncated=True)]
        result=summarize(rows)
        self.assertEqual(result["wrong_parsed"],1)
        self.assertEqual(result["unparsed"],1)
        self.assertEqual(result["accuracy_given_parsed"],0.5)
        self.assertEqual(result["p95_tokens"],30)

    def test_inference_repeats_do_not_update_and_reject_test_split(self):
        backend=TinyBackend()
        items=[i for i in synthetic_items() if i.split=="dev"][:2]
        with tempfile.TemporaryDirectory() as temp:
            result=run(backend,items,Path(temp)/"run",samples=2,replicates=2)
            self.assertTrue(result["trainable_parameters_unchanged"])
            self.assertEqual(result["aggregate"]["responses"],8)
            items[0].split="test"
            with self.assertRaises(ValueError):
                run(backend,items,Path(temp)/"forbidden")
