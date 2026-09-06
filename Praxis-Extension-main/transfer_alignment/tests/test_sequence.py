"""Exercise the reused sequence extractor without downloading a language model."""
import unittest
from types import SimpleNamespace

import torch

from task_0.src.gradient import PolicyGradientExtractor
from transfer_alignment.backends import TinyBackend, Response
from transfer_alignment.data import synthetic_items
from transfer_alignment.experiment import visual_gradient


class SequenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits_table = torch.nn.Parameter(torch.arange(25, dtype=torch.float64).reshape(5, 5)/10)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits_table[input_ids])


class SequenceTests(unittest.TestCase):
    def extractor(self):
        ex = PolicyGradientExtractor.__new__(PolicyGradientExtractor)
        ex.torch = torch
        ex.model = SequenceModel()
        ex.temperature = 1.0
        ex._ltk_kw = None
        ex._eos_ids = [4]
        return ex

    def test_prompt_mask_alignment_and_temperature(self):
        ex = self.extractor()
        seq = torch.tensor([0, 1, 2, 3, 4])
        ex.temperature = 0.7
        value, length, tokens = ex.sequence_logprob(seq, 3, {}, per_token=True)
        expected = torch.stack([
            (ex.model.logits_table[2].float()/0.7).log_softmax(0)[3],
            (ex.model.logits_table[3].float()/0.7).log_softmax(0)[4],
        ])
        self.assertEqual(length, 2)
        torch.testing.assert_close(tokens, expected)
        value.backward()
        self.assertEqual(ex.model.logits_table.grad[:2].abs().sum().item(), 0)
        self.assertGreater(ex.model.logits_table.grad[2:4].abs().sum().item(), 0)

    def test_eos_kept_padding_ignored(self):
        ex = self.extractor()
        resp = torch.tensor([[2, 4, 0, 0], [4, 0, 0, 0], [1, 2, 3, 1]])
        self.assertEqual(ex.response_lengths(resp).tolist(), [2, 1, 4])

    def test_visual_extractor_uses_sequence_sum(self):
        class FixedBackend(TinyBackend):
            def sample(self, item, modality, count, seed, **kwargs):
                return [Response((item, modality, 0), "A", 1, True, "A", 2),
                        Response((item, modality, 1), "B", 0, True, "B", 2)]

            def logprobs(self, response, **kwargs):
                return super().logprobs(response, **kwargs).repeat(2)

        backend = FixedBackend()
        item = synthetic_items()[8]
        lp = backend.distribution(item, "image")
        expected = torch.autograd.grad(lp[0]-lp[1], backend.model.weight)[0].float()
        actual, _, _ = visual_gradient(backend, [item], 2, 1)
        torch.testing.assert_close(actual["weight"], expected)


if __name__ == "__main__":
    unittest.main()
