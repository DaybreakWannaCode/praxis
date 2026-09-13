import unittest
from types import SimpleNamespace
import torch
from transfer_alignment.choice_probe import messages, label_tokens, normalized_target


class ChoiceProbeTests(unittest.TestCase):
    def test_gold_never_changes_prompt(self):
        item = SimpleNamespace(question='Which?', action_list=['A. left', 'B. right'], answer='A')
        before = messages(item)
        item.answer = 'B'
        self.assertEqual(before, messages(item))

    def test_gradient_matches_finite_difference_and_sign(self):
        x = torch.tensor([-.4, .7, .2], dtype=torch.float64, requires_grad=True)
        j = normalized_target(x, 1)
        g, = torch.autograd.grad(j, x)
        d = torch.tensor([.2, -.3, .1], dtype=torch.float64)
        eps = 1e-5
        actual = (normalized_target(x + eps*d, 1)-normalized_target(x-eps*d, 1))/(2*eps)
        self.assertAlmostEqual(float(actual.detach()), float(g@d), places=9)
        self.assertGreater(float(g[1]), 0)
        self.assertAlmostEqual(float(g.sum()), 0, places=12)

    def test_single_token_normalizer_cancels(self):
        logits = torch.tensor([.2, .9, -.3, 8.], dtype=torch.float64)
        self.assertAlmostEqual(float(normalized_target(logits[:3], 1)),
                               float(normalized_target(logits.log_softmax(0)[:3], 1)), places=12)

    def test_prefix_ambiguity_rejected(self):
        tok = SimpleNamespace(encode=lambda x, **kwargs: {'A':[1], 'B':[1, 2]}[x])
        with self.assertRaises(ValueError): label_tokens(tok, ['A','B'])
        tok.encode = lambda x, **kwargs: {'A':[1, 2], 'B':[3, 4]}[x]
        self.assertEqual(label_tokens(tok, ['A','B']), [[1,2],[3,4]])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): normalized_target(torch.tensor([0., float('nan')]), 0)

if __name__ == '__main__': unittest.main()
