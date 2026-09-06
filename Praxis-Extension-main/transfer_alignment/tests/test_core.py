"""Exact mathematical and warm-optimizer replay checks, runnable on CPU."""
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from transfer_alignment.backends import TinyBackend
from transfer_alignment.core import (alignment, capture, clipped_loss, digest, displacement,
                                     dot, isolated_rng, loo_advantages, restore, seed_all, weights)
from transfer_alignment.data import synthetic_items, validate_items
from transfer_alignment.experiment import run, text_step, visual_gradient


def config():
    return json.loads((Path(__file__).parents[1]/"configs/synthetic.json").read_text())


class CoreTests(unittest.TestCase):
    def test_rloo_exact_expectation_matches_reward_derivative(self):
        logits = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
        p = logits.softmax(0).detach()
        # Exhaust every possible independently sampled group: no MC tolerance.
        for g in (2, 3, 4):
            expected = torch.zeros_like(logits)
            for group in itertools.product(range(2), repeat=g):
                mass = float(torch.prod(p[list(group)]))
                reward = [float(a == 0) for a in group]
                adv = loo_advantages(reward)
                loss = sum(adv[j] * logits.log_softmax(0)[a] for j, a in enumerate(group))/g
                expected += mass * torch.autograd.grad(loss, logits)[0]
            true = torch.autograd.grad(logits.softmax(0)[0], logits)[0]
            torch.testing.assert_close(expected, true, atol=1e-12, rtol=1e-12)

    def test_no_length_normalization_and_degenerate_groups(self):
        self.assertEqual(loo_advantages([1, 0]).tolist(), [1, -1])
        self.assertEqual(loo_advantages([1, 1, 1]).tolist(), [0, 0, 0])
        with self.assertRaises(ValueError):
            loo_advantages([1])

    def test_directional_finite_difference(self):
        backend = TinyBackend()
        item = synthetic_items()[8]
        value = backend.distribution(item, "image").exp()[0]
        gradient = torch.autograd.grad(value, backend.model.weight)[0]
        direction = torch.tensor([[0.3, 0.4], [-0.1, 0.2]], dtype=torch.float64)
        before = backend.model.weight.detach().clone()
        epsilon = 1e-5
        vals = []
        with torch.no_grad():
            for sign in [1, -1]:
                backend.model.weight.copy_(before + sign*epsilon*direction)
                vals.append(float(backend.distribution(item, "image").exp()[0]))
            backend.model.weight.copy_(before)
        self.assertAlmostEqual((vals[0]-vals[1])/(2*epsilon), dot({"w": gradient}, {"w": direction}), places=10)

    def test_cancellation_precision_and_zero_cosine(self):
        a = {"x": torch.tensor([1e8, 1, -1e8])}
        self.assertEqual(dot(a, {"x": torch.ones(3)}), 1.0)
        self.assertIsNone(alignment(a, {"x": torch.zeros(3)})["cosine"])
        with self.assertRaises(ValueError):
            dot(a, {"wrong": torch.ones(3)})

    def test_rng_isolation(self):
        seed_all(123)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        seed_all(123)
        with isolated_rng(456):
            random.random(), np.random.rand(), torch.rand(8)
        actual = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_warm_adam_scheduler_reference_rng_restore(self):
        backend = TinyBackend()
        cfg = config()
        opt = torch.optim.AdamW(backend.model.parameters(), lr=cfg["lr"])
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
        batch = synthetic_items()[:2]
        text_step(backend, opt, sch, batch, cfg, 44)
        parent = capture(backend.model, opt, sch)
        self.assertTrue(parent["optimizer"]["state"])
        delta1, _, _ = text_step(backend, opt, sch, batch, cfg, 99)
        child = capture(backend.model, opt, sch)
        with torch.no_grad():
            backend.model.reference_weight.add_(99)
        restore(parent, backend.model, opt, sch)
        self.assertEqual(digest(capture(backend.model, opt, sch)), digest(parent))
        delta2, _, _ = text_step(backend, opt, sch, batch, cfg, 99)
        self.assertEqual(digest(delta1), digest(delta2))
        self.assertEqual(digest(capture(backend.model, opt, sch)), digest(child))

    def test_clipping_and_kl(self):
        x = torch.tensor([0.5], requires_grad=True)
        loss = clipped_loss(x, torch.zeros(1), torch.zeros(1), 1.0, clip=0.2, kl_coef=0)
        self.assertAlmostEqual(loss.item(), -1.2, places=6)
        self.assertEqual(torch.autograd.grad(loss.sum(), x)[0].item(), 0)
        kl = clipped_loss(x, x.detach(), torch.zeros(1), 0.0, clip=0.2, kl_coef=1)
        self.assertGreater(kl.item(), 0)

    def test_split_leakage_rejected(self):
        items = synthetic_items()
        items[8].group_id = items[0].group_id
        with self.assertRaises(ValueError):
            validate_items(items)
        with self.assertRaises(ValueError):
            visual_gradient(TinyBackend(), synthetic_items()[:2], 4, 1)

    def test_end_to_end_two_branches(self):
        torch.set_num_threads(1)
        seed_all(123)
        items, cfg = synthetic_items(), config()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"run"
            outcomes = run(TinyBackend(), items, [items[:2], items[2:4]], cfg, output)
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(all(x["replay_exact"] for x in outcomes))
            self.assertTrue((output/"scores_sealed.json").exists())
            self.assertEqual(json.loads((output/"manifest.json").read_text())["status"], "complete")
            rows = [json.loads(x) for x in (output/"evaluations.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["split"] == "dev" for r in rows))
            probe = [json.loads(x) for x in (output/"probe_responses.jsonl").read_text().splitlines()]
            self.assertFalse({r["group_id"] for r in rows} & {r["group_id"] for r in probe})
            self.assertNotEqual(rows[0]["seed"], rows[-1]["seed"])


if __name__ == "__main__":
    unittest.main()
