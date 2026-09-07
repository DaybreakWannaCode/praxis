import itertools
import unittest

import numpy as np

from transfer_alignment.production_statistics import bernoulli_kl_interval, outcome_contrast, alignment_contrasts


class StatisticsTests(unittest.TestCase):
    def test_heterogeneous_bernoulli_exact_enumeration(self):
        probabilities=[.01,.1,.2,.35,.65,.8,.9,.99]
        target=np.mean(probabilities)
        coverage=0.
        intervals=[bernoulli_kl_interval(k,8) for k in range(9)]
        for outcome in itertools.product((0,1),repeat=8):
            weight=np.prod([p if x else 1-p for p,x in zip(probabilities,outcome)])
            lo,hi=intervals[sum(outcome)]
            coverage+=weight*(lo<=target<=hi)
        self.assertGreaterEqual(coverage,.95)

    def test_zero_observed_changes_does_not_mean_zero_uncertainty(self):
        result=outcome_contrast(np.ones((32,16)),np.ones((32,16)))
        self.assertEqual(result['difference'],0.)
        self.assertLess(result['fixed_panel_interval'][0],0.)
        self.assertGreater(result['fixed_panel_interval'][1],0.)
        self.assertLess(result['fixed_panel_interval'][1],.01)

    def test_swapping_policies_reverses_contrast(self):
        a=np.array([[1,1,0],[0,1,0]])
        b=np.array([[0,1,0],[1,0,0]])
        first,second=outcome_contrast(a,b),outcome_contrast(b,a)
        self.assertAlmostEqual(first['difference'],-second['difference'])
        self.assertEqual(first['fixed_panel_interval'],[-x for x in reversed(second['fixed_panel_interval'])])

    def test_alignment_pairing_cancels_common_probe_noise(self):
        x=np.array([[[1.,2.],[2.,3.]],[[2.,2.],[3.,3.]],[[3.,2.],[4.,3.]],[[4.,2.],[5.,3.]]])
        first=alignment_contrasts(x)[0]
        second=alignment_contrasts(x+np.arange(4).reshape(4,1,1)*100)[0]
        self.assertEqual(first,second)
        self.assertAlmostEqual(first['difference'],.5)

    def test_bad_layout_and_nonbinary_outcomes_rejected(self):
        with self.assertRaises(ValueError):outcome_contrast([[.5]],[[0]])
        with self.assertRaises(ValueError):alignment_contrasts(np.zeros((1,8,4)))


if __name__=='__main__':unittest.main()
