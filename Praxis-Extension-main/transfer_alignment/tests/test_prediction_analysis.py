import unittest
from transfer_alignment.prediction_analysis import analyze_prediction, ranks


def rows(parent='p', reverse=False):
    return [dict(parent_id=parent,candidate_id=str(i),alignment=i,cosine=i/4,
                 update_norm=i+1,outcome_delta=(-i if reverse else i)) for i in range(5)]


class PredictionTests(unittest.TestCase):
    def test_ties_get_average_ranks(self):
        self.assertEqual(ranks([3,1,1,2]),[3,0.5,0.5,2])

    def test_within_parent_not_global_offset(self):
        data=rows('a')+rows('b',reverse=True)
        for r in data[5:]:r['alignment']+=10000;r['outcome_delta']+=10000
        result=analyze_prediction(data,permutations=99)
        a=result['results']['alignment']
        self.assertEqual(a['per_parent_spearman'],{'a':1.,'b':-1.})
        self.assertEqual(a['equal_parent_mean_spearman'],0.)
        self.assertEqual(a['two_sided_permutation_p'],1.)

    def test_flat_outcomes_not_zero_correlation_evidence(self):
        data=rows()
        for r in data:r['outcome_delta']=0
        self.assertEqual(analyze_prediction(data,permutations=9)['status'],'not_estimable')

    def test_common_denominator_and_reproducibility(self):
        data=rows('a')+rows('b')
        for r in data[5:]:r['cosine']=0
        a=analyze_prediction(data,permutations=199)
        self.assertEqual(a,analyze_prediction(data,permutations=199))
        self.assertEqual(a['comparison_parents'],['a'])
        self.assertEqual(a['alignment_all_informative_parents'],{'a':1.,'b':1.})
        self.assertIn('cosine',a['excluded_parents']['b']['fields'])
        self.assertGreater(a['results']['alignment']['two_sided_permutation_p'],0)

    def test_duplicate_and_nonfinite_rejected(self):
        data=rows()
        with self.assertRaises(ValueError):analyze_prediction(data+[data[0]])
        data[0]['alignment']=float('nan')
        with self.assertRaises(ValueError):analyze_prediction(data)


if __name__=='__main__':unittest.main()
