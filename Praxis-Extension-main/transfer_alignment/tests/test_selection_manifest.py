import copy
from pathlib import Path
import tempfile
import unittest
from transfer_alignment.selection_manifest import build_selection,save_selection


def pool():
    return [dict(candidate_id=str(i),prompt_ids=[f'{i}-a',f'{i}-b'],alignment=-float(i),
        parent_sha256='a'*64,scoring_protocol_sha256='b'*64,calibration_split_sha256='c'*64,
        score_receipt_sha256=f'{i:064x}',objective='normalized_label_prefix_log_likelihood_v1') for i in range(8)]

def build(data):return build_selection(data,seeds=[1,2,3],pool_size=8,selected_batches=4,prompts_per_batch=2)


class SelectionTests(unittest.TestCase):
    def test_negative_scores_rank_correctly_and_input_order_is_irrelevant(self):
        data=pool(); result=build(data)
        self.assertEqual(result,build(list(reversed(data))))
        for arm in result['arms']:
            self.assertEqual(len(set(arm['prompt_ids'])),8)
            if arm['selector']=='alignment':self.assertEqual(set(arm['candidate_ids']),{'0','1','2','3'})

    def test_random_membership_independent_of_scores(self):
        data=pool();first=build(data)
        for row in data:row['alignment']=-row['alignment']
        second=build(data)
        self.assertEqual([a for a in first['arms'] if a['selector']=='random'],
                         [a for a in second['arms'] if a['selector']=='random'])
        self.assertNotEqual(first['score_pool_sha256'],second['score_pool_sha256'])

    def test_leakage_mixed_parent_duplicates_and_overwrite_rejected(self):
        for mutate in [lambda d:d[0].update(outcome_delta=1),lambda d:d[0].update(parent_sha256='d'*64),
                       lambda d:d[0].update(prompt_ids=d[1]['prompt_ids'])]:
            data=pool();mutate(data)
            with self.assertRaises(ValueError):build(data)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.json';save_selection(build(pool()),path)
            with self.assertRaises(FileExistsError):save_selection(build(pool()),path)


if __name__=='__main__':unittest.main()
