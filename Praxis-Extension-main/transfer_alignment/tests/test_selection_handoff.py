import copy
from types import SimpleNamespace as NS
import unittest
from transfer_alignment.selection_manifest import validate_training_handoff
from transfer_alignment.tests.test_selection_manifest import build,pool


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.manifest=build(pool());self.ids=self.manifest['arms'][0]['prompt_ids']
        self.config=NS(data=NS(shuffle=False,rollout_batch_size=2),
            worker=NS(actor=NS(global_batch_size=2,ppo_epochs=1),rollout=NS(n=5)),
            trainer=NS(max_steps=4,total_episodes=1))
    def check(self):return validate_training_handoff(self.manifest,selector='alignment',seed=1,
        ordered_prompt_ids=self.ids,config=self.config)

    def test_sealed_training_pass(self):
        self.assertEqual(self.check()['prompts'],8)

    def test_original_prompt_shuffle_rejected(self):
        self.config.data.shuffle=True
        with self.assertRaisesRegex(ValueError,'shuffle'):self.check()

    def test_reordered_data_and_changed_minibatch_rejected(self):
        self.ids=list(reversed(self.ids))
        with self.assertRaisesRegex(ValueError,'order'):self.check()
        self.ids=self.manifest['arms'][0]['prompt_ids']
        self.config.worker.actor.global_batch_size=4
        with self.assertRaisesRegex(ValueError,'batch'):self.check()

    def test_inconsistent_manifest_rejected(self):
        self.manifest['arms'][0]['batch_prompt_ids'][0].reverse()
        with self.assertRaisesRegex(ValueError,'order'):self.check()


if __name__=='__main__':unittest.main()
