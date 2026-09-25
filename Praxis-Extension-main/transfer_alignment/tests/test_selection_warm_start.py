import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from transfer_alignment.selection_warm_start import install_selection_warm_start
from transfer_alignment.tests.test_selection_manifest import build, pool


class WarmStartTests(unittest.TestCase):
    def test_fresh_arm_consumes_every_selected_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = root / 'global_step_16'
            (parent / 'actor').mkdir(parents=True)
            (parent / 'dataloader.pt').write_bytes(b'old dataset state must not be read')
            manifest = build(pool())
            arm = manifest['arms'][0]
            config = NS(data=NS(shuffle=False, rollout_batch_size=2),
                worker=NS(actor=NS(global_batch_size=2, ppo_epochs=1), rollout=NS(n=5)),
                trainer=NS(max_steps=4, total_episodes=1, load_checkpoint_path=str(parent)))
            loads = []
            trainer = NS(use_critic=False, global_step=0, training_steps=4,
                config=config, actor_rollout_wg=NS(load_checkpoint=loads.append),
                train_dataloader=list(arm['batch_prompt_ids']))
            args = dict(parent=parent, manifest=manifest, selector='alignment', seed=1,
                ordered_prompt_ids=arm['prompt_ids'], receipt_path=root / 'load.json')
            install_selection_warm_start(trainer, **args)
            trainer._load_checkpoint()
            observed = []
            # Original loop increments the driver step before checking its cap.
            for batch in trainer.train_dataloader:
                trainer.global_step += 1
                if trainer.global_step > trainer.training_steps:
                    break
                observed.extend(batch)
            self.assertEqual(observed, arm['prompt_ids'])
            self.assertEqual(trainer.global_step, 4)
            self.assertEqual(loads, [str(parent.resolve() / 'actor')])
            self.assertFalse(json.loads((root / 'load.json').read_text())['parent_dataloader_restored'])
            with self.assertRaises(ValueError): trainer._load_checkpoint()
            with self.assertRaises(ValueError): install_selection_warm_start(trainer, **args)


if __name__ == '__main__':
    unittest.main()
