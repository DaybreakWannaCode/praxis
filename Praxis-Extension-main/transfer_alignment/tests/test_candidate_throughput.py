"""Guard against accidentally broadening the bounded cost experiment."""
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from transfer_alignment.candidate_throughput import (
    install_trainer, install_checkpoint_manager, optimizer_steps, validate_config)


class CandidateCostTests(unittest.TestCase):
    def config(self):
        return NS(trainer=NS(max_steps=1,total_episodes=1,load_checkpoint_path='/parent'),
                  data=NS(rollout_batch_size=32),
                  worker=NS(rollout=NS(n=5),actor=NS(global_batch_size=32,ppo_epochs=1)))

    def test_rejects_multi_update_budget(self):
        cfg=self.config()
        validate_config(cfg)
        for obj, field, value in [(cfg.trainer,'max_steps',128),
                                  (cfg.trainer,'total_episodes',3),
                                  (cfg.worker.actor,'ppo_epochs',4)]:
            old=getattr(obj,field)
            setattr(obj,field,value)
            with self.assertRaises(ValueError): validate_config(cfg)
            setattr(obj,field,old)

    def test_requires_parent(self):
        cfg=self.config(); cfg.trainer.load_checkpoint_path=None
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_hooks_are_not_implicitly_enabled(self):
        class Dummy: pass
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError): install_trainer(Dummy)
            with self.assertRaises(RuntimeError): install_checkpoint_manager(Dummy)
        self.assertFalse(hasattr(Dummy,'_load_checkpoint'))

    def test_step_inventory_does_not_invent_frozen_parameter_updates(self):
        optimizer=NS(state={'active':{'step':16},'frozen':{}})
        self.assertEqual(optimizer_steps(optimizer),[16])

    def test_restore_keeps_new_candidate_data_and_requires_one_batch(self):
        class Trainer: pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); calls=[]
            cfg=self.config(); cfg.trainer.load_checkpoint_path=str(root/'parent')
            instance=Trainer(); instance.config=cfg
            instance.train_dataset=list(range(32)); instance.train_dataloader=['new batch']
            instance.actor_rollout_wg=NS(load_checkpoint=lambda path:calls.append(path))
            with patch.dict(os.environ,{'PRAXIS_SINGLE_CANDIDATE_COST':'1',
                                       'PRAXIS_CANDIDATE_PARENT':str(root/'parent'),
                                       'PRAXIS_CANDIDATE_COST_DIR':str(root)}):
                install_trainer(Trainer); instance._load_checkpoint()
                self.assertEqual(calls,[str(root/'parent'/'actor')])
                self.assertEqual(instance.train_dataloader,['new batch'])
                self.assertEqual(instance.global_step,0)
                self.assertEqual(json.loads((root/'parent-restore.json').read_text())['driver_step_reset'],0)
                instance.train_dataset.append(32)
                with self.assertRaises(ValueError): instance._load_checkpoint()
                self.assertEqual(len(calls),1)

    def test_cold_optimizer_or_zero_lr_cannot_be_candidate_parent(self):
        class Manager:
            def load_checkpoint(self,path): self.loaded=path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.dict(os.environ,{'PRAXIS_SINGLE_CANDIDATE_COST':'1',
                                       'PRAXIS_CANDIDATE_PARENT':str(root)}):
                install_checkpoint_manager(Manager)
                manager=Manager()
                manager.optimizer=NS(state={'p':{'step':0}},param_groups=[{'lr':1e-6}])
                with self.assertRaises(ValueError): manager.load_checkpoint(root/'actor')
                manager.optimizer.state['p']['step']=16
                manager.optimizer.param_groups[0]['lr']=0
                with self.assertRaises(ValueError): manager.load_checkpoint(root/'actor')
                manager.optimizer.param_groups[0]['lr']=1e-6
                manager.load_checkpoint(root/'actor')
                self.assertEqual(manager._cost_parent_steps,[16])


if __name__ == '__main__': unittest.main()
