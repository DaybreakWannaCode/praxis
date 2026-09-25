import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from transfer_alignment.restore_only_hooks import install_manager, install_trainer


class RestoreHookTests(unittest.TestCase):
    def test_opt_in_required(self):
        with patch.dict(os.environ,{},clear=True), self.assertRaises(RuntimeError):
            install_trainer(type('Trainer',(),{}))

    def test_driver_only_loads_then_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir()
            calls=[]
            class Trainer:
                use_critic=False
                config=NS(trainer=NS(n_gpus_per_node=1,nnodes=1,load_checkpoint_path=str(parent)))
                def _load_checkpoint(self):
                    calls.append('load');self.global_step=16
                    (root/'loaded-state.json').write_text(json.dumps({'status':'passed'}))
            with patch.dict(os.environ,PRAXIS_RESTORE_ONLY='1',PRAXIS_RESTORE_RUN=tmp,PRAXIS_RESTORE_PARENT=str(parent)):
                install_trainer(Trainer);Trainer().fit()
            self.assertEqual(calls,['load'])
            self.assertEqual(json.loads((root/'driver-complete.json').read_text())['restored_driver_step'],16)

    def test_manager_checks_parent_and_prohibits_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            class Manager:
                def load_checkpoint(self,path):calls.append(path)
            with patch.dict(os.environ,PRAXIS_RESTORE_ONLY='1',PRAXIS_RESTORE_RUN=tmp,PRAXIS_RESTORE_PARENT=tmp):
                install_manager(Manager)
                with self.assertRaises(ValueError):Manager().load_checkpoint('/wrong')
                with self.assertRaises(RuntimeError):Manager().save_checkpoint('/any')
            self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
