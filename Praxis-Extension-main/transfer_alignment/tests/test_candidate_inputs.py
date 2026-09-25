import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transfer_alignment.candidate_inputs import capture_inputs, input_digest, install_worker_capture
from transfer_alignment.core import rng_state, digest


class Proto:
    def __init__(self):
        self.batch={'input_ids':torch.tensor([[1,2,3]])}
        self.meta_info={'temperature':1.}
        self.non_tensor_batch={'labels':np.array(['A'],dtype=object)}
    def to(self,device):
        self.batch={k:v.to(device) for k,v in self.batch.items()}
        return self


class CaptureTests(unittest.TestCase):
    def worker(self):
        return SimpleNamespace(fsdp_module=torch.nn.Linear(2,2))

    def test_roundtrip_preserves_objects_and_rng(self):
        with tempfile.TemporaryDirectory() as d:
            worker=self.worker();data=Proto();before=digest(rng_state())
            receipt=capture_inputs(worker,data,Path(d)/'capture','a'*64)
            self.assertEqual(receipt['status'],'captured')
            self.assertEqual(before,digest(rng_state()))
            self.assertEqual(receipt['input_digest'],input_digest(data))
            x=torch.load(Path(d)/'capture/fixed-update-input.pt',weights_only=False)
            self.assertEqual(input_digest(x['data']),input_digest(data))

    def test_refuses_overwriting_existing_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):capture_inputs(self.worker(),Proto(),d,'a'*64)

    def test_parent_identity_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):capture_inputs(self.worker(),Proto(),Path(d)/'x','bad')

    def test_wrapper_preserves_dispatch_metadata_and_original_result(self):
        class Worker:
            def update_actor(self,data):
                self.optimizer.state['p']['step']+=1
                return 'original result'
        Worker.update_actor.dispatch_marker='actor'
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{
            'PRAXIS_CAPTURE_CANDIDATE_INPUTS':'1','PRAXIS_CANDIDATE_INPUT_DIR':str(Path(d)/'x'),
            'PRAXIS_PARENT_RECEIPT_SHA256':'a'*64}):
            install_worker_capture(Worker)
            self.assertEqual(Worker.update_actor.dispatch_marker,'actor')
            w=Worker();w.fsdp_module=torch.nn.Linear(2,2)
            w.optimizer=SimpleNamespace(state={'p':{'step':16}})
            self.assertEqual(w.update_actor(Proto()),'original result')
            with self.assertRaises(ValueError):w.update_actor(Proto())

    def test_no_implicit_install(self):
        class Worker:pass
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ValueError):install_worker_capture(Worker)


if __name__=='__main__':unittest.main()
