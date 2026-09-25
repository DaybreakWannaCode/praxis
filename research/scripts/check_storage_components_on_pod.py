"""CPU-only type/filesystem check; creates only a new private smoke directory."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from verl.protocol import DataProto
from transfer_alignment.candidate_inputs import capture_inputs
from transfer_alignment.checkpoint_publication import CheckpointStore

root=Path('/workspace/praxis/runs/storage-components-20260915-001')
root.mkdir(exist_ok=False)
data=DataProto.from_dict(tensors={'input_ids':torch.tensor([[1,2,3]])},
    non_tensors={'labels':np.array(['A'],dtype=object)},meta_info={'temperature':1.})
worker=SimpleNamespace(fsdp_module=torch.nn.Linear(2,2))
receipt=capture_inputs(worker,data,root/'inputs','a'*64)
store=CheckpointStore(root/'checkpoints',create=True)
def write(step):
    return lambda path:torch.save({'weight':torch.ones(3),'step':step},path/'state.pt')
def validate(path):
    value=torch.load(path/'state.pt',weights_only=True)
    assert torch.isfinite(value['weight']).all()
    return {'status':'passed','step':value['step']}
store.publish(1,write(1),validate)
store.publish(2,write(2),validate)
assert not (store.root/'global_step_1').exists()
assert (store.root/'global_step_2/state.pt').exists()
def fail(path):raise OSError('intentional interrupted save')
try:store.publish(3,fail,validate)
except OSError:pass
else:raise AssertionError('Expected save failure')
assert json.loads((store.root/'latest.json').read_text())['step']==2
report=dict(status='passed',torch=torch.__version__,input_type='original verl.protocol.DataProto',
    captured_input_bytes=receipt['bytes'],serialization_roundtrip_exact=receipt['serialization_roundtrip_exact'],
    publication_and_directory_fsync=True,previous_preserved_after_failed_replacement=True,
    scope='CPU-only synthetic input and tiny checkpoint on the real network volume; no actor update or full-size checkpoint validation')
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
