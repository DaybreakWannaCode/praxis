"""CPU-only preflight of every selected prompt under the actual chat processor."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
import yaml
from transformers import AutoProcessor

p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
cfg=yaml.safe_load(a.config.read_text())
proc=AutoProcessor.from_pretrained(cfg['worker']['actor']['model']['model_path'],local_files_only=True)
result={}
for split in ('train','val'):
    rows=pq.read_table(cfg['data'][split+'_files']).to_pylist()
    lengths=[]
    for row in rows:
        messages=[{'role':'system','content':cfg['data']['system_prompt']},
                  {'role':'user','content':[{'type':'text','text':row['problem'].replace('<image>','').strip()}]}]
        prompt=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        inputs=proc(text=[prompt],images=None,add_special_tokens=False,return_tensors='pt')
        lengths.append(int(inputs['input_ids'].shape[1]))
    result[split]=dict(count=len(lengths),minimum=min(lengths),maximum=max(lengths),
                       mean=sum(lengths)/len(lengths),over_limit=sum(n>cfg['data']['max_prompt_length'] for n in lengths))
result['config_sha256']=hashlib.sha256(a.config.read_bytes()).hexdigest()
result['passed']=result['train']['count']==1024 and result['val']['count']==32 and not any(result[s]['over_limit'] for s in ('train','val'))
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if not result['passed']: raise SystemExit(1)
