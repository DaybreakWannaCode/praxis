"""Freeze 1,024 distinct textual situations for the ordinary single-GPU baseline."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from transfer_alignment.prepare import parse_text, normalize, file_hash
from transfer_alignment.training_signal import apply_prompt_contract
from task_0.src.rewards import option_labels


def prepare(source, visual_manifest, output, prompt_contract='released'):
    output.mkdir(parents=True, exist_ok=False)
    visual=json.loads(visual_manifest.read_text())
    forbidden={normalize(r['situation']) for r in visual if r.get('situation')}
    blocked_qa={normalize(r['question']+'\n'+'\n'.join(r['action_list'])) for r in visual}
    rows=pq.read_table(source).to_pylist()
    eligible=[]; excluded=[]; seen=set()
    for index, row in enumerate(rows):
        try:
            item=parse_text(row,index)
            if item.answer not in option_labels(item.action_list): raise ValueError('invalid answer')
            key=normalize(item.situation)
            if key in forbidden or normalize(item.question+'\n'+'\n'.join(item.action_list)) in blocked_qa:
                raise ValueError('visual-panel exact overlap')
            if key in seen: raise ValueError('duplicate textual situation')
            seen.add(key)
            order=hashlib.sha256(('20260914:'+key).encode()).hexdigest()
            eligible.append((order,index,item,row['problem']))
        except (ValueError,TypeError) as exc:
            excluded.append(dict(index=index,reason=str(exc)))
    eligible.sort(key=lambda row:row[0])
    if len(eligible)<1056: raise ValueError('Need 1,024 train and 32 disjoint text-validation situations')
    selected=eligible[:1056]
    for name, subset in [('train',selected[:1024]),('val',selected[1024:])]:
        pq.write_table(pa.Table.from_pylist([dict(problem=apply_prompt_contract(p,prompt_contract),answer=i.answer) for _,_,i,p in subset]),output/f'{name}.parquet')
    audit=dict(source_sha256=file_hash(source), visual_manifest_sha256=file_hash(visual_manifest),
               raw_rows=len(rows), eligible_situations=len(eligible), exclusions=excluded,
               train_rows=1024, val_rows=32, seed=20260914, prompt_contract=prompt_contract,
               train_indices=[x[1] for x in selected[:1024]], val_indices=[x[1] for x in selected[1024:]],
               output_sha256={n:file_hash(output/n) for n in ('train.parquet','val.parquet')},
               limits='Exact normalized scene and QA checks only; not proof against paraphrases or unknown shared source scenes. Source prompt/rationale fields are not fed to the model; only problem and answer columns are retained.')
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    (output/'selected-items.json').write_text(json.dumps([asdict(x[2]) for x in selected],indent=2))
    print(json.dumps({k:audit[k] for k in ('raw_rows','eligible_situations','train_rows','val_rows')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--visual-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    # `released` keeps the choice-only suffix (completed 2026-09-14 baseline); `reasoning` is a new declared contract.
    p.add_argument('--prompt-contract',choices=('released','reasoning'),default='released')
    a=p.parse_args(); prepare(a.source,a.visual_manifest,a.output,a.prompt_contract)
