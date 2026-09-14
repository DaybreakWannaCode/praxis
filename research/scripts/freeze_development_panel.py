"""Freeze the first 256 retained images and a 32-image shuffle control."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;review_path=root/'review-decisions.json'
    review=json.loads(review_path.read_text());audit=json.loads((root/'image-audit.json').read_text())
    decisions=review['decisions']
    if [r['id'] for r in decisions]!=audit['unflagged_ids_in_frozen_order'][:len(decisions)]:
        raise ValueError('Review decisions are not in the frozen order')
    if any(r['decision'] not in ('retain','exclude') for r in decisions):raise ValueError('Unresolved decision')
    retained=[r for r in decisions if r['decision']=='retain'][:256]
    if len(retained)!=256 or len({r['id'] for r in retained})!=256:raise ValueError('Need 256 reviewed distinct IDs')
    annotation_path=root/'VIVA_annotation.json'
    inventory=json.loads((root/'annotation-audit.json').read_text())
    expected=inventory['source_sha256']['/workspace/praxis/data/sources/VIVA_annotation.json']
    if sha(annotation_path)!=expected:raise ValueError('Annotation bytes changed')
    annotations={f"viva-{r['index']}":r for r in json.loads(annotation_path.read_text())}
    items=[];hashes={}
    with tarfile.open(root/'acquisition-backup.tar') as tar:
        for decision in retained:
            id=decision['id'];index=id.split('-')[-1];source=annotations[id]
            image=tar.extractfile(f'images/{index}.image').read()
            if hashlib.sha256(image).hexdigest()!=decision['image_sha256']:raise ValueError('Reviewed image differs')
            path=f'/workspace/praxis/data/independent-visual-inventory-20260915/images/{index}.image'
            if source['answer'] not in [o.split('.')[0].strip() for o in source['action_list']]:raise ValueError('Invalid answer label')
            items.append(dict(id=id,group_id=f'viva-scene-{index}',split='dev',
                question='Given the situation, which of the following actions is the most appropriate?',
                action_list=source['action_list'],answer=source['answer'],situation=source['situation_description'],image_path=path))
            hashes[path]=decision['image_sha256']
    subset=sorted(items,key=lambda r:hashlib.sha256(('image-dependence-20260915:'+r['group_id']).encode()).hexdigest())[:32]
    controls=[];mapping={}
    for j,item in enumerate(subset):
        donor=subset[(j+1)%32]
        assert donor['group_id']!=item['group_id']
        controls.append(dict(item,id=item['id']+'-shuffled',image_path=donor['image_path']))
        mapping[item['id']]=donor['id']
    result=dict(status='panel_frozen_before_endpoint_outcomes',items=items,shuffled_controls=controls,
        image_sha256=hashes,shuffle_mapping=mapping,selected_ids=[r['id'] for r in items],
        source_sha256={name:sha(root/name) for name in ['review-decisions.json','image-audit.json','annotation-audit.json','download-plan.json','local-backup-audit.json','VIVA_annotation.json']},
        scope='Reviewed development subset for adapted ordinary baseline; not official full-benchmark score or final test',
        limits='Assistant image/annotation review with exact and perceptual duplicate screening; source-scene independence remains uncertain. Scene IDs are proxies. Situation descriptions are metadata, not model input. Endpoint execution/prompt/parser/model lock required separately.')
    with (root/'development-panel.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(images=len(items),controls=len(controls),last_selected_id=items[-1]['id'],sha256=sha(root/'development-panel.json'))))


if __name__=='__main__':main()
