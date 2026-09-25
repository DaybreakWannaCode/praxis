"""Freeze independent baseline endpoints on the existing pod before outcomes."""
import hashlib
import json
from pathlib import Path
import time


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    base=Path('/workspace/praxis')
    root=base/'runs/independent-baseline-endpoints-20260915'
    root.mkdir(exist_ok=False)
    panel=base/'data/independent-visual-inventory-20260915/development-panel.json'
    assert sha(panel)=='ede6cd3af5199f89c7bb806e4adb5a6aadb6b08e01bf5b42d51b4f41e7b89439'
    baseline=base/'runs/ordinary-baseline-20260914'
    audit=baseline/'completion-audit/report.json'
    relative='checkpoints/global_step_32/actor/model_world_size_1_rank_0.pt'
    final=baseline/relative
    expected=json.loads(audit.read_text())['files'][relative]['sha256']
    original=json.loads((base/'runs/production-h4-preparation/visual-config.json').read_text())
    cfg={k:original[k] for k in ['model','answer_parser','max_pixels','max_new_tokens','system_prompt']}
    coordinates=base/'runs/candidate-input-capture-20260915-002/export-receipt/byte-exact/coordinates.json'
    source=Path('/workspace/praxis-endpoint-code/Praxis-Extension-main')
    models=sorted(p for p in Path(cfg['model']).rglob('*') if p.is_file())
    files=set(models+[panel,coordinates,final,audit,baseline/'config.yaml',base/'research/production_reward_contract_v3.json',Path(__file__).resolve()])
    files.update(source.rglob('*.py'))
    hashes={}
    for p in sorted(files):
        hashes[str(p)]=sha(p)
        if p==final and hashes[str(p)]!=expected:raise ValueError('Final checkpoint audit mismatch')
    plan=dict(panel=str(panel),coordinates=str(coordinates),model_files=[str(p) for p in models],artifact_sha256=hashes,
        visual_config=cfg,seed=20260915,endpoints={'initial':{'kind':'pinned_pretrained'},'final':{'kind':'checkpoint','path':str(final)}},
        scope='Independent development baseline; 256 correct images and 32 fixed shuffled controls; not final test or selection evidence',
        created_wall=time.time(),source_root=str(source),budget={'seconds_per_endpoint':7200,'primary_responses':576,'terminal_replays':2,'resume_replays':'one per resumed attempt'},
        limitations=['Panel frozen after training before endpoint outcomes','Assistant reviewed images; scene proxies','Single adapted training trajectory'])
    with (root/'plan.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps(dict(plan=str(root/'plan.json'),sha256=sha(root/'plan.json'),artifacts=len(hashes))))

if __name__=='__main__':main()
