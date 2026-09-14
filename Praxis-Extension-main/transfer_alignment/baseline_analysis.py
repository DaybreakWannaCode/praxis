"""Audit complete independent endpoints and apply the prospectively fixed analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def endpoint(root, name, plan_sha, items):
    directory=root/name
    if (root/(name+'.exit')).read_text().strip()!='0':raise ValueError('Endpoint did not exit successfully')
    summary=json.loads((directory/'summary.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    for obj in [summary,manifest]:
        if obj['status']!='complete' or obj['endpoint']!=name or obj['plan_sha256']!=plan_sha:
            raise ValueError('Endpoint identity or completion differs')
    if not summary['greedy_replay_exact'] or summary['count']!=288:raise ValueError('Missing replay or responses')
    expected={f'{i:04d}.json' for i in range(288)}
    if {p.name for p in (directory/'responses').glob('*.json')}!=expected:raise ValueError('Response coverage differs')
    records=[];digests={}
    for i,item in enumerate(items):
        p=directory/'responses'/f'{i:04d}.json';r=json.loads(p.read_text())
        kind='correct_image' if i<256 else 'shuffled_image'
        if (r['item_id'],r['group_id'],r['split'],r['kind'],r['plan_sha256'])!=(item['id'],item['group_id'],'dev',kind,plan_sha):
            raise ValueError('Response item identity differs')
        correct=float(bool(r['parsed']) and r['answer']==item['answer'])
        if r['correct']!=correct:raise ValueError('Saved correctness disagrees with frozen label')
        if not isinstance(r['parsed'],bool) or not isinstance(r['truncated'],bool):raise ValueError('Invalid response flags')
        if not isinstance(r.get('sequence_token_ids'),list) or not r['sequence_token_ids']:raise ValueError('Missing archived tokens')
        records.append(r);digests[str(p.relative_to(root))]=sha(p)
    for kind in ['correct_image','shuffled_image']:
        rows=[r for r in records if r['kind']==kind]
        counts=dict(count=len(rows),correct=sum(r['correct'] for r in rows),parsed=sum(r['parsed'] for r in rows),truncated=sum(r['truncated'] for r in rows))
        if counts!=summary[kind]:raise ValueError('Summary disagrees with archived responses')
    return records,digests


def paired(before,after,draws=10000,seed=20260915):
    if len(before)!=len(after) or not before:raise ValueError('Unpaired outcomes')
    delta=[b-a for a,b in zip(before,after)]
    if any(x not in (0,1) for x in before+after):raise ValueError('Expected binary correctness')
    rng=random.Random(seed);n=len(delta)
    boot=sorted(sum(rng.choices(delta,k=n))/n for _ in range(draws))
    def quantile(p):
        position=(len(boot)-1)*p;lower=int(position);upper=min(lower+1,len(boot)-1)
        return boot[lower]+(position-lower)*(boot[upper]-boot[lower])
    transitions={f'{a}_to_{b}':sum(x==a and y==b for x,y in zip(before,after)) for a in (0,1) for b in (0,1)}
    return dict(n=n,initial_accuracy=sum(before)/n,final_accuracy=sum(after)/n,accuracy_change=sum(delta)/n,
        transitions=transitions,paired_image_bootstrap_95=[quantile(.025),quantile(.975)],bootstrap_draws=draws,bootstrap_seed=seed)


def analyze(root,panel_path):
    plan_path=root/'plan.json';plan=json.loads(plan_path.read_text());identity=sha(plan_path)
    if sha(panel_path)!=plan['artifact_sha256'][plan['panel']]:raise ValueError('Frozen panel backup differs')
    panel=json.loads(panel_path.read_text());items=panel['items']+panel['shuffled_controls']
    if len(panel['items'])!=256 or len(panel['shuffled_controls'])!=32:raise ValueError('Wrong panel scope')
    initial,first=endpoint(root,'initial',identity,items);final,last=endpoint(root,'final',identity,items)
    result=paired([r['correct'] for r in initial[:256]],[r['correct'] for r in final[:256]])
    controls={}
    ids={item['id']:i for i,item in enumerate(panel['items'])}
    for name,rows in [('initial',initial),('final',final)]:
        a=[];b=[]
        for control,r in zip(panel['shuffled_controls'],rows[256:]):
            source_id=control['id'].removesuffix('-shuffled')
            if source_id not in panel['shuffle_mapping']:raise ValueError('Control outside locked mapping')
            a.append(rows[ids[source_id]]['correct']);b.append(r['correct'])
        controls[name]=dict(n=32,correct_image_accuracy=sum(a)/32,shuffled_accuracy=sum(b)/32,image_gap=(sum(a)-sum(b))/32)
    controls['image_gap_change']=controls['final']['image_gap']-controls['initial']['image_gap']
    diagnostics={name:{kind:{flag:sum(r[flag] for r in rows if r['kind']==kind) for flag in ['parsed','truncated']} for kind in ['correct_image','shuffled_image']} for name,rows in [('initial',initial),('final',final)]}
    return dict(status='audited',plan_sha256=identity,primary=result,image_controls=controls,diagnostics=diagnostics,
        response_sha256={**first,**last},scope=plan['scope'],
        limitations=['Conditional development-panel image bootstrap; not training-seed uncertainty','Scene independence not established','One adapted training trajectory; no selection intervention result'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=analyze(a.root,a.panel)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='response_sha256'},indent=2))
