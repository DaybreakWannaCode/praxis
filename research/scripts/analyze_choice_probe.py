"""Read-only descriptive analysis of a completed choice surrogate development run."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Measurement is incomplete')
    summary = json.loads((root/'summary.json').read_text())
    parent = json.loads((root/'parent.json').read_text())
    candidates = summary['results']
    ids = [r['id'] for r in parent['dev']]
    groups = list(dict.fromkeys(r['group_id'] for r in parent['dev']))
    deltas = []
    for c in candidates:
        if [r['id'] for r in c['dev']] != ids:
            raise ValueError('Development image order differs')
        deltas.append([r['log_q']-p['log_q'] for r,p in zip(c['dev'], parent['dev'])])
    changes = np.array(deltas)
    # Cluster-resample scenes, retaining all questions within each sampled scene.
    rng = np.random.default_rng(20260914)
    indices = [np.array([i for i,r in enumerate(parent['dev']) if r['group_id']==g]) for g in groups]
    boots = np.empty((2000, len(candidates)))
    for b in range(len(boots)):
        chosen = np.concatenate([indices[i] for i in rng.integers(len(groups), size=len(groups))])
        boots[b] = changes[:, chosen].mean(axis=1)
    intervals = np.quantile(boots, [.025, .975], axis=0).T
    rows = []
    for i,c in enumerate(candidates):
        rows.append(dict(candidate=i, alignment=c['alignment'], direct_lookahead=c['direct_lookahead'],
                         cosine=c['cosine'], update_norm=c['update_norm'], dev_change=c['dev_change'],
                         exploratory_scene_bootstrap_95=intervals[i].tolist()))
    orders = {key:sorted(range(len(rows)), key=lambda i:rows[i][key], reverse=True)
              for key in ('alignment','direct_lookahead','cosine','update_norm','dev_change')}
    return dict(candidates=rows, descending_orders=orders,
                cost={k:summary[k] for k in ('parent_forward_seconds','gradient_seconds','elapsed_seconds','peak_cuda_bytes')},
                interpretation='Four dependent candidates, one parent, mixed update hardware. Bootstrap describes scene sensitivity on an availability-selected panel; no population or accuracy-transfer claim.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('measurement', type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.measurement), indent=2, allow_nan=False))
