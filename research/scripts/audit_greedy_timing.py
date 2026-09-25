"""Verify a copied greedy-timing result against remote hashes and row aggregates."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(root, remote_hashes):
    hashes = json.loads(remote_hashes.read_text())
    for name in ('manifest.json', 'progress.json', 'responses.jsonl', 'summary.json'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f'Archive hash mismatch: {name}')
    manifest = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    progress = json.loads((root/'progress.json').read_text())
    rows = [json.loads(line) for line in (root/'responses.jsonl').read_text().splitlines()]
    assert manifest['status'] == 'complete'
    assert len(rows) == summary['count'] == progress['completed_images'] == 32
    assert len({r['item_id'] for r in rows}) == 32
    assert all(r['split'] == 'dev' and r['kind'] == 'greedy_timing' for r in rows)
    assert summary['greedy_replay_exact']
    for metric, field in [('correct','correct'), ('parsed','parsed'), ('truncated','truncated'), ('response_tokens','length')]:
        assert sum(r[field] for r in rows) == summary[metric]
    decode = sum(r['seconds'] for r in rows)
    assert math.isclose(decode, summary['decode_seconds'], rel_tol=1e-12)
    assert all(math.isfinite(r['seconds']) and r['seconds'] > 0 for r in rows)
    assert summary['elapsed_seconds'] >= summary['startup_seconds'] + decode
    assert manifest['max_new_tokens'] == 512
    return dict(status='passed', scope='Archive hashes and independent aggregate reconstruction; exact greedy replay checked inside the GPU run.',
        source_sha256=hashes, count=len(rows),
        mean_tokens=sum(r['length'] for r in rows)/len(rows),
        min_tokens=min(r['length'] for r in rows), max_tokens=max(r['length'] for r in rows),
        mean_decode_seconds=decode/len(rows),
        endpoint_sha256=manifest.get('endpoint_sha256'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--remote-hashes', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root, a.remote_hashes)
    (a.root/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}, indent=2))
