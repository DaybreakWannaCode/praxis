"""Audit a local compact-score backup before authorizing managed scratch release.

Checks identities and scalar consistency; raw delta bytes are verified by the
scorer, not reverified here. This does not establish independent visual transfer.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(score, receipt):
    read = lambda name: json.loads((score / name).read_text())
    summary = read('summary.json')
    manifest = read('manifest.json')
    projection = read('alignment.json')
    parent = read('parent-score.json')['rows']
    child = read('candidate-score.json')
    delta_path = receipt / 'delta-manifest.json'
    delta = json.loads(delta_path.read_text())
    cost = json.loads((receipt / 'cost.json').read_text())
    require = lambda condition, message: None if condition else fail(message)
    require(summary['status'] == manifest['status'] == cost['status'] == 'complete', 'Incomplete run')
    require(summary['parent_replay_exact'] and cost['child_reconstruction_exact'], 'Reconstruction/replay failed')
    require(summary['new_model_or_gradient_files'] == 0, 'Unexpected large output contract')
    plan = manifest['plan']
    require(plan['parent_model'] == cost['parent_model'], 'Parent identity differs')
    actor = Path(plan['candidate_dir'])
    for relative, name in [('cost.json', 'cost.json'), ('coordinates.json', 'coordinates.json'),
                           ('delta/manifest.json', 'delta-manifest.json')]:
        actual = hashlib.sha256((receipt / name).read_bytes()).hexdigest()
        require(plan['artifact_sha256'][str(actor / relative)] == actual, 'Export identity differs: ' + name)
    require(len(parent) == len(child['rows']) == 16, 'Wrong panel size')
    ids = [r['id'] for r in parent]
    require(len(set(ids)) == 16 and ids == manifest['score_ids'] == [r['id'] for r in child['rows']], 'Panel identity differs')
    for before, after in zip(parent, child['rows']):
        require(before['group_id'] == after['group_id'] and before['tokens'] == after['tokens'], 'Scoring definition differs')
        require(math.isfinite(before['log_q']) and math.isfinite(after['log_q']), 'Nonfinite objective')
    direct = sum(r['log_q'] for r in child['rows']) / 16 - sum(r['log_q'] for r in parent) / 16
    close = lambda a, b: math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    require(close(direct, child['direct_lookahead']) and close(direct, summary['direct_lookahead']), 'Direct change differs')
    require(close(projection['alignment'], child['alignment']) and close(projection['alignment'], summary['alignment']), 'Alignment differs')
    norm = projection['update_norm']
    require(close(norm, delta['update_norm']) and close(norm, cost['update_norm']) and norm > 0, 'Update norm differs')
    require(projection['gradient_norm'] > 0, 'Zero gradient')
    cosine = projection['alignment'] / (norm * projection['gradient_norm'])
    require(close(cosine, projection['cosine']) and close(cosine, child['cosine']) and abs(cosine) <= 1 + 1e-12, 'Cosine differs')
    names = [r['name'] for r in delta['parameters']]
    digests = projection['child_tensor_digests']
    require(len(names) == len(set(names)) == len(digests) == projection['validated_tensors'] == summary['validated_tensors'] == 824, 'Tensor coverage differs')
    require(set(names) == {r['name'] for r in digests}, 'Child tensor coverage differs')
    require(all(len(r['child_fp32_sha256']) == 64 and all(c in '0123456789abcdef' for c in r['child_fp32_sha256']) for r in digests), 'Malformed child digest')
    require(projection['canonical_numel'] == delta['canonical_numel'] == sum(r['numel'] for r in delta['parameters']) == 3754622976, 'Coordinate count differs')
    require(summary['source_delta_bytes'] == cost['export_bytes'] == sum(r['compressed_bytes'] for r in delta['parameters']), 'Export size differs')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in score.iterdir()
             if p.is_file() and p.name != 'local-audit.json'}
    return dict(status='passed', files=files, score_images=16, tensors=824,
                alignment=projection['alignment'], direct_change_recomputed=direct,
                scope='Local backup identities and scalar consistency; raw chunks verified by scorer; calibration only')


def fail(message):
    raise ValueError(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.score, args.receipt)
    (args.score / 'local-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
