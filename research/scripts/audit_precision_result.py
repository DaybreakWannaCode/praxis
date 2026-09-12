"""Read-only archive audit; does not certify upstream gradients or training parity.

Reconstructs outcomes from raw responses, checks replay and split inventories,
and recomputes statistics using the production statistical definitions. No GPU,
model loading, sampling, or experiment adaptation is performed.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label)
        for key in expected:
            compare(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{label}[{i}]")
    elif isinstance(expected, float):
        require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10,
                                                      abs_tol=1e-12), label)
    else:
        require(actual == expected, label)


def audit(root):
    root = Path(root)
    def read(name):
        return json.loads((root / name).read_text())
    manifest, summary, sealed = [read(n) for n in
                                ("manifest.json", "summary.json", "scores.json")]
    require(manifest['status'] == summary['status'] == 'complete', 'Incomplete run')
    require((root / 'run.exit').read_text().strip() == '0', 'Nonzero or missing exit')
    cfg = manifest['config']
    require(cfg['frozen'] is True and cfg['horizon'] in (1, 4), 'Not frozen H1/H4')
    require(cfg['answer_parser'] == 'explicit_final_v3', 'Parser contract')
    require(cfg['response_budget'] == 1800, 'Frozen response budget')
    require(len(cfg['score_ids']) == 16 and len(cfg['dev_ids']) == 32, 'Panel sizes')
    require((cfg['probe_repeats'], cfg['probe_samples'], cfg['outcome_samples']) ==
            (4, 4, 8), 'Sampling budget')
    require(cfg['seed'] == (20260916 if cfg['horizon'] == 1 else 20260917), 'Seed')
    ids = cfg['score_ids'] + cfg['dev_ids']
    require(len(ids) == len(set(ids)), 'Overlapping or repeated images')
    rows = [json.loads(line) for line in (root / 'responses.jsonl').read_text().splitlines()]
    require(len(rows) == cfg['response_budget'], 'Response count')
    candidates = len(cfg['candidate_gates'])
    require(candidates == 4, 'Candidate count')
    kinds = ['parent', 'null_independent'] + [f'child_{i}' for i in range(candidates)]
    expected = [('probe', r, item, j) for r in range(4) for item in cfg['score_ids']
                for j in range(4)]
    expected += [(kind, None, item, j) for kind in kinds for item in cfg['dev_ids']
                 for j in range(8)]
    expected += [('null_replay', None, cfg['dev_ids'][0], j) for j in range(8)]
    actual = [(r['kind'], r.get('repeat'), r['item_id'], r['sample']) for r in rows]
    require(actual == expected, 'Ordered response inventory')
    indexed = dict(zip(actual, rows))
    groups = {'score': set(), 'dev': set()}
    for r in rows:
        split = 'score' if r['kind'] == 'probe' else 'dev'
        require(r['split'] == split, 'Wrong split or final-test record')
        groups[split].add(r['group_id'])
        require(r['correct'] in (0, 1) and isinstance(r['parsed'], bool), 'Reward type')
        require(r['parsed'] or r['correct'] == 0, 'Unparsed response marked correct')
        require(len(r['sequence_token_ids']) == r['prompt_length'] + r['length'],
                'Token inventory')
        if r['kind'].startswith('child_') or r['kind'] == 'null_replay':
            parent = indexed[('parent', None, r['item_id'], r['sample'])]
            require(r['seed'] == parent['seed'], 'Paired seed mismatch')
            if r['kind'] == 'null_replay':
                compare(r['sequence_token_ids'], parent['sequence_token_ids'], 'Replay')
    require(not groups['score'] & groups['dev'], 'Shared scene group across splits')
    outcomes = {}
    for kind in kinds + ['null_replay']:
        selected = [r for r in rows if r['kind'] == kind]
        panel = cfg['dev_ids'][:1] if kind == 'null_replay' else cfg['dev_ids']
        matrix = [[indexed[(kind, None, item, j)]['correct'] for j in range(8)]
                  for item in panel]
        outcomes[kind] = matrix
        parsed = sum(r['parsed'] for r in selected)
        correct = sum(r['correct'] for r in selected)
        reconstructed = dict(correctness=correct / len(selected), matrix=matrix,
                             parse_success=parsed / len(selected),
                             conditional_correctness=correct / parsed if parsed else None,
                             truncations=sum(r['truncated'] for r in selected),
                             responses=len(selected))
        compare(read(f'outcome-{kind}.json'), reconstructed, f'outcome-{kind}')
    source = Path(__file__).resolve().parents[2] / 'Praxis-Extension-main/transfer_alignment/production_statistics.py'
    spec = importlib.util.spec_from_file_location('precision_statistics_audit', source)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    projections = [json.loads(s) for s in (root / 'image-projections.jsonl').read_text().splitlines()]
    require([(r['repeat'], r['item_id']) for r in projections] ==
            [(r, i) for r in range(4) for i in cfg['score_ids']], 'Projection inventory')
    x = np.asarray([r['alignment'] for r in projections]).reshape(4, 16, 4)
    require(np.isfinite(x).all(), 'Nonfinite projections')
    projection_residuals = []
    for repeat in range(4):
        for candidate in range(4):
            score = sealed['replicates'][repeat]['scores'][candidate]
            # The saved aggregate uses an FP32 gradient sum; per-image dots
            # are reduced separately. Cancellation makes relative-to-mean
            # tolerances misleading. This is a numerical diagnostic, not an
            # exact floating-point identity or a proof of gradient validity.
            mean = float(x[repeat, :, candidate].mean())
            residual = abs(score['alignment'] - mean)
            tolerance = 64 * float(np.finfo(np.float32).eps) * float(
                np.abs(x[repeat, :, candidate]).mean()) + 1e-12
            require(residual <= tolerance, 'Aggregate/per-image projection discrepancy')
            projection_residuals.append(dict(repeat=repeat, candidate=candidate,
                                            absolute_residual=residual, tolerance=tolerance))
            require(score['update_norm'] > 0 and score['visual_gradient_norm'] >= 0, 'Norms')
            denom = score['update_norm'] * score['visual_gradient_norm']
            compare(score['cosine'], score['alignment'] / denom if denom else None, 'Cosine')
    compare(sealed['pairwise'], stats.alignment_contrasts(x), 'Score intervals')
    compare(sealed['bonferroni_pairwise'], stats.alignment_contrasts(x, alpha=.05/6), 'Score simultaneous')
    pairs = [dict(first=a, second=b,
                  pointwise=stats.outcome_contrast(outcomes[f'child_{a}'], outcomes[f'child_{b}']),
                  simultaneous=stats.outcome_contrast(outcomes[f'child_{a}'], outcomes[f'child_{b}'], alpha=.05/6))
             for a, b in itertools.combinations(range(4), 2)]
    compare(summary['candidate_contrasts'], pairs, 'Candidate contrasts')
    compare(summary['parent_contrasts'],
            {k: stats.outcome_contrast(outcomes[k], outcomes['parent']) for k in kinds[1:]},
            'Parent contrasts')
    compare(summary['scores'], sealed, 'Sealed score copy')
    require(sealed['sealed_before_outcomes'] is True, 'Missing seal assertion')
    require(summary['parent_replay_exact'] is True, 'Missing replay assertion')
    decision = stats.precision_decision(sealed['pairwise'], pairs,
        resolution=cfg['decision_rule']['resolution'],
        minimum_pairs=cfg['decision_rule']['minimum_jointly_resolved_pairs'], horizon=cfg['horizon'])
    compare(summary['precision_decision'], decision, 'Decision')
    require(0 < summary['seconds'] <= cfg['wall_time_cap_seconds'] + 30, 'Runtime cap')
    require(summary['max_torch_allocated_bytes'] > 0, 'GPU peak missing')
    require(read('resource-usage.json')['max_child_rss_kib'] > 0, 'Host peak missing')
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   p.name not in ('archive-sha256.json', 'audit.json'))
    return dict(status='passed', horizon=cfg['horizon'], responses=len(rows), decision=decision,
                projection_residuals=projection_residuals,
                file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                limitations='Archive consistency and statistics recomputation only. Upstream parser correctness, '
                'gradient validity, parity, frozen input identity and pre-outcome sealing require separate evidence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.result), indent=2))
