"""CPU-only prospective cost and paired-outcome sensitivity; launches nothing."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def assess(root):
    sources = {}
    def read(path):
        raw = (root / path).read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    profiles = []
    for suffix in ('001', '002'):
        base = Path('runs/candidate-input-capture-20260915-' + suffix) / 'completed'
        launch = read(base / 'launcher.json')
        score = read(base / 'scoring/summary.json')
        cleanup = read(base / 'cleanup.json')
        assert launch['exit'] == 0 and launch['status'] == 'complete'
        assert score['status'] == 'complete' and score['parent_replay_exact']
        assert cleanup['status'] == 'complete'
        # Export is nested in launcher: do not add cost.json again.
        profiles.append((suffix, launch['elapsed_seconds'] + score['elapsed_seconds']))
    train = read(Path('runs/ordinary-baseline-20260914/completed/launcher.json'))
    audit = read(Path('runs/ordinary-baseline-20260914/completed/checkpoint-audit.json'))
    restore = read(Path('runs/candidate-cost-20260914-001/completed/parent-restore.json'))
    assert train['exit'] == 0 and audit['status'] == 'passed'
    endpoint = Path('runs/independent-baseline-endpoints-20260915')
    # Full measured 288-image endpoint plus replay, including startup/integrity.
    eval_times = []
    for name in ('initial', 'final'):
        summary = read(endpoint / name / 'summary.json')
        manifest = read(endpoint / name / 'manifest.json')
        assert summary['status'] == 'complete' and summary['greedy_replay_exact']
        assert manifest['status'] == 'complete'
        eval_times.append(sum(a['elapsed_seconds'] for a in manifest['attempts']))
    eval_seconds = max(eval_times)
    scenarios = []
    for pool in (64, 128):
        for arms in (2, 3):
            runs = 3 * arms
            for name, candidate_seconds in profiles:
                parts = {
                    'candidate_construction_and_cold_scoring': pool * candidate_seconds,
                    'fresh_training_including_saves_and_warm_restore': runs * (train['elapsed_seconds'] + restore['seconds']),
                    'checkpoint_audits': runs * audit['elapsed_seconds'],
                    'parent_and_final_endpoints_288_each': (runs + 1) * eval_seconds,
                }
                hours = sum(parts.values()) / 3600
                scenarios.append(dict(pool=pool, selected_batches=32, arms=arms, seeds=3,
                    candidate_profile=name, seconds=parts, hours=hours,
                    compute_at_historical_1_59=hours*1.59,
                    allowance_25pct_hours=hours*1.25))
    # Optional full prediction endpoint: a distinct development panel for each
    # tentative child plus its common parent. Final-model test evaluation above
    # is on another panel and cannot be reused as the prediction parent outcome.
    prediction_scenarios = []
    for scenario in scenarios:
        extra = (scenario['pool'] + 1) * eval_seconds / 3600
        total = scenario['hours'] + extra
        prediction_scenarios.append(dict(
            pool=scenario['pool'], arms=scenario['arms'], seeds=scenario['seeds'],
            candidate_profile=scenario['candidate_profile'],
            independent_candidate_endpoint_hours=extra,
            intervention_plus_prediction_hours=total,
            compute_at_historical_1_59=total*1.59,
            allowance_25pct_compute_at_historical_1_59=total*1.59*1.25))
    # Sensitivity scenarios, not a power guarantee or fitted future discordance.
    sensitivity = []
    for n in (256, 512):
        for q in (.05, .10, .20):
            sensitivity.append(dict(images=n, discordance=q,
                approximate_95pct_halfwidth_pp=100*1.96*math.sqrt(q/n)))
    return dict(source_sha256=sources, scenarios=scenarios,
        paired_null_sensitivity=sensitivity,
        intervention_plus_prediction_scenarios=prediction_scenarios,
        limitations=[
            'Forecast only; no launch authorization. Profiles are two timings, not bounds.',
            'Original scenarios cover intervention only. Independent per-candidate generated outcomes are costed separately.',
            'Prediction endpoint uses 288-answer timing as a proxy; incremental child reconstruction/verification overhead is unmeasured.',
            'Repeated cold scoring includes probe gradient per candidate; no unmeasured parent-reuse speedup.',
            'Backup, cleanup, engineering, idle time and archive transfer are outside measured component boundaries.',
            'Evaluation uses full measured 288-answer jobs as a budgeting proxy, not a locked test-panel design.',
            '512 test images are an inventory scenario; reserved image validity and independence are unverified.',
            'No seed pooling into 3N independent observations; sensitivity is conditional on a single paired comparison.',
            'Training-length and I/O changes can invalidate these projections. Historical price is not a live quote.',
            'Storage retention is bounded for candidates; safe full-training checkpoint rotation is not integrated.',
        ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    for s in result['scenarios']:
        print(s['pool'], s['arms'], s['candidate_profile'], round(s['hours'], 2), round(s['compute_at_historical_1_59'], 2))
    print(result['paired_null_sensitivity'])
