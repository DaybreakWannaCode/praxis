"""Recompute conditional execution estimates from privately archived measurements.

Does not launch jobs. Pass the completed endpoint summary when available.
"""
import argparse
import hashlib
import json
from pathlib import Path


def estimate(root, endpoint=None):
    sources = {}
    def read(relative):
        p = root / relative
        b = p.read_bytes()
        sources[str(relative)] = hashlib.sha256(b).hexdigest()
        return json.loads(b)
    cand = Path('runs/candidate-cost-20260914-001/completed')
    launcher = read(cand / 'launcher.json')
    export = read(cand / 'cost.json')
    restore = read(cand / 'parent-restore.json')
    baseline = read(Path('runs/ordinary-baseline-20260914/completed/launcher.json'))
    audit = read(Path('runs/ordinary-baseline-20260914/completed/checkpoint-audit.json'))
    probe = read(Path('runs/choice-probe-dev-20260914-002/completed/summary.json'))
    old = read(Path('runs/greedy-throughput-20260914-001/completed/summary.json'))
    assert launcher['status'] == 'complete' and launcher['exit'] == 0
    assert not launcher['tagged_processes_remaining']
    assert export['child_reconstruction_exact'] and export['update_norm'] > 0
    assert audit['status'] == 'passed'
    avg_load = sum(r['load_dot_seconds'] for r in probe['results']) / len(probe['results'])
    avg_forward = sum(r['score_seconds'] for r in probe['results']) / len(probe['results'])
    # The measured residual contains setup, parent norm, final restoration and
    # orchestration. Include once per pool; do not add the enclosing elapsed again.
    shared = probe['elapsed_seconds'] - sum(
        r['load_dot_seconds'] + r['score_seconds'] + r['dev_seconds']
        for r in probe['results'])
    evaluations = {'old_parent': old}
    command_overhead = 0.
    if endpoint:
        evaluations['ordinary_endpoint'] = read(Path(endpoint))
        timing = read(Path(endpoint).with_name('command-timing.json'))
        command_overhead = timing['command_seconds'] - evaluations['ordinary_endpoint']['elapsed_seconds']
        assert command_overhead >= 0
    rate = 1.59  # frozen live receipt; not a claim of a current market quote
    scenarios = []
    for pools in [1, 3]:
        for profile, measurement in evaluations.items():
            per_image = measurement['decode_seconds'] / measurement['count']
            # One exact greedy replay per model is part of our checking protocol.
            for test_images in [256, 1000]:
                final = 10 * (measurement['startup_seconds'] + (test_images + 1) * per_image)
                development = 2 * (measurement['startup_seconds'] + (256 + 32 + 1) * per_image)
                components = {
                    'candidate_construction': pools * 128 * launcher['elapsed_seconds'],
                    'candidate_visual_scoring': pools * (shared + 128 * (avg_load + avg_forward)),
                    'nine_warm_training_runs': 9 * (baseline['elapsed_seconds'] + restore['seconds']),
                    'nine_checkpoint_audits': 9 * audit['elapsed_seconds'],
                    'final_test_parent_plus_nine_models': final,
                    'ordinary_before_after_dev_and_shuffle': development,
                    'evaluation_command_overhead': 12 * command_overhead,
                }
                hours = sum(components.values()) / 3600
                scenarios.append(dict(pools=pools, candidates=128*pools,
                    evaluation_profile=profile, final_test_images=test_images,
                    evaluated_responses=10*test_images + 576,
                    replay_responses=12, model_loads=12,
                    seconds=components, gpu_allocated_hours=hours,
                    compute_usd=hours*rate, contingency_25pct_hours=hours*1.25,
                    contingency_25pct_usd=hours*rate*1.25))
    return dict(status='conditional estimate, not a launch authorization',
        endpoint_measured=bool(endpoint), evaluation_command_overhead_seconds=command_overhead, frozen_compute_usd_per_hour=rate,
        source_sha256=sources,
        shared_visual_setup_gradient_and_restore_seconds=shared,
        mean_archived_load_check_dot_apply_seconds=avg_load,
        mean_archived_direct_score_seconds=avg_forward,
        candidate_storage_128_bytes=128*export['export_bytes'],
        nine_checkpoint_pairs_bytes=9*2*41272696651,
        existing_pre_candidate_storage_bytes=210493066752,
        engineering_person_hours=[12,24],
        qualifications=[
            'Single fresh candidate; pool linear extrapolation has no confidence interval.',
            'Visual scoring rates use archived H4 deltas and an older parent; fresh H1 scoring integration remains to implement.',
            'Nine runs use complete measured 32-step baseline plus real warm restore. Response length and throughput may change.',
            'Audit extrapolation assumes each run retains two full checkpoints.',
            '256 test scenes are a proposed inventory scenario, not a frozen downloaded split. 1000 is scaling only and may be infeasible.',
            'GPU-allocated time includes CPU, offload and I/O. Excludes engineering, idle gaps, storage fees and external benchmarks.',
            '25 percent contingency and engineering hours are planning allowances, not measured uncertainty bounds.',
        ], scenarios=scenarios)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--endpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = estimate(a.root, a.endpoint)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps([{k:s[k] for k in ('pools','evaluation_profile','final_test_images','gpu_allocated_hours','compute_usd')} for s in result['scenarios']], indent=2))
