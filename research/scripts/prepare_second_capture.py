"""Prepare a second disjoint text batch; never launch or alter live snapshots."""
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transfer_alignment.prepare import parse_text, normalize, file_hash


def main():
    base = Path('/workspace/praxis')
    first = base / 'data/candidate-input-capture-20260915'
    plan = json.loads((first / 'plan.json').read_text())
    for name, expected in plan['files'].items():
        if file_hash(Path(name)) != expected:
            raise ValueError('Frozen source changed: ' + name)
    source = base / 'data/sources/train.parquet'
    baseline_path = base / 'data/ordinary-baseline-20260914/audit.json'
    prior_path = base / 'data/candidate-cost-20260914/audit.json'
    baseline = json.loads(baseline_path.read_text())
    prior = json.loads(prior_path.read_text())
    if file_hash(source) != baseline['source_sha256'] or prior['source_sha256'] != baseline['source_sha256']:
        raise ValueError('Released source identity differs')
    excluded = set(baseline['train_indices'] + baseline['val_indices'] + prior['indices'])
    excluded.update(r['index'] for r in baseline['exclusions'])
    rows = pq.read_table(source).to_pylist()
    forbidden = {normalize(parse_text(rows[i], i).situation)
                 for i in baseline['train_indices'] + baseline['val_indices'] + prior['indices']}
    eligible = []
    for index, row in enumerate(rows):
        if index in excluded:
            continue
        item = parse_text(row, index)
        key = normalize(item.situation)
        if key in forbidden:
            continue
        order = hashlib.sha256(('candidate-cost-20260914:' + key).encode()).hexdigest()
        eligible.append((order, index, key, row['problem'], item.answer))
    selected = sorted(eligible)[:32]
    if len(selected) != 32 or len({r[2] for r in selected}) != 32:
        raise ValueError('Need 32 distinct eligible situations')
    output = base / 'data/candidate-input-capture-20260915-002'
    output.mkdir(exist_ok=False)
    train = output / 'train.parquet'
    pq.write_table(pa.Table.from_pylist([dict(problem=r[3], answer=r[4]) for r in selected]), train)
    config = yaml.safe_load((first / 'config.yaml').read_text())
    scratch = Path('/tmp/praxis-input-capture-20260915-002')
    config['data']['train_files'] = str(train)
    config['trainer']['save_checkpoint_path'] = str(scratch / 'exports')
    config['trainer']['experiment_name'] = 'candidate-input-capture-20260915-002'
    config_path = output / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    audit = dict(status='passed', scope='Second sequential integration candidate, not a selection study',
                 source_sha256=file_hash(source), baseline_audit_sha256=file_hash(baseline_path),
                 prior_candidate_audit_sha256=file_hash(prior_path), indices=[r[1] for r in selected],
                 count=32, train_sha256=file_hash(train), normalized_scene_overlap=0,
                 selection='Next 32 by fixed source hash order after prior/baseline exclusions; no visual outcome used',
                 limitations='Exact normalized scene checks; no paraphrase or independent scene-level audit')
    audit_path = output / 'audit.json'
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    plan.update(scope=audit['scope'], run=str(base / 'runs/candidate-input-capture-20260915-002'),
                scratch=str(scratch), config=str(config_path), config_sha256=file_hash(config_path),
                prerequisite_cleanup=str(base / 'runs/candidate-input-capture-20260915-001/cleanup.json'))
    for path in [train, config_path, audit_path, source, baseline_path, prior_path, Path(config['data']['val_files'])]:
        plan['files'][str(path)] = file_hash(path)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(status='prepared_not_launched', data=str(output), indices=audit['indices']), indent=2))


if __name__ == '__main__':
    main()
