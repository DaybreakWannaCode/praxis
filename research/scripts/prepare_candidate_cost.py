"""Prepare, but never launch, one cold-process candidate cost benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from instrument_baseline import instrument
from transfer_alignment.prepare import parse_text, normalize, file_hash


def prepare(args):
    baseline = yaml.safe_load(args.baseline_config.read_text())
    audit = json.loads(args.baseline_audit.read_text())
    if file_hash(args.source) != audit['source_sha256']:
        raise ValueError('Released source changed since baseline audit')
    if file_hash(args.baseline_config) != args.baseline_config_sha256:
        raise ValueError('Unexpected baseline configuration')
    # Baseline eligibility checks already exclude exact visual-panel overlaps.
    excluded = set(audit['train_indices'] + audit['val_indices'])
    excluded.update(row['index'] for row in audit['exclusions'])
    ranked = []
    for index, row in enumerate(pq.read_table(args.source).to_pylist()):
        if index in excluded:
            continue
        item = parse_text(row, index)
        order = hashlib.sha256(('candidate-cost-20260914:'+normalize(item.situation)).encode()).hexdigest()
        ranked.append((order, index, row['problem'], item.answer))
    selected = sorted(ranked)[:32]
    if len(selected) != 32:
        raise ValueError('Need 32 eligible prompts disjoint from ordinary baseline')
    args.data_output.mkdir(parents=True, exist_ok=False)
    train = args.data_output/'train.parquet'
    pq.write_table(pa.Table.from_pylist([dict(problem=p, answer=a) for _,_,p,a in selected]), train)
    preparation = dict(scope='single fresh candidate cost only; no selection',
                       source_sha256=audit['source_sha256'],
                       baseline_audit_sha256=file_hash(args.baseline_audit),
                       indices=[r[1] for r in selected], train_sha256=file_hash(train),
                       count=32, retained_choice_only_suffix=True,
                       limitations='Baseline exact overlap checks only; no new scene audit')
    (args.data_output/'audit.json').write_text(json.dumps(preparation, indent=2))
    baseline['data']['train_files'] = str(train)
    # A text validation pass is part of a cold original-trainer invocation.
    # It gets its own phase timer, and is not charged as a candidate update.
    baseline['trainer'].update(max_steps=1, total_episodes=1, save_freq=1, save_limit=1,
                               load_checkpoint_path=str(args.parent),
                               save_checkpoint_path=str(args.run_output/'exports'),
                               experiment_name='single-candidate-cost-20260914')
    config_path = args.data_output/'config.yaml'
    config_path.write_text(yaml.safe_dump(baseline, sort_keys=False))
    manifest = instrument(args.original_source, args.instrumented_source)
    additions = {
        'verl/trainer/ray_trainer.py': '\nfrom transfer_alignment.candidate_throughput import install_trainer\ninstall_trainer(RayPPOTrainer)\n',
        'verl/workers/fsdp_workers.py': '\nfrom transfer_alignment.candidate_throughput import install_checkpoint_manager\ninstall_checkpoint_manager(FSDPCheckpointManager)\n'}
    for relative, text in additions.items():
        file = args.instrumented_source/relative
        file.write_text(file.read_text()+text)
        manifest[relative] = dict(original=file_hash(args.original_source/relative), observed=file_hash(file))
    (args.instrumented_source/'telemetry-source-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(config=str(config_path), config_sha256=file_hash(config_path),
                         source=str(args.instrumented_source), data=str(args.data_output),
                         launch_status='NOT LAUNCHED'), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-config','baseline-audit','source','data-output','original-source',
                 'instrumented-source','parent','run-output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--baseline-config-sha256', required=True)
    prepare(p.parse_args())
