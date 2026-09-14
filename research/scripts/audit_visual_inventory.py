"""Outcome-blind annotation eligibility audit; no images fetched or models run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq
from transfer_alignment.prepare import parse_text, normalize, file_hash

QUESTION = 'Given the situation, which of the following actions is the most appropriate?'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('/workspace/praxis'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    text_path = base / 'data/sources/train.parquet'
    annotation_path = base / 'data/sources/VIVA_annotation.json'
    old_path = base / 'data/production-precision/visual-manifest.json'
    baseline_path = base / 'data/ordinary-baseline-20260914/audit.json'
    baseline = json.loads(baseline_path.read_text())
    if file_hash(text_path) != baseline['source_sha256']:
        raise ValueError('Text source differs from baseline audit')
    text_rows = pq.read_table(text_path).to_pylist()
    all_scenes, all_qa, trained_scenes, trained_qa = set(), set(), set(), set()
    baseline_indices = set(baseline['train_indices'] + baseline['val_indices'])
    invalid_text = []
    for index, row in enumerate(text_rows):
        try:
            item = parse_text(row, index)
        except (ValueError, TypeError) as exc:
            invalid_text.append(dict(index=index, reason=str(exc)))
            continue
        scene = normalize(item.situation)
        qa = normalize(item.question + '\n' + '\n'.join(item.action_list))
        all_scenes.add(scene); all_qa.add(qa)
        if index in baseline_indices:
            trained_scenes.add(scene); trained_qa.add(qa)
    old = json.loads(old_path.read_text())
    old_ids = {int(r['id'].split('-')[-1]) for r in old if r['id'].startswith('viva-')}
    old_scenes = {normalize(r['situation']) for r in old if r.get('image_path')}
    annotations = json.loads(annotation_path.read_text())
    old_urls = {r['image_url'] for r in annotations if int(r['index']) in old_ids}
    blocked = Counter(); eligible = []; seen_urls = set(); seen_scenes = set()
    for row in sorted(annotations, key=lambda r: int(r['index'])):
        index = int(row['index']); raw_url = row.get('image_url', '')
        url = raw_url if isinstance(raw_url, str) else ''
        description = row.get('situation_description')
        scene = normalize(description) if isinstance(description, str) else ''
        raw_options = row.get('action_list')
        options = raw_options if isinstance(raw_options, list) and all(isinstance(x, str) for x in raw_options) else []
        qa = normalize(QUESTION + '\n' + '\n'.join(options))
        reasons = []
        if index in old_ids or url in old_urls or scene in old_scenes: reasons.append('existing_visual_panel')
        if scene in trained_scenes or qa in trained_qa: reasons.append('baseline_text_overlap')
        if scene in all_scenes or qa in all_qa: reasons.append('full_text_source_overlap')
        if not scene or len(options) < 2 or row.get('answer') not in [x.split('.')[0].strip() for x in options]: reasons.append('invalid_annotation')
        if urlsplit(url).scheme != 'https' or not urlsplit(url).hostname: reasons.append('invalid_https_url')
        if url in seen_urls or scene in seen_scenes: reasons.append('duplicate_url_or_description')
        # Deduplication is conservative across all annotations, including excluded rows.
        seen_urls.add(url); seen_scenes.add(scene)
        if reasons:
            blocked.update(reasons)
            continue
        rank = hashlib.sha256(('independent-visual-inventory-20260915:' + str(index)).encode()).hexdigest()
        eligible.append(dict(index=index, image_url=url, description_sha256=hashlib.sha256(scene.encode()).hexdigest(), rank=rank))
    eligible.sort(key=lambda r: r['rank'])
    report = dict(status='annotation_audit_complete', raw_annotations=len(annotations),
                  existing_visual_ids=len(old_ids), text_rows=len(text_rows), invalid_text_rows=invalid_text,
                  exclusion_counts=dict(blocked), eligible_annotation_rows=len(eligible), eligible=eligible,
                  source_sha256={str(p):file_hash(p) for p in [text_path, annotation_path, old_path, baseline_path]},
                  limits='No downloads, content hashes, perceptual duplicates, scene identity review, split freeze, or outcome evaluation. Counts are eligible annotations, not verified independent images. Exclusion reasons may overlap.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('eligible','source_sha256','invalid_text_rows')}, indent=2))


if __name__ == '__main__':
    main()
