"""Verify acquired image identities and flag duplicates before panel review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps


def features(path):
    with Image.open(path) as image:
        image.load(); rgb = ImageOps.exif_transpose(image).convert('RGB')
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        values = list(rgb.convert('L').resize((9, 8)).getdata())
        bits = [values[y*9+x] > values[y*9+x+1] for y in range(8) for x in range(8)]
        dhash = sum(int(bit) << j for j, bit in enumerate(bits))
        return pixels, dhash


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--old-manifest', type=Path, required=True); args = parser.parse_args()
    root = args.root
    plan = json.loads((root/'download-plan.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if summary['status'] != 'acquisition_complete' or (root/'acquisition.exit').read_text().strip() != '0':
        raise ValueError('Acquisition is not complete')
    if summary['plan_sha256'] != hashlib.sha256((root/'download-plan.json').read_bytes()).hexdigest():
        raise ValueError('Acquisition plan changed')
    old = json.loads(args.old_manifest.read_text())
    expected = json.loads(Path(plan['annotation_audit']).read_text())['source_sha256'][str(args.old_manifest)]
    if hashlib.sha256(args.old_manifest.read_bytes()).hexdigest() != expected:
        raise ValueError('Original visual panel identity changed')
    entries = []; failures = Counter(); records = {}; total = 0
    for item in old:
        if not item.get('image_path'): continue
        pixel, perceptual = features(Path(item['image_path']))
        entries.append(dict(id=item['id'], old=True, pixel=pixel, dhash=perceptual))
    for row in plan['candidates']:
        index = row['index']; receipt = root/'receipts'/f'{index}.json'
        data = json.loads(receipt.read_text())
        if data['index'] != index or data['source_url'] != row['image_url']:
            raise ValueError('Receipt identity differs')
        records[str(index)] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        if data['status'] != 'downloaded':
            failures[data.get('error','unknown').split(':')[0]] += 1
            continue
        image = root/'images'/f'{index}.image'
        if image.stat().st_size != data['bytes'] or data['bytes'] > plan['per_image_byte_cap']:
            raise ValueError('Image size differs')
        if hashlib.sha256(image.read_bytes()).hexdigest() != data['sha256']:
            raise ValueError('Image bytes changed')
        pixel, perceptual = features(image)
        if pixel != data['pixel_sha256'] or f'{perceptual:016x}' != data['dhash64']:
            raise ValueError('Decoded image identity differs')
        entries.append(dict(id=f'viva-{index}', old=False, pixel=pixel, dhash=perceptual))
        total += data['bytes']
    count = sum(not item['old'] for item in entries)
    if count != summary['downloaded'] or total != summary['downloaded_bytes']:
        raise ValueError('Summary count/size differs')
    pairs = []; flagged = set()
    for j, right in enumerate(entries):
        for left in entries[:j]:
            if left['old'] and right['old']: continue
            distance = bin(left['dhash'] ^ right['dhash']).count('1')
            if left['pixel'] == right['pixel'] or distance <= 6:
                pairs.append(dict(left=left['id'], right=right['id'], old_panel_overlap=left['old'] or right['old'],
                                  exact_pixels=left['pixel']==right['pixel'], dhash_distance=distance))
                flagged.update(r['id'] for r in (left,right) if not r['old'])
    unflagged = [r['id'] for r in entries if not r['old'] and r['id'] not in flagged]
    report = dict(status='content_audit_complete', downloaded=count, downloaded_bytes=total,
                  failure_types=dict(failures), duplicate_review_pairs=pairs, flagged_new_ids=sorted(flagged),
                  unflagged_ids_in_frozen_order=unflagged, receipt_sha256=records,
                  limits='Byte/pixel verification plus dHash<=6 screening only. False positives and missed crops/scenes remain possible. Manual image/annotation review and panel freeze still required; no model outcomes.')
    with (root/'image-audit.json').open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(dict(status=report['status'],downloaded=count,downloaded_bytes=total,
                         duplicate_review_pairs=len(pairs),unflagged=len(unflagged),failure_types=dict(failures)),indent=2))


if __name__ == '__main__': main()
