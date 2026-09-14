"""Bounded, resumable acquisition of a frozen development-only image pool."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def acquire(row, root, limit):
    index = row['index']; record = root / 'receipts' / f'{index}.json'
    target = root / 'images' / f'{index}.image'
    if record.exists():
        result = json.loads(record.read_text())
        if result['source_url'] != row['image_url']:
            raise ValueError('Existing receipt source differs')
        if result['status'] == 'downloaded' and sha(target) != result['sha256']:
            raise ValueError('Previously downloaded bytes changed')
        return result
    partial = target.with_suffix('.partial')
    started = time.monotonic()
    result = dict(index=index, source_url=row['image_url'], status='failed')
    try:
        request = Request(row['image_url'], headers={'User-Agent': 'PraxisResearch-ImageAudit/1.0'})
        with urlopen(request, timeout=10) as response, partial.open('wb') as stream:
            if not response.url.startswith('https://'):
                raise ValueError('Non-HTTPS redirect')
            result['resolved_url'] = response.url
            size = 0
            while True:
                if time.monotonic() - started > 30:
                    raise TimeoutError('Per-image wall-time cap')
                block = response.read(min(65536, limit + 1 - size))
                if not block:
                    break
                size += len(block)
                if size > limit:
                    raise ValueError('Per-image byte cap')
                stream.write(block)
            stream.flush(); os.fsync(stream.fileno())
        with Image.open(partial) as image:
            if image.width * image.height > 40_000_000 or min(image.size) < 48:
                raise ValueError('Image dimensions outside audit bounds')
            image.load()
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            tiny = rgb.convert('L').resize((9, 8))
            values = list(tiny.getdata())
            bits = [values[y*9+x] > values[y*9+x+1] for y in range(8) for x in range(8)]
            dhash = sum(int(bit) << j for j, bit in enumerate(bits))
            result.update(width=rgb.width, height=rgb.height, pixel_sha256=pixel_hash,
                          dhash64=f'{dhash:016x}', sha256=sha(partial), bytes=size)
        os.replace(partial, target)
        result['status'] = 'downloaded'
    except Exception as exc:
        if partial.exists(): partial.unlink()
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['elapsed_seconds'] = time.monotonic() - started
    save(record, result)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args(); plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    audit_path = Path(plan['annotation_audit'])
    if sha(audit_path) != plan['annotation_audit_sha256']:
        raise ValueError('Annotation audit changed')
    audit = json.loads(audit_path.read_text())
    expected = audit['eligible'][:512]
    if plan['candidates'] != expected or plan['reserved_indices'] != [r['index'] for r in audit['eligible'][512:]]:
        raise ValueError('Frozen development/reserved partition differs')
    if plan['per_image_byte_cap'] != 2_000_000 or plan['workers'] != 4:
        raise ValueError('Unexpected acquisition budget')
    for name in ['images', 'receipts']: (root / name).mkdir(exist_ok=True)
    results = []; start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(acquire, row, root, plan['per_image_byte_cap']) for row in expected]
        for future in as_completed(futures):
            results.append(future.result())
            save(root / 'progress.json', dict(status='running', completed=len(results),
                 downloaded=sum(r['status']=='downloaded' for r in results)))
    good = [r for r in results if r['status']=='downloaded']
    save(root / 'summary.json', dict(status='acquisition_complete', attempted=len(results), downloaded=len(good),
         downloaded_bytes=sum(r['bytes'] for r in good), elapsed_seconds=time.monotonic()-start,
         plan_sha256=sha(args.plan), limits='Development candidate pool only; duplicate/scene review and final panel freeze still required. No model outcomes.'))
    print((root / 'summary.json').read_text())


if __name__ == '__main__': main()
