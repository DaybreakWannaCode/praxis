"""Prepare a small engineering manifest from local, revision-pinned source files.

No model is loaded and no network request is made. Scientific splits still need
scene/source audit beyond this exact-content duplicate check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

from .data import Item, validate_items


def normalize(text):
    return " ".join(text.lower().split())


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_text(row, index):
    if any(image is not None for image in (row.get("images") or [])):
        raise ValueError(f"Text row {index}: contains an actual image")
    # The current released dataset uses `problem`, not the launcher's `question`.
    prompt = row.get("problem", row.get("question"))
    if not isinstance(prompt, str):
        raise ValueError(f"Text row {index}: missing problem/question string")
    match = re.fullmatch(r".*?## Situation:\s*(.*?)\s*## Question:\s*(.*?)\s*Now answer the question\..*",
                         prompt, re.DOTALL)
    if not match:
        raise ValueError(f"Text row {index}: unrecognized prompt template")
    situation, body = match.groups()
    starts = list(re.finditer(r"(?m)^\s*([A-Z])\.\s+", body))
    if len(starts) < 2:
        raise ValueError(f"Text row {index}: missing multiple-choice options")
    question = body[:starts[0].start()].strip()
    options = [body[m.start():starts[j+1].start() if j+1<len(starts) else len(body)].strip()
               for j, m in enumerate(starts)]
    answer = str(row.get("answer", "")).strip().upper()
    # Do not pass assistant messages, rationales or gold answers into the prompt.
    key = hashlib.sha256(normalize(situation).encode()).hexdigest()
    return Item(id=f"praxis-train-{index}", group_id=f"text-{key}", split="train",
                question=question, action_list=options, answer=answer, situation=situation)


def read_text_rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path) as f:
        return json.load(f)


def prepare(text_path, viva_path, image_dir, output, *, text_revision, viva_revision,
            seed=20260906, train_n=2, score_n=2, dev_n=2):
    for value in (text_revision, viva_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("Dataset revisions must be full commit SHAs")
    if min(train_n, score_n, dev_n) < 1 or max(train_n, score_n, dev_n) > 32:
        raise ValueError("This helper is for 1-32 examples per engineering split")
    text_path, viva_path, image_dir, output = map(Path, (text_path, viva_path, image_dir, output))
    if output.exists():
        raise FileExistsError(output)
    raw_text = read_text_rows(text_path)
    raw_viva = json.loads(viva_path.read_text())
    rng = random.Random(seed)
    indices = list(range(len(raw_text)))
    rng.shuffle(indices)
    selected, seen_text = [], set()
    for idx in indices:
        item = parse_text(raw_text[idx], idx)
        if item.group_id in seen_text:
            continue
        selected.append(item)
        seen_text.add(item.group_id)
        if len(selected) == train_n:
            break
    if len(selected) != train_n:
        raise ValueError("Not enough distinct training situations")
    raw_viva = sorted(raw_viva, key=lambda r: int(r["index"]))
    rng.shuffle(raw_viva)
    descriptions = {normalize(i.situation) for i in selected}
    image_hashes, source_urls, chosen_images, rejected = set(), set(), {}, []
    for row in raw_viva:
        index = row["index"]
        name = row.get("image_file", f"{index}.jpg")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Unsafe image filename in source annotations")
        path = image_dir/name
        description = row.get("situation_description")
        if not isinstance(description, str) or not description.strip():
            rejected.append({"index": index, "reason": "missing_description"})
            continue
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.load()
                if min(image.size) < 48:
                    raise ValueError("Image too small")
            ih = file_hash(path)
        except (OSError, ValueError):
            rejected.append({"index": index, "reason": "missing_or_invalid_image"})
            continue
        url = row.get("image_url")
        url = url if isinstance(url, str) else ""
        desc = normalize(description)
        if ih in image_hashes or desc in descriptions or (url and url in source_urls):
            rejected.append({"index": index, "reason": "exact_duplicate"})
            continue
        visual_count = len(selected)-train_n
        split = "score" if visual_count < score_n else "dev"
        selected.append(Item(id=f"viva-{index}", group_id=f"viva-image-{ih}", split=split,
                             question="Given the situation, which of the following actions is the most appropriate?",
                             action_list=row["action_list"], answer=str(row["answer"]).strip().upper(),
                             situation=description, image_path=str(path.resolve())))
        image_hashes.add(ih)
        descriptions.add(desc)
        source_urls.add(url)
        chosen_images[f"viva-{index}"] = ih
        if len(selected) == train_n+score_n+dev_n:
            break
    if len(selected) != train_n+score_n+dev_n:
        raise ValueError("Insufficient usable independent images; supply more images, do not reuse scenes across splits")
    validate_items(selected, real_images=True)
    output.mkdir(parents=True)
    (output/"manifest.json").write_text(json.dumps([asdict(i) for i in selected], indent=2)+"\n")
    (output/"provenance.json").write_text(json.dumps({
        "purpose": "engineering_only_not_final_paper_split", "seed": seed,
        "text_repo": "zhehuderek/textual_decisionmaking_data", "text_revision": text_revision,
        "text_file_sha256": file_hash(text_path),
        "viva_repo": "zhehuderek/VIVA_Benchmark_EMNLP24", "viva_revision": viva_revision,
        "viva_file_sha256": file_hash(viva_path), "images_sha256": chosen_images,
        "rejected": rejected,
        "remaining_audit": "Semantic near-duplicates, scene families and source-video overlap require manual audit",
    }, indent=2)+"\n")
    return output/"manifest.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-file", type=Path, required=True)
    p.add_argument("--viva-file", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--text-revision", required=True)
    p.add_argument("--viva-revision", required=True)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--train-n", type=int, default=2)
    p.add_argument("--score-n", type=int, default=2)
    p.add_argument("--dev-n", type=int, default=2)
    a = p.parse_args()
    print(prepare(a.text_file, a.viva_file, a.image_dir, a.output,
                  text_revision=a.text_revision, viva_revision=a.viva_revision,
                  seed=a.seed, train_n=a.train_n, score_n=a.score_n, dev_n=a.dev_n))


if __name__ == "__main__":
    main()
