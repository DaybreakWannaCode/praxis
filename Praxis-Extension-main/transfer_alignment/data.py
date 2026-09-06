"""Normalized local inputs with explicit split and scene identity."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from task_0.src.rewards import option_labels


@dataclass
class Item:
    id: str
    group_id: str
    split: str
    question: str
    action_list: list
    answer: str
    situation: str = ""
    image_path: str = ""
    feature: list = field(default_factory=list)  # Synthetic backend only.

    def image(self):
        from PIL import Image
        with Image.open(self.image_path) as im:
            return im.convert("RGB")


def load_manifest(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("Manifest must be a nonempty JSON list")
    rows = []
    for r in raw:
        row = Item(**r)
        if row.image_path:
            row.image_path = str((path.parent / row.image_path).resolve())
        rows.append(row)
    validate_items(rows, real_images=True)
    return rows


def validate_items(items, *, real_images=False):
    ids, groups, content, images = set(), {}, {}, {}
    for item in items:
        if not item.id or item.id in ids:
            raise ValueError("Missing or duplicate item ID")
        ids.add(item.id)
        if not item.group_id or item.split not in {"train", "score", "dev", "test"}:
            raise ValueError(f"Invalid group/split: {item.id}")
        if not item.question.strip() or len(item.action_list) < 2:
            raise ValueError(f"Missing question/options: {item.id}")
        labels = option_labels(item.action_list)
        if len(set(labels)) != len(labels) or item.answer not in labels:
            raise ValueError(f"Invalid option labels/answer: {item.id}")
        if item.split == "train" and not item.situation.strip():
            raise ValueError(f"Training situation missing: {item.id}")
        tokens = []
        if item.situation.strip():
            # Exact normalized description duplicates are conservatively rejected.
            tokens.append((content, " ".join(item.situation.lower().split())))
        tokens.append((groups, item.group_id))
        if real_images and item.split != "train" and not item.image_path:
            raise ValueError(f"Visual image missing: {item.id}")
        if real_images and item.image_path:
            data = Path(item.image_path).read_bytes()
            item.image()  # Decode now, not halfway through model evaluation.
            tokens.append((images, hashlib.sha256(data).hexdigest()))
        for mapping, key in tokens:
            if key in mapping and mapping[key] != item.split:
                raise ValueError(f"Cross-split scene/content/image overlap: {item.id}")
            mapping[key] = item.split
    for required in ("train", "score", "dev"):
        if not any(x.split == required for x in items):
            raise ValueError(f"Required split absent: {required}")


def synthetic_items():
    rows = []
    for split in ("train", "score", "dev"):
        for i in range(8):
            rows.append(Item(id=f"{split}-{i}", group_id=f"{split}-scene-{i}",
                             split=split, question="Choose the action", action_list=["A. left", "B. right"],
                             answer="A" if i % 2 == 0 else "B",
                             situation=f"Synthetic {split} situation {i}",
                             feature=[1.0 if i % 2 == 0 else -1.0, (i - 3.5)/4]))
    return rows
