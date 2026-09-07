"""Prepare a tiny deterministic text-only engineering split, never a paper split."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing engineering split")
    table = pq.read_table(args.source, columns=["problem", "answer"])
    rows = table.to_pylist()
    order = list(range(len(rows)))
    random.Random(20260910).shuffle(order)
    selected, seen = [], set()
    for index in order:
        row = rows[index]
        if not all(isinstance(row[k], str) and row[k].strip() for k in ("problem", "answer")):
            continue
        digest = hashlib.sha256(row["problem"].strip().encode()).hexdigest()
        if digest not in seen:
            selected.append({"source_row": index, "prompt_sha256": digest})
            seen.add(digest)
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise SystemExit("Insufficient distinct valid prompts")
    args.output.mkdir(parents=True)
    manifest = {
        "purpose": "one-step text-only engineering baseline; not a scientific split",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_revision": "9724bba24752a793369dee7e71ac1bb75db47404",
        "seed": 20260910,
        "splits": {},
    }
    for name, entries in (("train", selected[:4]), ("val", selected[4:])):
        path = args.output / (name + ".parquet")
        pq.write_table(table.take([x["source_row"] for x in entries]), path)
        manifest["splits"][name] = {"rows": entries, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
