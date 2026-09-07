"""Observe the released text reward without replacing it."""
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import append_json, write_json


COMPONENTS = ("accuracy", "format", "tag_count", "length")
WEIGHTS = np.array([1.0, 0.8, 0.4, 0.5])


def verify_contract(contract_path, scorer):
    contract = json.loads(Path(contract_path).read_text())
    source = Path(inspect.getsourcefile(scorer))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != contract["text"]["source_sha256"]:
        raise ValueError("Released text reward source differs from frozen contract")
    root = Path(__file__).resolve().parents[1]
    for name, expected in contract["visual"]["source_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Visual parser source differs from frozen contract")
    return contract


def group_diagnostics(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["group_id"]].append(row)
    result = []
    for group_id, rows in groups.items():
        components = np.array([[r["components"][k] for k in COMPONENTS] for r in rows])
        total = np.array([r["components"]["overall"] for r in rows])
        weighted = components * WEIGHTS
        centered = weighted - weighted.mean(axis=0)
        covariance = centered.T @ centered / len(rows)
        result.append({"group_id": group_id, "n": len(rows),
                       "correctness_constant": bool(np.ptp(components[:, 0]) == 0),
                       "composite_constant": bool(np.ptp(total) == 0),
                       "composite_population_variance": float(np.var(total)),
                       "weighted_component_population_covariance": covariance.tolist()})
    return result


def audit_text_batch(data, tokenizer, scorer, output, contract_path):
    """Validate exact terminal reward and save every component, grouped by prompt."""
    output = Path(output)
    contract = verify_contract(contract_path, scorer)
    records = []
    for i in range(len(data)):
        row = data[i]
        count = int(row.batch["response_mask"].sum())
        if count < 1:
            raise ValueError("Empty response cannot receive a terminal reward")
        ids = row.batch["responses"][:count].detach().cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        gold = str(row.non_tensor_batch["ground_truth"])
        scores = {k: float(v) for k, v in scorer(text, gold).items()}
        total = float(sum(scores[k] * w for k, w in zip(COMPONENTS, WEIGHTS)))
        if abs(total - scores["overall"]) > 1e-7:
            raise ValueError("Reward coefficients differ from contract")
        expected = torch.zeros_like(row.batch["token_level_scores"])
        expected[count-1] = scores["overall"]
        if not torch.equal(expected, row.batch["token_level_scores"]):
            raise ValueError("Scorer disagrees with the actual Praxis reward tensor")
        group = row.non_tensor_batch.get("uid")
        if group is None:
            raise ValueError("Missing original GRPO prompt-group identity")
        records.append({"index": i, "group_id": str(group), "text": text,
                        "response_token_ids": ids, "ground_truth": gold,
                        "components": scores, "response_length": count})
    for row in records:
        append_json(output / "text-rewards.jsonl", row)
    write_json(output / "text-reward-summary.json", {
        "contract": contract, "component_order": list(COMPONENTS),
        "groups": group_diagnostics(records),
        "note": "Covariance included; component variances do not add independently."})
    return records
