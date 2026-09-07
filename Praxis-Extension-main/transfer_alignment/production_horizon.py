"""Validate consecutive parity-gated updates before exporting a horizon delta.

A driver call is not necessarily one optimizer step. These checks use the actual
observer records and the complete parameter/Adam/scheduler state chain instead.
Fresh rollout generation remains the responsibility of the original driver.
"""
import json
from pathlib import Path

CHAIN_FIELDS = ("parameters", "buffers", "optimizer", "scheduler")


def validate_step(root, previous=None):
    root = Path(root)
    report = json.loads((root / "parity.json").read_text())
    if report.get("status") != "passed":
        raise ValueError("Horizon step did not pass parity")
    required = set(CHAIN_FIELDS) | {"worker"}
    if set(report.get("equal", {})) != required or not all(report["equal"].values()):
        raise ValueError("Incomplete or unequal parity state")
    for field in required:
        if report["control_digests"][field] != report["observed_digests"][field]:
            raise ValueError("Parity digest mismatch")
    if previous is not None:
        for field in CHAIN_FIELDS:
            if report["parent_digests"][field] != previous["observed_digests"][field]:
                raise ValueError(f"Discontinuous horizon state: {field}")
    observer = root / "observer" / "rank-00000"
    steps = [json.loads(x) for x in (observer / "steps.jsonl").read_text().splitlines()]
    boundaries = [json.loads(x) for x in (observer / "boundaries.jsonl").read_text().splitlines()]
    if len(steps) != 1 or len(boundaries) != 1:
        raise ValueError("Each horizon step must contain exactly one optimizer boundary")
    if steps[0].get("status") != "applied" or boundaries[0].get("optimizer_calls") != 1:
        raise ValueError("Horizon optimizer step was skipped or failed")
    if boundaries[0].get("status") == "failed":
        raise ValueError("Horizon optimizer boundary failed")
    rates = steps[0].get("learning_rates_at_step", [])
    if not rates or any(not (0 < float(rate) < float("inf")) for rate in rates):
        raise ValueError("Horizon requires finite positive learning rates")
    return report


def validate_four_steps(roots):
    if len(roots) != 4:
        raise ValueError("H=4 requires exactly four completed updates")
    reports = []
    for root in roots:
        reports.append(validate_step(root, reports[-1] if reports else None))
    return reports
