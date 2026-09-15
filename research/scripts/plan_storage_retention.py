"""Planning arithmetic only: no remote calls, file cleanup, or allocation."""
import argparse
import json


def plan(existing_bytes, runs):
    # Decimal GB; measured old history already includes the common parent.
    items = {
        "existing_history_including_common_parent": existing_bytes / 1e9,
        "active_recovery_and_verified_replacement": 2 * 41.28,
        "retained_final_fp32_models": runs * 16.27,
        "one_candidate_export_bound": 32.0,
        "preselected_audit_export_cap": 32.0,
        "additional_receipts_and_data_allowance": 4.0,
        "free_space_reserve": 16.0,
    }
    return {
        "units": "decimal GB", "training_runs": runs,
        "components": items, "planned_total_gb": sum(items.values()),
        "scope": "Keep existing history; one active job; no offsite archive assumed",
        "limitations": [
            "A planning envelope, not a measured integrated peak or storage authorization.",
            "Final model copies counted even during checkpoint publication: conservative overlap.",
            "New full-state checkpoints must pass validation before old managed copies retire.",
            "Existing artifacts are not deleted. Candidate evidence must be backed up before cleanup.",
            "Abort before exceeding caps; larger optimizer state or datasets require reassessment.",
            "Filesystem-wide df capacity is not the purchased network-volume quota.",
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--existing-bytes", type=int, required=True)
    p.add_argument("--runs", type=int, default=6)
    args = p.parse_args()
    if args.existing_bytes < 0 or args.runs < 1:
        p.error("existing bytes must be nonnegative and runs positive")
    print(json.dumps(plan(args.existing_bytes, args.runs), indent=2))
