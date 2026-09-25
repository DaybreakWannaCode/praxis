"""Run the frozen H4 visual check once, inside tmux, with resource accounting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


CONFIG_SHA256 = "7bce8e2b9bd65435fd35da077a494a789507180be16c5d815c970a0e20288eab"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TMUX"):
        parser.error("Run in tmux; the measurement must survive SSH disconnects")
    if hashlib.sha256(args.config.read_bytes()).hexdigest() != CONFIG_SHA256:
        parser.error("H4 configuration differs from the frozen bytes")
    cfg = json.loads(args.config.read_text())
    if not args.python.is_file() or not args.manifest.is_file():
        parser.error("Missing interpreter or visual manifest")
    for gate in cfg["candidate_gates"]:
        root = Path(gate)
        report = json.loads((root.parent / "completion.json").read_text())
        if report.get("complete") is not True or report.get("train_exit") != 0:
            parser.error(f"Candidate not complete: {root}")
        if json.loads((root / "horizon.json").read_text()) != {
            "status": "passed", "horizon": 4, "completed_steps": 4
        }:
            parser.error(f"Four-step report did not pass: {root}")
    # The evaluator additionally validates full state chains, coordinates,
    # reward contract, image hashes and disjoint development panels.
    args.output.mkdir(parents=True, exist_ok=False)
    command = ["timeout", "--signal=TERM", "--kill-after=30s", "36000",
               str(args.python), "-m", "transfer_alignment.production_precision",
               "--config", str(args.config), "--manifest", str(args.manifest)]
    for gate in cfg["candidate_gates"]:
        command.extend(["--gate", gate])
    command.extend(["--output", str(args.output / "result")])
    start = time.monotonic()
    (args.output / "launch.json").write_text(json.dumps({
        "command": command, "start_unix": time.time(),
        "config_sha256": CONFIG_SHA256, "timeout_seconds": 36000,
    }, indent=2))
    with (args.output / "run.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    (args.output / "resource-usage.json").write_text(json.dumps({
        "seconds": time.monotonic() - start,
        "max_child_rss_kib": usage.ru_maxrss,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "scope": "Linux child resource accounting; not summed process-tree RAM",
    }, indent=2))
    (args.output / "run.exit").write_text(str(result.returncode) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
