"""Independently read every exported tensor and validate the four-step chain."""
import argparse
import json
import math
from pathlib import Path

import torch

from transfer_alignment.production_displacement import load_tensor
from transfer_alignment.production_horizon import validate_horizon_export

parser = argparse.ArgumentParser()
parser.add_argument("gate", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
torch.set_num_threads(4)
report = json.loads((args.gate / "parity.json").read_text())
validate_horizon_export(args.gate, report, 4)
manifest = json.loads((args.gate / "delta/manifest.json").read_text())
squared = 0.0
count = 0
for index, row in enumerate(manifest["parameters"]):
    tensor = load_tensor(args.gate / "delta", row).reshape(-1)
    for start in range(0, tensor.numel(), 262144):
        chunk = tensor[start:start + 262144].double()
        if not torch.isfinite(chunk).all():
            raise ValueError(f"Nonfinite tensor: {row['name']}")
        squared += chunk.square().sum().item()
    count += tensor.numel()
    if (index + 1) % 100 == 0:
        print(f"Verified {index + 1} tensors", flush=True)
norm = math.sqrt(squared)
assert count == manifest["canonical_numel"]
assert math.isclose(norm, manifest["update_norm"], rel_tol=1e-10, abs_tol=1e-12)
result = dict(status="passed", verified_tensors=len(manifest["parameters"]),
              canonical_numel=count, update_norm=norm,
              all_file_checksums_verified=True,
              float64_tensors=sum(r.get("dtype") == "float64" for r in manifest["parameters"]))
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
