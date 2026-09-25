"""Fail when a text GRPO run never activates the reasoning reward (reads metrics.jsonl only)."""
import argparse
import json
from pathlib import Path
from transfer_alignment.training_signal import audit_metrics

p=argparse.ArgumentParser()
p.add_argument('metrics',type=Path)
p.add_argument('--output',type=Path)
p.add_argument('--window',type=int,default=8)
p.add_argument('--min-format',type=float,default=0.1)
p.add_argument('--min-length',type=float,default=64.0)
a=p.parse_args()
rows=[json.loads(line) for line in a.metrics.read_text().splitlines() if line.strip()]
result=audit_metrics(rows,window=a.window,min_format=a.min_format,min_length=a.min_length)
result['metrics_file']=str(a.metrics)
if a.output: a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['passed']: raise SystemExit(1)
