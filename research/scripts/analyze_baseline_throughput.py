"""Summarize measured ordinary-training throughput, not an inference microbenchmark."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(metrics, events):
    steps = [r for r in metrics if 'timing_s/step' in r['metrics']]
    if len({r['step'] for r in steps}) != len(steps): raise ValueError('Duplicate training steps')
    steps.sort(key=lambda r:r['step'])
    # Retain all steps, but separate the first two from the sustained estimate.
    sustained = [r for r in steps if r['step'] >= 3]
    if len(sustained) < 8: raise ValueError('Need at least eight sustained steps after two startup steps')
    def distribution(values):
        return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                    minimum=min(values), maximum=max(values))
    checkpoint_events = [r for r in events if r['event']=='end' and r['phase']=='_save_checkpoint']
    if not checkpoint_events or any(r['status']!='complete' for r in checkpoint_events):
        raise ValueError('At least one completed checkpoint save is required')
    compute = [r['metrics']['timing_s/step']-r['metrics'].get('timing_s/save_checkpoint',0)
               -r['metrics'].get('timing_s/validation',0) for r in sustained]
    wall_gaps = [b['wall_time']-a['wall_time'] for a,b in zip(steps,steps[1:]) if b['step']>=3]
    return dict(total_steps=len(steps), sustained_steps=len(sustained),
                end_to_end_step_seconds=distribution(wall_gaps),
                train_step_excluding_save_validation_seconds=distribution(compute),
                checkpoint_seconds=distribution([r['seconds'] for r in checkpoint_events]),
                response_length_mean=distribution([r['metrics']['response_length/mean'] for r in sustained]),
                events=events,
                caveat='GPU-allocated wall time includes CPU work and waiting; this is not GPU-kernel time. First two steps are excluded only from sustained rates, never from total cost.')

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    def read(name): return [json.loads(s) for s in (a.directory/name).read_text().splitlines() if s.strip()]
    print(json.dumps(summarize(read('metrics.jsonl'),read('events.jsonl')),indent=2))
