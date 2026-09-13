"""Summarize measured ordinary-training throughput, not an inference microbenchmark."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(metrics, events, launcher=None):
    steps = [r for r in metrics if 'timing_s/step' in r['metrics']]
    if len({r['step'] for r in steps}) != len(steps): raise ValueError('Duplicate training steps')
    steps.sort(key=lambda r:r['step'])
    if [r['step'] for r in steps] != list(range(1,len(steps)+1)):
        raise ValueError('Need contiguous training steps starting at one')
    for row in steps:
        values=[row['wall_time'],row['metrics']['response_length/mean']]
        values.extend(v for k,v in row['metrics'].items() if k.startswith('timing_s/'))
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid timing or response-length measurement')
    if any(b['wall_time']<=a['wall_time'] for a,b in zip(steps,steps[1:])):
        raise ValueError('Nonmonotone training timestamps')
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
    if min(compute) < 0:
        raise ValueError('Checkpoint/validation time exceeds its containing step')
    wall_gaps = [b['wall_time']-a['wall_time'] for a,b in zip(steps,steps[1:]) if b['step']>=3]
    steady_wall = [b['wall_time']-a['wall_time']
                   -b['metrics'].get('timing_s/save_checkpoint',0)
                   -b['metrics'].get('timing_s/validation',0)
                   for a,b in zip(steps,steps[1:]) if b['step']>=3]
    if min(steady_wall) < 0:
        raise ValueError('Excluded phases exceed observed wall-clock step gap')
    phase_ends=[r for r in events if r.get('event')=='end' and 'seconds' in r]
    if any(not math.isfinite(r['seconds']) or r['seconds']<0 for r in phase_ends):
        raise ValueError('Invalid phase duration')
    phases={name:sum(r['seconds'] for r in phase_ends if r['phase']==name and r.get('status')=='complete')
            for name in sorted({r['phase'] for r in phase_ends})}
    all_train=sum(r['metrics']['timing_s/step']-r['metrics'].get('timing_s/save_checkpoint',0)
                  -r['metrics'].get('timing_s/validation',0) for r in steps)
    # Checkpoint/validation events may overlap step/fit durations. Never sum
    # enclosing fit with its children or add saves twice to per-step work.
    accounted=(all_train+phases.get('init_workers',0)+phases.get('_load_checkpoint',0)
               +phases.get('_save_checkpoint',0)+phases.get('_validate',0))
    terminal=bool(launcher and launcher.get('status')=='complete' and launcher.get('exit')==0)
    total=launcher.get('elapsed_seconds') if terminal else None
    if total is not None and (not math.isfinite(total) or total < accounted-1.):
        raise ValueError('Phase accounting exceeds complete launcher duration')
    components={key:distribution([r['metrics'][key] for r in sustained])
                for key in ('timing_s/gen','timing_s/old','timing_s/ref','timing_s/reward','timing_s/update_actor')
                if all(key in r['metrics'] for r in sustained)}
    return dict(total_steps=len(steps), sustained_steps=len(sustained),
                end_to_end_step_seconds=distribution(wall_gaps),
                train_step_excluding_save_validation_seconds=distribution(compute),
                wall_step_excluding_save_validation_seconds=distribution(steady_wall),
                checkpoint_seconds=distribution([r['seconds'] for r in checkpoint_events]),
                response_length_mean=distribution([r['metrics']['response_length/mean'] for r in sustained]),
                first_two_step_seconds=sum(r['metrics']['timing_s/step'] for r in steps[:2]),
                all_training_excluding_save_validation_seconds=all_train,
                sustained_components=components,
                completed_phase_seconds=phases,
                terminal_success=terminal,
                launcher_elapsed_seconds=total,
                accounted_seconds=accounted,
                other_startup_orchestration_cleanup_seconds=(total-accounted if total is not None else None),
                events=events,
                caveat='GPU-allocated wall time includes CPU work and waiting; this is not GPU-kernel time. First two steps are excluded only from sustained rates, never from total cost.')

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    def read(name): return [json.loads(s) for s in (a.directory/name).read_text().splitlines() if s.strip()]
    launcher_path=a.directory/'launcher.json'
    launcher=json.loads(launcher_path.read_text()) if launcher_path.exists() else None
    print(json.dumps(summarize(read('metrics.jsonl'),read('events.jsonl'),launcher),indent=2))
