"""Create a separate telemetry-only copy of original Praxis. No training edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

EVENT_HELPER = '''

def _praxis_measure_call(name, method):
    import functools
    @functools.wraps(method)
    def measured(self, *args, **kwargs):
        import json, os, time
        path = os.environ.get("PRAXIS_THROUGHPUT_EVENTS")
        start = time.monotonic()
        def record(event, **fields):
            if path:
                with open(path, "a") as stream:
                    stream.write(json.dumps(dict(event=event, phase=name, wall_time=time.time(),
                                                 step=getattr(self, "global_step", None), **fields)) + "\\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        record("start")
        status = "failed"
        try:
            result = method(self, *args, **kwargs)
            status = "complete"
            return result
        finally:
            record("end", seconds=time.monotonic()-start, status=status)
    return measured

for _method in ("init_workers", "fit", "_save_checkpoint", "_load_checkpoint", "_validate"):
    setattr(RayPPOTrainer, _method, _praxis_measure_call(_method, getattr(RayPPOTrainer, _method)))
'''


def instrument(source, destination):
    if destination.exists(): raise ValueError('Destination must be new')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    changes = []
    trainer = destination/'verl/trainer/ray_trainer.py'
    before = trainer.read_text()
    trainer.write_text(before + EVENT_HELPER)
    changes.append(str(trainer.relative_to(destination)))
    logger = destination/'verl/utils/logger/logger.py'
    before = logger.read_text()
    needle = '        print(f"Step {step}\\n" + convert_dict_to_str(unflatten_dict(data)))'
    if before.count(needle) != 1: raise ValueError('Unexpected ConsoleLogger source')
    addition = '''
        path = os.environ.get("PRAXIS_THROUGHPUT_METRICS")
        if path:
            import json, time
            with open(path, "a") as stream:
                stream.write(json.dumps(dict(step=step, wall_time=time.time(), metrics=data), default=float) + "\\n")
                stream.flush()
                os.fsync(stream.fileno())'''
    logger.write_text(before.replace(needle, needle + addition))
    changes.append(str(logger.relative_to(destination)))
    inventory = {}
    for relative in changes + ['verl/utils/reward_score/mcq.py', 'verl/workers/actor/dp_actor.py']:
        inventory[relative] = {kind:hashlib.sha256((root/relative).read_bytes()).hexdigest()
                              for kind, root in [('original',source),('observed',destination)]}
    if any(inventory[p]['original'] != inventory[p]['observed'] for p in inventory if p not in changes):
        raise ValueError('Training or reward changed')
    (destination/'telemetry-source-manifest.json').write_text(json.dumps(inventory, indent=2))
    return inventory

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    a=p.parse_args()
    instrument(a.source,a.destination)
