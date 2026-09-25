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


def verify_recovered_step(original, recovered):
    """Match saved rollout and model/optimizer trajectory, not fresh driver RNG."""
    if original['input_digest'] != recovered['input_digest']:
        raise ValueError('Recovery optimizer input differs from archived input')
    for boundary in ('parent_digests', 'control_digests', 'observed_digests'):
        for field in CHAIN_FIELDS:
            if original[boundary][field] != recovered[boundary][field]:
                raise ValueError(f'Recovery differs: {boundary}/{field}')


def prompt_inventory(data):
    """Order-independent text/gold inventory including rollout multiplicities."""
    import hashlib
    rows = data.non_tensor_batch
    if 'problem' not in rows or 'ground_truth' not in rows:
        raise ValueError('H=4 requires explicit text problem and ground truth')
    problems, gold = rows['problem'], rows['ground_truth']
    if len(problems) != len(gold) or not len(problems):
        raise ValueError('Empty or mismatched H=4 text inventory')
    pairs = sorted((str(p), str(g)) for p,g in zip(problems,gold))
    return hashlib.sha256(json.dumps(pairs,ensure_ascii=False).encode()).hexdigest()


def run_four_step_gate(worker, data, *, scorer=None):
    """Consume one fresh driver batch; export only after four verified calls.

    This opt-in wrapper never generates rollouts or advances the optimizer itself.
    The original driver must supply fresh on-policy inputs on each invocation.
    Any failure poisons this worker's horizon; restart from the frozen parent.
    """
    import os
    import time
    import torch
    from .experiment import write_json
    from .production_parity import parameters, run_fixed_rollout_gate
    from .production_coordinates import manifest, verify_values, canonical_views
    from .production_displacement import save_displacement

    if os.environ.get('PRAXIS_FIXED_INPUT') or os.environ.get('PRAXIS_RESUME_DELTA_DIR'):
        raise ValueError('H=4 requires fresh driver inputs; H=1 recovery is forbidden')
    recovery_root = os.environ.get('PRAXIS_H4_RECOVERY_DIR')
    prior_state = getattr(worker, '_alignment_horizon', None)
    if recovery_root:
        recovery_root = Path(recovery_root)
        # These are trusted project-generated torch files, never arbitrary uploads.
        # Validate the whole archived four-step chain before consuming any input.
        archived = validate_four_steps([recovery_root/f'step-{i}' for i in range(1,5)])
        next_index = prior_state['count'] if prior_state else 0
        if next_index >= 4:
            raise RuntimeError('Recovery already contains four updates')
        data = torch.load(recovery_root/f'step-{next_index+1}'/'fixed-update-input.pt',
                          map_location='cpu', weights_only=False)
    inventory = prompt_inventory(data)
    expected_inventory = os.environ.get('PRAXIS_H4_PROMPT_INVENTORY')
    if not expected_inventory or inventory != expected_inventory:
        raise ValueError('H=4 input does not match the frozen H1 candidate inventory')
    state = getattr(worker, '_alignment_horizon', None)
    if state is None:
        if getattr(worker, '_parity_gate_done', False):
            raise ValueError('H=4 must start from a fresh warm-parent worker')
        root = Path(os.environ['PRAXIS_PARITY_DIR'])
        root.mkdir(parents=True, exist_ok=False)
        state = dict(root=root, count=0, failed=True, start=time.monotonic())
        worker._alignment_horizon = state
        mapping = manifest(worker)
        checkpoint = torch.load(os.environ['PRAXIS_PARENT_MODEL'], map_location='cpu',
                                mmap=True, weights_only=False)
        validation = verify_values(parameters(worker), mapping, checkpoint)
        del checkpoint
        initial = {k: p.detach().cpu().clone() for k,p in parameters(worker).items()}
        state.update(mapping=mapping, validation=validation, initial=initial, failed=False,
                     prompt_inventory=inventory, recovery_root=str(recovery_root) if recovery_root else None)
        write_json(root/'coordinates.json', mapping)
        write_json(root/'horizon.json', dict(status='running', horizon=4, completed_steps=0))
    if state['failed'] or state['count'] >= 4:
        raise RuntimeError('Horizon failed or already contains four updates')
    if state['recovery_root'] != (str(recovery_root) if recovery_root else None):
        raise ValueError('Recovery source changed during the horizon')
    root = state['root']
    step_root = root / f"step-{state['count']+1}"
    saved_env = {k: os.environ.get(k) for k in ('PRAXIS_PARITY_DIR','PRAXIS_PARENT_MODEL')}
    try:
        if inventory != state['prompt_inventory']:
            raise ValueError('H=4 candidate text inventory changed between updates')
        # Per-step parity retains all restore checks but avoids intermediate
        # delta exports and comparison of later weights to the initial parent.
        os.environ['PRAXIS_PARITY_DIR'] = str(step_root)
        os.environ.pop('PRAXIS_PARENT_MODEL', None)
        worker._parity_gate_done = False
        if recovery_root:
            from .production_parity import state_digests
            current = state_digests(worker)
            for field in CHAIN_FIELDS:
                if current[field] != archived[state['count']]['parent_digests'][field]:
                    raise ValueError(f'Recovery parent differs: {field}')
        result = run_fixed_rollout_gate(worker, data, scorer=scorer)
        previous = state.get('previous')
        report = validate_step(step_root, previous)
        if recovery_root:
            verify_recovered_step(archived[state['count']], report)
            write_json(step_root/'recovery.json', dict(
                source=str(recovery_root/f"step-{state['count']+1}"),
                input_and_state_match=True,
                scope='saved optimizer input and parameter/buffer/Adam/scheduler trajectory; not driver RNG replay'))
        if state['count'] == 0:
            state['first'] = report
        state['previous'] = report
        state['count'] += 1
        write_json(root/'horizon.json', dict(status='running', horizon=4,
                   completed_steps=state['count']))
        if state['count'] == 4:
            reports = validate_four_steps([root/f'step-{i}' for i in range(1,5)])
            if manifest(worker) != state['mapping']:
                raise ValueError('Canonical coordinates changed over H=4')
            delta = save_displacement(canonical_views(state['initial'], state['mapping']),
                                      canonical_views(parameters(worker), state['mapping']), root/'delta',
                                      allow_float64=True)
            combined = dict(report)
            combined.update(status='passed', horizon=4, completed_steps=4,
                            prompt_inventory=inventory,
                            parent_digests=state['first']['parent_digests'],
                            canonical_validation=state['validation'],
                            canonical_displacement={k:v for k,v in delta.items() if k!='parameters'},
                            step_input_digests=[r['input_digest'] for r in reports],
                            elapsed_seconds=time.monotonic()-state['start'],
                            scope='four consecutive one-rank parity-gated worker updates')
            if recovery_root:
                combined['recovery_source'] = str(recovery_root)
                combined['recovery_scope'] = 'Exact archived optimizer inputs and state trajectory; fresh driver rollouts discarded'
            write_json(root/'parity.json', combined)
            write_json(root/'horizon.json', dict(status='passed', horizon=4, completed_steps=4))
            del state['initial']
        return result
    except BaseException as exc:
        state['failed'] = True
        write_json(root/'horizon.json', dict(status='failed', horizon=4,
                   completed_steps=state['count'], error=repr(exc)))
        raise
    finally:
        for key,value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_horizon_export(root, report, horizon):
    """Reject relabelled or incomplete trajectories before visual sampling."""
    root = Path(root)
    if report.get('horizon', 1) != horizon:
        raise ValueError('Export horizon differs from requested horizon')
    if horizon == 1:
        return
    if horizon != 4 or report.get('completed_steps') != 4:
        raise ValueError('H=4 export requires four completed steps')
    progress = json.loads((root/'horizon.json').read_text())
    if progress != dict(status='passed', horizon=4, completed_steps=4):
        raise ValueError('H=4 export is incomplete')
    steps = validate_four_steps([root/f'step-{i}' for i in range(1,5)])
    if report['parent_digests'] != steps[0]['parent_digests']:
        raise ValueError('H=4 initial parent differs from first step')
    for field in ('control_digests', 'observed_digests'):
        if report[field] != steps[-1][field]:
            raise ValueError('H=4 final state differs from fourth step')
    if report.get('step_input_digests') != [s['input_digest'] for s in steps]:
        raise ValueError('H=4 input inventory differs')
    delta = json.loads((root/'delta/manifest.json').read_text())
    if not delta.get('child_reconstruction_exact'):
        raise ValueError('H=4 total displacement cannot reconstruct child')
    if report['canonical_displacement'] != {k:v for k,v in delta.items() if k!='parameters'}:
        raise ValueError('H=4 displacement metadata differs')
