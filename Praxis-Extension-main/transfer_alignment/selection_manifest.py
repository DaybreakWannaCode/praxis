"""Seal static batch selection from scores only, before independent outcomes.

This does not generate candidates, run training, establish split independence,
verify score receipts, or authorize compute. Input receipt validation is upstream.
"""
import hashlib
import json
import math
from pathlib import Path


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def order_key(tag, seed, candidate):
    return hashlib.sha256(encoded([tag,seed,candidate])).hexdigest()


def build_selection(rows, *, seeds, pool_size=64, selected_batches=32, prompts_per_batch=32):
    if any(type(v) is not int or v<=0 for v in (pool_size,selected_batches,prompts_per_batch)):
        raise ValueError('Positive integer batch counts required')
    if selected_batches>=pool_size or len(rows)!=pool_size:
        raise ValueError('Need the entire fixed pool and a strict subset')
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Explicit unique integer seeds required')
    keys={'candidate_id','prompt_ids','alignment','parent_sha256','scoring_protocol_sha256',
          'calibration_split_sha256','score_receipt_sha256','objective'}
    identities=set();prompts=set();contexts=set()
    for row in rows:
        if set(row)!=keys:raise ValueError('Score-only schema required; outcome fields are prohibited')
        identity=row['candidate_id']
        if not isinstance(identity,str) or not identity or identity in identities:
            raise ValueError('Unique candidate IDs required')
        identities.add(identity)
        ids=row['prompt_ids']
        if len(ids)!=prompts_per_batch or any(not isinstance(s,str) or not s for s in ids):
            raise ValueError('Wrong prompt count or invalid prompt IDs')
        if len(set(ids))!=len(ids) or prompts.intersection(ids):raise ValueError('Repeated prompt across pool')
        prompts.update(ids)
        if type(row['alignment']) not in (int,float) or not math.isfinite(row['alignment']):
            raise ValueError('Finite alignment required')
        for name in ('parent_sha256','scoring_protocol_sha256','calibration_split_sha256','score_receipt_sha256'):
            value=row[name]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid provenance hash')
        if row['objective']!='normalized_label_prefix_log_likelihood_v1':raise ValueError('Unexpected score objective')
        contexts.add(tuple(row[k] for k in ('parent_sha256','scoring_protocol_sha256','calibration_split_sha256','objective')))
    if len(contexts)!=1:raise ValueError('Candidates must share parent and scoring contract')
    ordered=sorted(rows,key=lambda r:r['candidate_id'])
    by_id={r['candidate_id']:r for r in rows}
    aligned=sorted(identities,key=lambda c:(-by_id[c]['alignment'],order_key('alignment-tie',0,c)))[:selected_batches]
    arms=[]
    for seed in seeds:
        random_ids=sorted(identities,key=lambda c:order_key('random-membership',seed,c))[:selected_batches]
        for name,members in [('alignment',aligned),('random',random_ids)]:
            training_order=sorted(members,key=lambda c:order_key('training-order',seed,c))
            arms.append(dict(selector=name,seed=seed,candidate_ids=training_order,
                batch_prompt_ids=[list(by_id[c]['prompt_ids']) for c in training_order],
                prompt_ids=[p for c in training_order for p in by_id[c]['prompt_ids']]))
    return dict(version=1,status='selection_manifest_only',pool_size=pool_size,
        selected_batches=selected_batches,prompts_per_batch=prompts_per_batch,
        score_pool_sha256=hashlib.sha256(encoded(ordered)).hexdigest(),
        parent_sha256=ordered[0]['parent_sha256'],objective=ordered[0]['objective'],
        calibration_split_sha256=ordered[0]['calibration_split_sha256'],
        scoring_protocol_sha256=ordered[0]['scoring_protocol_sha256'],arms=arms,
        contract='Fresh training rollouts required; scores do not authorize reusing tentative child updates; '
                 'same static alignment membership across seeds; random membership uses no alignment values')


def save_selection(manifest, path):
    """Exclusive creation; caller must durably back up before training/evaluation."""
    import os
    path=Path(path)
    with path.open('xb') as stream:
        stream.write(encoded(manifest)+b'\n');stream.flush();os.fsync(stream.fileno())


def validate_training_handoff(manifest, *, selector, seed, ordered_prompt_ids, config):
    """Validate actual dataset order and effective original-trainer configuration.

    Invoke before model initialization. Does not validate prompt contents against
    source hashes; source/receipt integrity must already have passed upstream.
    """
    matches=[a for a in manifest['arms'] if a['selector']==selector and a['seed']==seed]
    if len(matches)!=1:raise ValueError('Exactly one sealed arm required')
    arm=matches[0];size=manifest['prompts_per_batch'];steps=manifest['selected_batches']
    batches=arm['batch_prompt_ids']
    if len(batches)!=steps or any(len(b)!=size for b in batches):
        raise ValueError('Sealed batch shape differs')
    flat=[p for batch in batches for p in batch]
    if len(set(flat))!=len(flat) or flat!=arm['prompt_ids'] or list(ordered_prompt_ids)!=flat:
        raise ValueError('Dataset order or sealed batch membership changed')
    if len(arm['candidate_ids'])!=steps or len(set(arm['candidate_ids']))!=steps:
        raise ValueError('Invalid selected candidate coverage')
    if config.data.shuffle is not False:
        raise ValueError('Prompt-level shuffle destroys selected candidate batches')
    if config.data.rollout_batch_size!=size or config.worker.actor.global_batch_size!=size:
        raise ValueError('Rollout/optimizer batch must match scored candidate size')
    if config.worker.actor.ppo_epochs!=1 or config.worker.rollout.n!=5:
        raise ValueError('Expected one actor epoch and five fresh rollouts per prompt')
    if config.trainer.max_steps!=steps or config.trainer.total_episodes!=1:
        raise ValueError('Expected one pass over sealed batches')
    return dict(status='passed',selector=selector,seed=seed,batches=steps,prompts=len(flat),
        scope='Sealed membership/order and configured training budget; runtime update count and fresh rollout provenance still require auditing')
