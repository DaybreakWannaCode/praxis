"""Within-parent rank diagnostics on independently measured candidate outcomes.

No pooling of raw scores across checkpoints. Permutation p-values assume outcome
exchangeability among candidate batches within each parent; they do not establish
causality, handle adaptive selection, or prove independence of visual scenes.
"""
import math
import random
from collections import defaultdict


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        for index in order[start:end]: result[index] = (start + end - 1) / 2
        start = end
    return result


def correlation(x, y):
    mx, my = sum(x)/len(x), sum(y)/len(y)
    xx = sum((a-mx)**2 for a in x); yy = sum((b-my)**2 for b in y)
    if not xx or not yy: return None
    return sum((a-mx)*(b-my) for a,b in zip(x,y)) / math.sqrt(xx*yy)


def analyze_prediction(rows, *, permutations=5000, seed=20260915):
    if type(permutations) is not int or permutations < 1: raise ValueError('Positive permutation count required')
    if not rows: raise ValueError('No candidate outcomes')
    groups = defaultdict(list); identities=set()
    predictors = ('alignment', 'cosine', 'update_norm')
    for row in rows:
        identity = (row['parent_id'], row['candidate_id'])
        if identity in identities: raise ValueError('Repeated parent/candidate identity')
        identities.add(identity)
        for name in (*predictors, 'outcome_delta'):
            if type(row[name]) not in (int,float) or not math.isfinite(row[name]):
                raise ValueError('Finite numeric predictors and outcomes required')
        if row['update_norm'] < 0 or not -1 <= row['cosine'] <= 1:
            raise ValueError('Invalid norm or cosine')
        groups[row['parent_id']].append(row)
    if any(len(g)<3 for g in groups.values()): raise ValueError('Need at least three candidates per parent')
    # Require all predictors and outcome to vary on the same comparison parents.
    # This prevents different denominator sets from manufacturing a winner.
    alignment_all = {}
    for parent, group in sorted(groups.items()):
        value = correlation(ranks([r['alignment'] for r in group]),
                            ranks([r['outcome_delta'] for r in group]))
        if value is not None: alignment_all[parent] = value
    usable = {}; excluded = {}
    for parent, group in sorted(groups.items()):
        tied = [name for name in (*predictors,'outcome_delta') if len({r[name] for r in group})<2]
        if tied: excluded[parent] = dict(reason='constant_fields', fields=tied, candidates=len(group))
        else: usable[parent] = group
    if not usable:
        return dict(status='not_estimable', excluded_parents=excluded, parent_count=len(groups),
                    alignment_all_informative_parents=alignment_all,
                    reason='No common parent has variation in every comparator and outcome')
    results = {}
    y = {p:ranks([r['outcome_delta'] for r in g]) for p,g in usable.items()}
    for predictor in predictors:
        x = {p:ranks([r[predictor] for r in g]) for p,g in usable.items()}
        per_parent={p:correlation(x[p],y[p]) for p in usable}
        observed=sum(per_parent.values())/len(per_parent)
        # Same randomized permutations for every predictor, fixed before results.
        rng=random.Random(seed); extreme=0
        for _ in range(permutations):
            scores=[]
            for parent in usable:
                shuffled=y[parent].copy();rng.shuffle(shuffled)
                scores.append(correlation(x[parent],shuffled))
            null=sum(scores)/len(scores)
            extreme += abs(null) >= abs(observed)-1e-12
        results[predictor]=dict(per_parent_spearman=per_parent,
            equal_parent_mean_spearman=observed,
            two_sided_permutation_p=(extreme+1)/(permutations+1))
    return dict(status='complete', results=results, comparison_parents=list(usable),
                alignment_all_informative_parents=alignment_all,
                excluded_parents=excluded, permutations=permutations, permutation_seed=seed,
                inference_scope='Conditional within-parent exchangeability; secondary comparator p-values are unadjusted; '
                    'descriptive comparator differences are not significance tests between predictors; '
                    'outcome/probe separation must be verified by the experiment manifest')
