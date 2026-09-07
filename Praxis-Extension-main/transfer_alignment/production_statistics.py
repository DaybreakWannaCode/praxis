"""Separate fixed-panel sampling uncertainty from variation across images."""
import itertools
import math

import numpy as np


def bernoulli_kl_interval(successes, count, alpha=.05):
    """Conservative Chernoff-KL interval for independent Bernoulli observations.

    Success probabilities may differ. Jensen bounds their joint MGF by that of
    count iid Bernoullis with the average success probability. Invert each
    Chernoff tail at alpha/2. This is not a Clopper-Pearson interval.
    """
    if count<1 or int(count)!=count or int(successes)!=successes or not 0<=successes<=count or not 0<alpha<1:
        raise ValueError("Invalid Bernoulli counts or confidence level")
    q=successes/count
    threshold=math.log(2/alpha)/count
    def kl(p):
        if p<=0:return 0. if q==0 else math.inf
        if p>=1:return 0. if q==1 else math.inf
        return (q*math.log(q/p) if q else 0.)+((1-q)*math.log((1-q)/(1-p)) if q<1 else 0.)
    lower=0.
    if successes:
        left,right=0.,q
        for _ in range(70):
            mid=(left+right)/2
            if kl(mid)>threshold:left=mid
            else:right=mid
        lower=left  # outward rounding
    upper=1.
    if successes<count:
        left,right=q,1.
        for _ in range(70):
            mid=(left+right)/2
            if kl(mid)>threshold:right=mid
            else:left=mid
        upper=right
    return [lower,upper]


def outcome_contrast(first, second, alpha=.05):
    """First minus second on the same image/response grid, with independent cells.

    Coupling the two policies within a cell is allowed (common random numbers).
    Equal response counts per image are required for the image-uniform objective.
    The interval is conditional on this fixed image panel, not population-wide.
    """
    a,b=np.asarray(first),np.asarray(second)
    if a.ndim!=2 or a.shape!=b.shape or not a.size or not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():
        raise ValueError("Need matching binary image-by-response arrays")
    d=a.astype(float)-b
    positive=int((d>0).sum());negative=int((d<0).sum())
    plus=bernoulli_kl_interval(positive,d.size,alpha/2)
    minus=bernoulli_kl_interval(negative,d.size,alpha/2)
    per_image=d.mean(axis=1)
    return {"difference":float(d.mean()),"positive_pairs":positive,"negative_pairs":negative,
            "response_pairs":int(d.size),"images":int(d.shape[0]),
            "fixed_panel_interval":[plus[0]-minus[1],plus[1]-minus[0]],
            "confidence":1-alpha,"method":"paired sign-category Chernoff-KL union bound",
            "per_image_differences":per_image.tolist(),
            "descriptive_image_sem":float(per_image.std(ddof=1)/len(per_image)**.5) if len(per_image)>1 else None,
            "multiplicity":"pointwise; caller must allocate alpha for simultaneous claims"}


def alignment_contrasts(projected, alpha=.05):
    """Input: repeat x image x candidate projections, before image averaging."""
    from scipy.stats import t
    x=np.asarray(projected,dtype=float)
    if x.ndim!=3 or x.shape[0]<2 or x.shape[1]<1 or x.shape[2]<2 or not np.isfinite(x).all() or not 0<alpha<1:
        raise ValueError("Need finite repeated per-image candidate projections")
    repeats=x.shape[0]
    critical=float(t.ppf(1-alpha/2,repeats-1))
    rows=[]
    for a,b in itertools.combinations(range(x.shape[2]),2):
        contrast=x[:,:,a]-x[:,:,b]
        means=contrast.mean(axis=1)
        mean=float(means.mean())
        sem=float(means.std(ddof=1)/math.sqrt(repeats))
        image_means=contrast.mean(axis=0)
        rows.append({"first":a,"second":b,"difference":mean,"replicate_values":means.tolist(),
                     "fixed_panel_mc_sem":sem,"approximate_t_interval":[mean-critical*sem,mean+critical*sem],
                     "descriptive_image_sem":float(image_means.std(ddof=1)/len(image_means)**.5) if len(image_means)>1 else None,
                     "note":"t interval is an approximation with few response repeats; image variation is separate"})
    return rows
