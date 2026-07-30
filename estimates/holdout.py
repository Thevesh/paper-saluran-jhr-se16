"""
The unrestricted-vs-within-polling-district holdout, reported numerically.
Usage: python3 holdout.py [election ...]

Standalone: reuses groundtruth.py's split and joint.py's holdout machinery. Writes only
its own file (validation_design_ab.csv). Nothing in 00-16 is touched or re-run.

WHY THIS EXISTS
groundtruth.py already runs both designs and prints the comparison, but only for the
SEPARATE model -- joint.py's re-run of the holdout with joint rows is design A only, so the
specification the paper actually reports has never been shown under design B. Since the joint
holdout is the paper's headline validation result (separate model misses 70+ BN by ~16pp, joint
model closes it), the confound defence has to be made on that model, not on its predecessor.

THE OBJECTION
Pure and mixed salurans are not randomly assigned. A saluran is age-pure because its polling
place had enough voters of one band to fill a whole stream, so pure salurans sit in larger,
denser localities. A prediction gap therefore mixes aggregation bias with a genuine
urban/rural behavioural difference.

    DESIGN A  fit on all mixed salurans, predict all pure ones. Maximum power, confounded.
    DESIGN B  restrict BOTH sides to dms holding pure AND mixed salurans, so prediction and
              truth come from the same polling place. Locality held roughly fixed.

If a gap survives B it is not locality. If it collapses under B it was. This script reports
that cell by cell for both models and both dimensions, plus the coverage cost of B (which is
real: B discards every dm that is entirely one kind).
"""

import os

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES, ORDER

import groundtruth as gt
import joint as jm
import reconcile as rec

OUT = common.OUT
THR = 0.95
# A group earns a measured error bar only if its held-out stratum spans enough polling
# districts to estimate variability ACROSS clusters -- the bootstrap of errors.py resamples
# districts, so with a handful of them there is nothing to resample.
#
# The value is not doing the work; the separation in these data is. Every group either spans at
# least 19 districts or at most 4, so ANY threshold from 5 to 19 yields the identical
# classification: Malay, Chinese and all six age bands earn bars, Indian and Other do not.
# We take 10. Two reported groups sit below the 30-cluster floor conventionally cited for
# cluster-based variance estimation (Cameron & Miller 2015) -- Chinese at 32-36 and the over-70s
# at 19-29 -- and the text flags both as the least reliable of the reported bars.
MIN_CLUSTERS = 10
JOINT = jm.JOINT


def design_masks(df, dim, design, thr=THR):
    """(pure, mixed, argmax group index) under design A or B."""
    pure, arg = gt.split(df, dim, thr)
    if design == "B":
        t = pd.DataFrame({"dm": df.dm.to_numpy(), "pure": pure})
        agg = t.groupby("dm").pure.agg(["sum", "count"])
        both = set(agg.index[(agg["sum"] > 0) & (agg["sum"] < agg["count"])])
        keep = df.dm.isin(both).to_numpy()
        return pure & keep, (~pure) & keep, arg
    return pure, ~pure, arg


def holdout(df, el, dim, design, keep_groups=None):
    """Fit joint AND separate on the design's mixed set; predict its pure stratum."""
    pure, mixed, arg = design_masks(df, dim, design)
    groups = DIMS[dim]
    fit_df = df[mixed].reset_index(drop=True)

    total, _ = jm.fit_joint(fit_df, el)
    Nf = jm.jcounts(fit_df).sum(axis=0)
    b = {e: jm.jrates(total, Nf, el, e) for e in ESTIMAND_NAMES}

    sep_seed = rec.build_seed(fit_df, el, dim)
    sep_tot, _ = rec.reconcile(fit_df, el, dim, sep_seed)
    sep_N = common.counts(fit_df, dim).sum(axis=0)
    sep = {e: rec.rates(sep_tot, sep_N, el, e) for e in ESTIMAND_NAMES}

    Njoint, V, N4 = jm.jcounts(df), df.V.to_numpy(float), common.counts(df, dim)
    rows = []
    for j, g in enumerate(groups):
        if keep_groups is not None and g not in keep_groups:
            continue
        m = pure & (arg == j)
        if m.sum() == 0:
            continue
        cover = N4[m, j].sum() / N4[:, j].sum()
        if keep_groups is None and df.dm[m].nunique() < MIN_CLUSTERS:
            continue
        idx = [k for k, (e, a) in enumerate(JOINT) if (e if dim == "ethnicity" else a) == g]
        wts = Njoint[m][:, idx].sum(axis=0)
        for e in ESTIMAND_NAMES:
            Y = common.outcome(df, el, e)[m]
            lo = np.maximum(0.0, Y - (V[m] - N4[m, j])).sum() / N4[m, j].sum()
            hi = np.minimum(Y, N4[m, j]).sum() / N4[m, j].sum()
            pj = float((wts * b[e][idx]).sum() / max(wts.sum(), 1e-9))
            ps = float(sep[e][j])
            for model, p in [("joint", pj), ("separate", ps)]:
                rows.append(dict(
                    election=el, dim=dim, design=design, group=g, estimand=e, model=model,
                    pred=p, truth_lo=lo, truth_hi=hi, truth_mid=(lo + hi) / 2,
                    truth_width=hi - lo, miss=max(lo - p, p - hi, 0.0),
                    inside=bool(lo - 1e-9 <= p <= hi + 1e-9), signed=p - (lo + hi) / 2,
                    n_pure=int(m.sum()), n_mixed=int(mixed.sum()),
                    n_dm=int(df.dm[mixed | pure].nunique()),
                    pure_electorate=float(N4[m, j].sum()), coverage=float(cover),
                ))
    return rows


def selection(df, dim, design, thr=THR):
    """Pure-vs-mixed composition, within the design's own universe."""
    pure, mixed, _ = design_masks(df, dim, design)
    V = df.V.to_numpy(float)
    turn = 1 - df.NONVOTE.to_numpy(float) / V
    dmsize = df.groupby("dm").V.transform("sum").to_numpy(float)
    other = "ethnicity" if dim == "age" else "age"
    Xo = common.shares(df, other)
    d = dict(design=design, dim=dim, n_pure=int(pure.sum()), n_mixed=int(mixed.sum()),
             elec_pure=V[pure].mean(), elec_mixed=V[mixed].mean(),
             dmsize_pure=dmsize[pure].mean(), dmsize_mixed=dmsize[mixed].mean(),
             turnout_pure=np.average(turn[pure], weights=V[pure]),
             turnout_mixed=np.average(turn[mixed], weights=V[mixed]))
    for k, gg in enumerate(common.groups(other)):
        d[f"{gg}_pure"] = np.average(Xo[pure, k], weights=V[pure])
        d[f"{gg}_mixed"] = np.average(Xo[mixed, k], weights=V[mixed])
    return d


def main():
    els = common.arg_elections()
    rows, sel = [], []
    for el in els:
        df = jm.load_joint(el)
        for dim in DIMS:
            keep = sorted({r["group"] for r in holdout(df, el, dim, "A")})
            for design in ["A", "B"]:
                rows += holdout(df, el, dim, design, keep_groups=keep)
                sel.append(dict(election=el, **selection(df, dim, design)))
        print(f"  {el}: done")

    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/validation_design_ab.csv", index=False)
    S = pd.DataFrame(sel)
    S.to_csv(f"{OUT}/validation_design_ab_selection.csv", index=False)

    for el in els:
        for model in ["joint", "separate"]:
            s = R[(R.election == el) & (R.model == model)]
            if s.empty:
                continue
            p = s.pivot_table(index=["dim", "group", "estimand"], columns="design",
                              values="miss") * 100
            p["A_minus_B"] = p.A - p.B
            print(f"\n=== {el} / {model} : |miss| pp by design ===")
            print(p.round(2).to_string())
            print(f"mean A {p.A.mean():.2f}  B {p.B.mean():.2f}")
    print(f"\nwrote {OUT}/validation_design_ab.csv")


if __name__ == "__main__":
    main()
