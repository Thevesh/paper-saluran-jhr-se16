"""Held-out validation error, per group, election AND estimand.
Usage: python3 errors.py [election ...]

WHAT holdout.py LEAVES UNDONE. It measures the holdout miss by pooling every near-pure
saluran of a group into one aggregate rate and comparing that against one aggregate interval.
That yields four numbers per group and election, one per estimand, and taking their root mean
square -- as the manuscript previously did -- prints the same error bar beside turnout, BN, PH
and PN even though the three coalitions are predicted with visibly different accuracy. Malay
PH support is estimated far better than Malay BN support, and a pooled bar says otherwise in
both directions at once. This script keeps the estimands apart.

TWO SCALES, TWO KINDS OF ERROR. For each held-out stream i whose dominant group is g, the
margins pin the group's rate to an exact interval:

    lo_i = max(0, Y_i - (V_i - N_ig)) / N_ig        floor
    hi_i = min(Y_i, N_ig) / N_ig                    ceiling
    p_i  = the model's prediction at THIS stream's own joint-cell composition
    e_i  = max(lo_i - p_i, p_i - hi_i, 0)           zero if the prediction is admissible

Summarising e_i across streams answers "how far is the model from one stream of 500 voters?"
That is NOT the question the error bar on a statewide estimate should answer, and the two
differ by a factor of four to five in these data (see the diagnostic printout). The gap is
mostly genuine stream-to-stream variation in behaviour, which is real and which averages out
of a statewide number; the electorate-weighted saluran RMSE would charge the statewide
estimate for heterogeneity it does not suffer from. We report the saluran distribution as a
diagnostic and measure the error bar at the scale of the estimand it decorates.

THE ERROR BAR: a block bootstrap of the aggregate miss. The estimand is statewide, so the
miss is computed statewide -- pool the held-out streams, form the aggregate interval, compare.
The uncertainty in that single number is which corners of Johor happened to be validatable, so
we resample POLLING DISTRICTS with replacement among the held-out streams, recompute the
aggregate miss in each replicate, and report the root mean square across replicates:

    E_gct = sqrt( mean_b [ miss_b^2 ] )

Blocking on the polling district respects the fact that streams in one place are not
independent draws. The result is at or above the observed aggregate miss -- it carries both
how far the model lands from observed behaviour and how much that distance moves when the
validation set changes -- and it is specific to one group, one election and one estimand.

BOTH REPORTING SCALES, AND WHY THE SECOND CANNOT BE SCORED DIRECTLY. The manuscript reports
each party as a share of the group's REGISTERED voters and as a share of its voters. Scoring
the second against its own bounds looks more rigorous than deriving it, and is not, because a
ratio's admissible interval is roughly twice as wide as its components' and the two component
errors can cancel. Malay BN at SE-16 is the clean illustration: the model undershoots turnout
(74.33 against bounds of [76.79, 78.76]) and undershoots BN votes (62.81 against [63.67,
65.64]), both wrong and both wrong in the same direction -- yet 62.81/74.33 = 84.54 lands
inside the ratio's wider interval and scores a miss of exactly zero. An error bar that
vanishes when the estimator is demonstrably wrong on both inputs is measuring the looseness of
its target, not the accuracy of the estimate.

So the vote-share panel inherits its error, worst-case, from the two quantities it is built
from:

    E_v = ( E_party + share * E_turnout ) / turnout

which is the exact bound on |d(party/turnout)| given |d(party)| and |d(turnout)|, assuming no
cancellation. We report max(measured, propagated) so that the direct measurement still binds
in the handful of cases where it is the larger. These appear as BN_v, PH_v, PN_v.

DESIGN A throughout: fit on every mixed saluran, predict every pure one, leaving locality
differences inside the measured error. See holdout.py for why that is conservative.

Outputs
    out/validation_residuals.csv   one row per election x group x estimand x saluran
    out/validation_errors.csv      the error bars, consumed by tables.py
"""

import os

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES

import groundtruth as gt
import joint as jm
import holdout as hold

OUT = common.OUT
JOINT = jm.JOINT
NBOOT = 2000
SEED = 20260711        # polling day, so the seed is at least honest about being arbitrary
MIN_TURNOUT = 0.05     # below this a vote-share ratio is not meaningfully bounded
SCALED = [f"{p}_v" for p in common.PARTIES]
ALL_ESTIMANDS = ESTIMAND_NAMES + SCALED


def stream_table(df, el, dim, keep_groups):
    """Per-saluran ingredients of the holdout, for every group with a pure stratum."""
    pure, mixed, arg = hold.design_masks(df, dim, "A")
    fit_df = df[mixed].reset_index(drop=True)

    total, _ = jm.fit_joint(fit_df, el)
    Nf = jm.jcounts(fit_df).sum(axis=0)
    b = {e: jm.jrates(total, Nf, el, e) for e in ESTIMAND_NAMES}

    Njoint, V, N4 = jm.jcounts(df), df.V.to_numpy(float), common.counts(df, dim)
    out = {}
    for j, g in enumerate(DIMS[dim]):
        if g not in keep_groups:
            continue
        m = pure & (arg == j)
        if m.sum() == 0:
            continue
        idx = [k for k, (e, a) in enumerate(JOINT) if (e if dim == "ethnicity" else a) == g]
        w = Njoint[m][:, idx]
        t = dict(dm=df.dm.to_numpy()[m], V=V[m], N=N4[m, j],
                 wt=np.maximum(w.sum(axis=1), 1e-9), group_electorate=N4[:, j].sum())
        for e in ESTIMAND_NAMES:
            t[f"Y_{e}"] = common.outcome(df, el, e)[m]
            t[f"p_{e}"] = (w * b[e][idx]).sum(axis=1) / t["wt"]
        out[g] = pd.DataFrame(t)
    return out


def aggregate_miss(t, rows=None):
    """Statewide-scale miss per estimand over a (possibly resampled) set of streams."""
    s = t if rows is None else t.iloc[rows]
    N, V, wt = s.N.to_numpy(), s.V.to_numpy(), s.wt.to_numpy()
    Ns, ws = N.sum(), wt.sum()
    lo, hi, pred = {}, {}, {}
    for e in ESTIMAND_NAMES:
        Y = s[f"Y_{e}"].to_numpy()
        lo[e] = np.maximum(0.0, Y - (V - N)).sum() / Ns
        hi[e] = np.minimum(Y, N).sum() / Ns
        pred[e] = (wt * s[f"p_{e}"].to_numpy()).sum() / ws
    if lo["turnout"] > MIN_TURNOUT:
        for p in common.PARTIES:
            lo[f"{p}_v"] = lo[p] / hi["turnout"]
            hi[f"{p}_v"] = min(hi[p] / lo["turnout"], 1.0)
            pred[f"{p}_v"] = pred[p] / max(pred["turnout"], 1e-9)
    return {e: max(lo[e] - pred[e], pred[e] - hi[e], 0.0) for e in lo}


def bootstrap(t, rng):
    """Block bootstrap of the aggregate miss, resampling polling districts."""
    blocks = t.groupby("dm").indices
    keys = list(blocks)
    draws = {e: [] for e in ALL_ESTIMANDS}
    for _ in range(NBOOT):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([blocks[keys[k]] for k in pick])
        m = aggregate_miss(t, rows)
        for e in draws:
            if e in m:
                draws[e].append(m[e])
    return {e: np.array(v) for e, v in draws.items() if v}


def saluran_residuals(t, el, dim, g):
    """The per-stream misses, kept for the appendix diagnostic."""
    N, V, wt = t.N.to_numpy(), t.V.to_numpy(), t.wt.to_numpy()
    lo, hi, pred = {}, {}, {}
    for e in ESTIMAND_NAMES:
        Y = t[f"Y_{e}"].to_numpy()
        lo[e] = np.maximum(0.0, Y - (V - N)) / N
        hi[e] = np.minimum(Y, N) / N
        pred[e] = t[f"p_{e}"].to_numpy()
    ok = lo["turnout"] > MIN_TURNOUT
    for p in common.PARTIES:
        e = f"{p}_v"
        pred[e] = pred[p] / np.maximum(pred["turnout"], 1e-9)
        lo[e] = np.where(ok, lo[p] / np.maximum(hi["turnout"], 1e-9), np.nan)
        hi[e] = np.where(ok, np.minimum(hi[p] / np.maximum(lo["turnout"], 1e-9), 1.0), np.nan)

    frames = []
    for e in ALL_ESTIMANDS:
        miss = np.maximum.reduce([lo[e] - pred[e], pred[e] - hi[e], np.zeros_like(pred[e])])
        k = np.isfinite(miss)
        frames.append(pd.DataFrame(dict(
            election=el, dim=dim, group=g, estimand=e, dm=t.dm.to_numpy()[k],
            n_group=N[k], pred=pred[e][k], lo=lo[e][k], hi=hi[e][k],
            miss=miss[k], inside=miss[k] <= 1e-12,
            group_electorate=t.group_electorate.iloc[0])))
    return frames


def cluster_separation(els):
    """Why the reportable/greyed split needs no finely chosen threshold.

    The error bar is the variability of the miss across polling districts, so a group needs a
    held-out stratum spanning enough districts for that variability to be estimable. What the
    data give is a gap: every group spans either >=19 districts or <=4, nothing between. Any
    threshold in that range therefore classifies every group identically, which is the actual
    justification -- the chosen value is incidental.
    """
    rows = []
    for el in els:
        df = common.load(el)
        V = df.V.to_numpy(float)
        for g in common.ETH + common.AGE:
            dim = "ethnicity" if g in common.ETH else "age"
            N = common.counts(df, dim)[:, common.groups(dim).index(g)].astype(float)
            m = (N / V) >= hold.THR
            k = int(df.dm[m].nunique()) if m.sum() else 0
            rows.append(dict(election=common.cfg(el)["short"], group=g, salurans=int(m.sum()),
                             clusters=k, coverage=round(100 * N[m].sum() / N.sum(), 1),
                             reported=k >= hold.MIN_CLUSTERS))
    d = pd.DataFrame(rows)
    print("\n\nPolling districts spanned by each held-out stratum\n")
    print(d.pivot_table(index="group", columns="election", values="clusters")
           .reindex(common.ETH + common.AGE)
           .reindex(columns=[common.cfg(c)["short"] for c in common.ORDER])
           .astype(int).to_string())
    lo, hi = d[d.reported].clusters.min(), d[~d.reported].clusters.max()
    print(f"\nlowest reported {lo}   highest greyed {hi}   -> any threshold in [{hi + 1}, {lo}]")
    print(f"gives the identical classification; we use {hold.MIN_CLUSTERS}.")
    marginal = sorted(set(d[d.reported & (d.clusters < 30)].group))
    print(f"below the conventional 30-cluster floor, flagged in the text: {marginal}")
    return d


def propagate_scaled(S):
    """Give each vote-share bar the worst-case error of the two quantities it is derived from.

    Scoring a ratio against its own (much wider) bounds understates its error whenever the
    numerator and denominator errors cancel, which is exactly when the estimate looks best and
    is not. We keep the larger of the measured miss and the propagated one.
    """
    marg = pd.read_csv(f"{OUT}/joint_marginals.csv")
    scaled = pd.read_csv(f"{OUT}/joint_scaled.csv")
    idx = S.set_index(["election", "group", "estimand"]).err

    def prop(r):
        if not r.estimand.endswith("_v"):
            return r.err
        p = r.estimand[:-2]
        sh = scaled[(scaled.election == r.election) & (scaled.group == r.group)
                    & (scaled.party == p)].share_of_voters
        tn = marg[(marg.election == r.election) & (marg.group == r.group)
                  & (marg.estimand == "turnout")].rate
        if sh.empty or tn.empty or float(tn.iloc[0]) <= 0:
            return r.err
        e_p, e_t = idx.get((r.election, r.group, p)), idx.get((r.election, r.group, "turnout"))
        if e_p is None or e_t is None:
            return r.err
        return max(r.err, (e_p + float(sh.iloc[0]) * e_t) / float(tn.iloc[0]))

    S = S.copy()
    S["err_measured"] = S.err
    S["err"] = S.apply(prop, axis=1)
    return S


THRESHOLDS = [0.90, 0.95, 0.98]


def misses_at(thr, elections):
    """Aggregate holdout miss per group x election x estimand at one purity threshold."""
    orig = hold.design_masks
    hold.design_masks = hold.design_masks = lambda df, dim, design, t=thr: orig(
        df, dim, design, thr=t)
    try:
        rows = []
        for el in elections:
            df = jm.load_joint(el)
            for dim in common.DIMS:
                keep = sorted({r["group"] for r in hold.holdout(df, el, dim, "A")})
                for g, t in stream_table(df, el, dim, keep).items():
                    for e, v in aggregate_miss(t).items():
                        rows.append(dict(election=el, group=g, estimand=e,
                                         miss=v, n_pure=len(t), thr=thr))
        return pd.DataFrame(rows)
    finally:
        hold.design_masks = hold.design_masks = orig


def threshold_main():
    els = common.arg_elections()
    d = pd.concat([misses_at(t, els) for t in THRESHOLDS], ignore_index=True)
    d.to_csv(f"{OUT}/threshold_sensitivity.csv", index=False)

    p = d.pivot_table(index=["election", "group", "estimand"], columns="thr", values="miss")
    p = p.dropna()
    print("Aggregate holdout miss (pp), mean over all group x election x estimand\n")
    print((100 * p.mean()).round(2).to_string())
    print(f"\nheld-out streams (mean per group-election): "
          f"{d.groupby('thr').n_pure.mean().round(0).to_dict()}")
    print(f"\ncells compared: {len(p)}")
    print("\nStricter purity narrows the permitted interval, so the same prediction error is")
    print("more often caught outside it: the miss rises with the sharpness of the test.")
    print(f"\nwrote {OUT}/threshold_sensitivity.csv")


def main():
    els = common.arg_elections()
    rng = np.random.default_rng(SEED)
    res, errs = [], []
    for el in els:
        df = jm.load_joint(el)
        for dim in DIMS:
            keep = sorted({r["group"] for r in hold.holdout(df, el, dim, "A")})
            for g, t in stream_table(df, el, dim, keep).items():
                res += saluran_residuals(t, el, dim, g)
                point = aggregate_miss(t)
                draws = bootstrap(t, rng)
                for e, d in draws.items():
                    errs.append(dict(
                        election=el, dim=dim, group=g, estimand=e,
                        err=float(np.sqrt((d ** 2).mean())),
                        point_miss=point.get(e, np.nan),
                        boot_p90=float(np.quantile(d, 0.90)),
                        n_pure=int(len(t)), n_dm=int(t.dm.nunique()),
                        holdout_share=100 * t.N.sum() / t.group_electorate.iloc[0]))
        print(f"  {el}: done")

    R = pd.concat(res, ignore_index=True)
    R.to_csv(f"{OUT}/validation_residuals.csv", index=False)
    S = propagate_scaled(pd.DataFrame(errs))
    S.to_csv(f"{OUT}/validation_errors.csv", index=False)

    print("\nHeld-out validation error (pp), design A, per estimand\n")
    p = 100 * S.pivot_table(index=["group", "estimand"], columns="election", values="err")
    p = p.reindex(columns=[c for c in common.ORDER if c in p.columns])
    p.columns = [common.cfg(c)["short"] for c in p.columns]
    order = [(g, e) for g in common.ETH + common.AGE for e in ALL_ESTIMANDS]
    print(p.reindex([o for o in order if o in p.index]).round(2).to_string())

    sal = (R.groupby(["election", "group", "estimand"])
             .apply(lambda s: np.sqrt(np.average(s.miss ** 2, weights=s.n_group)),
                    include_groups=False))
    j = S.set_index(["election", "group", "estimand"]).err
    print(f"\nScale check -- mean saluran-level RMSE {100 * sal.mean():.2f}pp vs mean statewide "
          f"error bar {100 * j.mean():.2f}pp ({sal.mean() / j.mean():.1f}x)")
    print("(single-stream prediction error is dominated by genuine local variation,\n"
          " which averages out of a statewide estimate; hence the aggregate scale)")

    cluster_separation(els).to_csv(f"{OUT}/cluster_separation.csv", index=False)

    got = set(S.group)
    print(f"\nNo pure stratum, no error bar, not citable: "
          f"{[g for g in common.ETH + common.AGE if g not in got]}")
    print(f"\nwrote {OUT}/validation_residuals.csv, {OUT}/validation_errors.csv")


if __name__ == "__main__":
    main()
    threshold_main()
