"""
Out-of-sample validation against REAL ground truth. Usage: python3 groundtruth.py [election ...]

Standalone: reads the existing outputs and modules, writes only its own files. Nothing in
00-08 is touched or re-run.

THE IDEA
validate.py measures error against a truth we planted ourselves, which tests the estimator
but not the electorate: if real Johor voters violate the model in a way our simulation did not
imagine, world B will never say so. What is wanted is a truth we did not invent.

Malaysia supplies one for free. SPR sorts voters into salurans by age, so a large share of
salurans contain exactly one age band. In such a saluran the group's turnout and vote choice
are *directly observed* -- there is nothing to infer, the margin is the answer. That gives a
held-out stratum of real behaviour to predict:

    fit on the MIXED salurans only  ->  predict the PURE ones  ->  compare to what they did

This is the test simulation cannot run. It measures whether rates estimated where groups live
side-by-side transport to places where they do not -- which is exactly the constant-rate
assumption the whole project leans on, checked against real voters.

GROUND TRUTH IS AN INTERVAL, NOT A POINT
Requiring exact purity is too strict away from the age dimension (Johor has 11 exactly-pure
ethnic salurans). At a threshold below 1.0 a "pure" saluran still holds a few outsiders, so
the group's rate is not observed exactly -- but it is *bounded*, by the same Frechet argument
as everywhere else, and the bound is tight because the contamination is small. So the truth is
computed as a bound over the pure subset rather than asserted as a point. At 95% purity the
age truth intervals are 0.2pp wide, which is a point for practical purposes; the Malay one is
2.0pp wide, which is still far tighter than anything the model produces.

A prediction "passes" if it lands inside the truth interval. The reported miss is the distance
outside it, so a tight truth interval is held to a tight standard and a loose one is not.

WHAT THIS CAN AND CANNOT VALIDATE
Coverage is what it is, and it is not uniform:
  age        18-29 / 30-49 / 50-69 have 38-56% of their electorate in pure salurans -- strong.
             70+ has ~4% -- weak, report but do not lean on.
  ethnicity  Malay has ~25% -- usable. Chinese ~3% -- marginal.
             Indian and Other have NO pure salurans at any usable threshold.
The last line is not a gap in the test, it is the test returning a verdict: the groups this
design cannot check are precisely the groups 07 already stars as uncitable. The starring is
not a hedge, it is a description of which parameters Johor's geography can and cannot identify.

THE CONFOUND, STATED PLAINLY
Pure and mixed salurans are not a random split. A saluran is age-pure because its polling
place had enough voters to fill a whole stream from one age band, so pure salurans sit in
larger, denser localities and mixed ones in smaller, more rural ones. A gap between prediction
and truth therefore mixes two things: the aggregation bias we want to measure, and a genuine
urban/rural difference in behaviour.

They cannot be fully separated, but the design narrows them:
  DESIGN A  fit on all mixed salurans, predict all pure ones. Maximum power, fully confounded.
  DESIGN B  restrict both sides to dms that contain BOTH pure and mixed salurans, so the
            comparison is made within the same polling place. Locality is held roughly fixed
            and what remains is much closer to the aggregation bias alone.
Report both. Where they agree, locality was not driving the gap.

Note also that arguably no separation is needed for the headline claim: if a group's rate
genuinely differs between dense and sparse localities, that IS a context effect, and a model
asserting one constant rate per group is wrong in exactly the way this test is designed to
catch. B is the stricter reading; A is the one that matches how the estimates are used.
"""

import os

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES, ORDER

import bounds as bnd
import models
import reconcile as rec

OUT = common.OUT
THR = 0.95          # purity threshold; swept in the sensitivity table
MIN_CLUSTERS = 10   # polling districts the pure stratum must span to be testable; see holdout.py
NBOOT = 150
RNG = np.random.default_rng(20260717)


# ---------------------------------------------------------------- split
def split(df, dim, thr=THR):
    """(pure mask, index of the dominant group in each saluran)."""
    X = common.shares(df, dim)
    return X.max(axis=1) >= thr - 1e-9, X.argmax(axis=1)


def truth_interval(df, el, dim, j, mask):
    """Frechet bounds on group j's rates over `mask`, i.e. the observed truth + slack."""
    out = {}
    V = df.V.to_numpy(float)[mask]
    N = common.counts(df, dim)[mask, j]
    for e in ESTIMAND_NAMES:
        Y = common.outcome(df, el, e)[mask]
        lo = np.maximum(0.0, Y - (V - N)).sum() / N.sum()
        hi = np.minimum(Y, N).sum() / N.sum()
        out[e] = (lo, hi)
    return out


# ---------------------------------------------------------------- predictors
def predict(fit_df, el, dim):
    """Every estimator, fitted on `fit_df` (the mixed salurans) only."""
    X = common.shares(fit_df, dim)
    V = fit_df.V.to_numpy(float)
    preds = {}

    seed = rec.build_seed(fit_df, el, dim)
    total, _ = rec.reconcile(fit_df, el, dim, seed)
    N_tot = common.counts(fit_df, dim).sum(axis=0)
    preds["reconciled"] = {e: rec.rates(total, N_tot, el, e) for e in ESTIMAND_NAMES}

    preds["M2_logit"] = {}
    preds["M3_probit"] = {}
    preds["M1_lpm"] = {}
    preds["bivariate"] = {}
    for e in ESTIMAND_NAMES:
        Y = common.outcome(fit_df, el, e)
        preds["M2_logit"][e], _ = models.fit_link(X, Y, V, "logit")
        preds["M3_probit"][e], _ = models.fit_link(X, Y, V, "probit")
        preds["M1_lpm"][e] = models.fit_lpm(X, Y, V)
        # the scatterplot version: fit each group's share on its own, read off at x=1
        biv = []
        for j in range(X.shape[1]):
            A = np.column_stack([np.ones(len(X)), X[:, j]])
            w = np.sqrt(V)
            c, *_ = np.linalg.lstsq(A * w[:, None], (Y / V) * w, rcond=None)
            biv.append(c[0] + c[1])
        preds["bivariate"][e] = np.array(biv)
    return preds


def boot_se(fit_df, el, dim, nboot=NBOOT):
    """Cluster bootstrap over seats on the reconciled predictor, fitted on mixed only."""
    cls = fit_df.cluster.unique()
    idx = {c: np.flatnonzero((fit_df.cluster == c).to_numpy()) for c in cls}
    draws = []
    for _ in range(nboot):
        pick = RNG.choice(cls, size=len(cls), replace=True)
        d2 = fit_df.iloc[np.concatenate([idx[c] for c in pick])]
        try:
            seed = rec.build_seed(d2, el, dim)
            total, _ = rec.reconcile(d2, el, dim, seed)
            N_tot = common.counts(d2, dim).sum(axis=0)
            draws.append({e: rec.rates(total, N_tot, el, e) for e in ESTIMAND_NAMES})
        except Exception:
            continue
    return {e: np.array([d[e] for d in draws]).std(axis=0) for e in ESTIMAND_NAMES}


# ---------------------------------------------------------------- designs
def run_design(df, el, dim, design, thr=THR):
    pure, arg = split(df, dim, thr)
    gs = common.groups(dim)

    if design == "B":
        # keep only dms holding both kinds, so prediction and truth share a locality
        t = pd.DataFrame({"dm": df.dm.to_numpy(), "pure": pure})
        agg = t.groupby("dm").pure.agg(["sum", "count"])
        both = set(agg.index[(agg["sum"] > 0) & (agg["sum"] < agg["count"])])
        keep = df.dm.isin(both).to_numpy()
        pure = pure & keep
        mixed = (~pure) & keep
    else:
        mixed = ~pure

    if mixed.sum() < 200:
        return []
    fit_df = df[mixed].reset_index(drop=True)
    preds = predict(fit_df, el, dim)
    ses = boot_se(fit_df, el, dim)

    rows = []
    Nall = common.counts(df, dim)
    for j, g in enumerate(gs):
        pm = pure & (arg == j)
        if pm.sum() == 0:
            continue
        cover = Nall[pm, j].sum() / Nall[:, j].sum()
        if df.dm[pm].nunique() < MIN_CLUSTERS:
            continue
        ti = truth_interval(df, el, dim, j, pm)
        for e in ESTIMAND_NAMES:
            lo, hi = ti[e]
            for m, pv in preds.items():
                p = pv[e][j]
                miss = max(lo - p, p - hi, 0.0)      # distance outside the truth interval
                rows.append(dict(
                    election=el, dim=dim, design=design, thr=thr, group=g, estimand=e,
                    model=m, pred=p, truth_lo=lo, truth_hi=hi, truth_mid=(lo + hi) / 2,
                    truth_width=hi - lo, miss=miss, inside=bool(miss <= 1e-9),
                    signed=p - (lo + hi) / 2,
                    se=(ses[e][j] if m == "reconciled" else np.nan),
                    n_pure=int(pm.sum()), n_mixed=int(mixed.sum()), coverage=cover,
                ))
    return rows


def selection_diagnostic(df, dim, thr=THR):
    """How unlike each other are the pure and mixed salurans? The confound, quantified."""
    pure, _ = split(df, dim, thr)
    V = df.V.to_numpy(float)
    turn = 1 - df.NONVOTE.to_numpy(float) / V
    dmsize = df.groupby("dm").V.transform("sum").to_numpy(float)
    other = "ethnicity" if dim == "age" else "age"
    Xo = common.shares(df, other)
    d = {
        "n_salurans": (pure.sum(), (~pure).sum()),
        "mean_electorate": (V[pure].mean(), V[~pure].mean()),
        "mean_dm_electorate": (dmsize[pure].mean(), dmsize[~pure].mean()),
        "turnout": (np.average(turn[pure], weights=V[pure]),
                    np.average(turn[~pure], weights=V[~pure])),
    }
    for k, gg in enumerate(common.groups(other)):
        d[f"share_{gg}"] = (np.average(Xo[pure, k], weights=V[pure]),
                            np.average(Xo[~pure, k], weights=V[~pure]))
    return d


# ---------------------------------------------------------------- main
def main():
    els = common.arg_elections()
    rows, diags = [], []
    for el in els:
        df = common.load(el)
        for dim in DIMS:
            for design in ["A", "B"]:
                rows += run_design(df, el, dim, design)
            for thr in [0.99, 0.90, 0.85]:      # sensitivity to the purity threshold
                rows += run_design(df, el, dim, "A", thr=thr)
            dg = selection_diagnostic(df, dim)
            diags.append(dict(election=el, dim=dim,
                              **{k: v[0] for k, v in dg.items()},
                              **{f"{k}_mixed": v[1] for k, v in dg.items()}))
        print(f"  {el}: done")

    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/groundtruth.csv", index=False)
    pd.DataFrame(diags).to_csv(f"{OUT}/groundtruth_selection.csv", index=False)

    main_r = r[(r.thr == THR) & (r.design == "A")]

    print("\n" + "=" * 100)
    print(f"OUT-OF-SAMPLE VALIDATION AGAINST REAL GROUND TRUTH  (purity >= {THR:.0%}, design A)")
    print("Fitted on MIXED salurans only; compared to what the PURE salurans actually did.")
    print("=" * 100)
    for el in els:
        for dim in DIMS:
            s = main_r[(main_r.election == el) & (main_r.dim == dim)]
            if s.empty:
                continue
            print(f"\n### {common.cfg(el)['label']} -- {dim}\n")
            print(f"{'group':9s} {'estimand':9s} {'TRUTH':>14s} {'reconciled':>12s} "
                  f"{'M2_logit':>10s} {'M3_probit':>10s} {'M1_lpm':>9s} {'bivariate':>10s}")
            for g in s.group.unique():
                for e in ESTIMAND_NAMES:
                    q = s[(s.group == g) & (s.estimand == e)].set_index("model")
                    if q.empty:
                        continue
                    t = q.iloc[0]
                    cells = []
                    for m in ["reconciled", "M2_logit", "M3_probit", "M1_lpm", "bivariate"]:
                        v = q.loc[m]
                        mark = "" if v.inside else "x"
                        cells.append(f"{100 * v.pred:6.1f}{mark}")
                    print(f"{g:9s} {e:9s} {100 * t.truth_lo:6.1f}-{100 * t.truth_hi:<7.1f}"
                          + "".join(f"{c:>12s}" for c in cells[:1])
                          + "".join(f"{c:>10s}" for c in cells[1:]))
                print()

    print("\n" + "=" * 100)
    print("THE HORSE RACE -- mean |miss| outside the truth interval, in pp")
    print("(0.0 = always landed inside the real, observed answer)")
    print("=" * 100)
    tab = (main_r.groupby(["dim", "model"]).miss.agg(
        mean=lambda x: 100 * x.mean(), max=lambda x: 100 * x.max()))
    tab["inside_%"] = main_r.groupby(["dim", "model"]).inside.mean() * 100
    print(tab.round(2).to_string())
    print("\noverall:")
    ov = (main_r.groupby("model").miss.agg(mean_pp=lambda x: 100 * x.mean(),
                                           max_pp=lambda x: 100 * x.max()))
    ov["inside_%"] = main_r.groupby("model").inside.mean() * 100
    print(ov.sort_values("mean_pp").round(2).to_string())

    print("\n" + "=" * 100)
    print("DESIGN A vs B -- is the gap aggregation bias, or just urban/rural?")
    print("B restricts both sides to dms holding pure AND mixed salurans, fixing locality.")
    print("=" * 100)
    ab = r[(r.thr == THR) & (r.model == "reconciled")]
    cmp = ab.pivot_table(index=["dim", "group", "estimand"], columns="design", values="miss") * 100
    if {"A", "B"} <= set(cmp.columns):
        cmp["shrinks_under_B"] = cmp.B < cmp.A
        print(cmp.round(2).to_string())
        print(f"\nmean |miss| A {cmp.A.mean():.2f}pp vs B {cmp.B.mean():.2f}pp -- "
              f"B smaller in {100 * cmp.shrinks_under_B.mean():.0f}% of cells")
        print("A gap that shrinks under B was partly locality; one that does not is the"
              "\nconstant-rate assumption failing on its own.")

    print("\n" + "=" * 100)
    print("SENSITIVITY TO THE PURITY THRESHOLD (reconciled, design A)")
    print("=" * 100)
    sw = r[r.model == "reconciled"].pivot_table(index=["dim", "group"], columns="thr",
                                                values="miss", aggfunc="mean") * 100
    print(sw.round(2).to_string())

    print("\n" + "=" * 100)
    print("THE CONFOUND: how do pure and mixed salurans differ?")
    print("=" * 100)
    dd = pd.DataFrame(diags)
    for _, x in dd[dd.election == els[-1]].iterrows():
        print(f"\n{x.dim}:  (pure vs mixed)")
        for k in ["n_salurans", "mean_electorate", "mean_dm_electorate", "turnout"]:
            print(f"  {k:20s} {x[k]:10.3f}  vs {x[k + '_mixed']:10.3f}")

    print(f"\nwrote {OUT}/groundtruth.csv, groundtruth_selection.csv")
    print("\nNOTE: Indian and Other never appear above -- Johor has no ethnically pure")
    print("salurans for them at any usable threshold, so this design cannot test them.")
    print("That is the same fact that makes 07 star them as uncitable.")


if __name__ == "__main__":
    main()
