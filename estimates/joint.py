"""
The joint ethnicity x age model: 24 rows instead of separate 4-row and 6-row problems.
Usage: python3 joint.py [election ...]

Standalone: builds its own margins, reads the existing per-election saluran.parquet for the
choice columns, reuses the fitting/raking functions from 02 and 03. Nothing in 00-09 is
touched or re-run.

WHY
groundtruth.py established, against real observed behaviour, that the separate age model
fails: 70+ BN misses by up to ~19pp and 50-69 BN by ~6pp, while every ethnic estimate misses by
~0. The diagnosis was specific. SPR sorts voters into salurans by age, and Chinese Malaysians
skew old, so "the oldest stream" is mechanically the Chinese stream -- Johor's age-pure 70+
salurans are 86% Chinese against 48% for 70+ overall. A model with one rate per age band
cannot tell a 70+ Chinese voter from a 70+ Malay voter and assigns them the same number. It
is wrong in exactly the way the data guarantees it will be wrong.

THE FIX COSTS NOTHING
The public roll carries ethnicity and birth_year *on the same individual*. So the 24-cell
ethnicity x age margin is not modelled, assumed or imputed -- it is counted, exactly, per
saluran, the same way the marginal counts were. Using marginal counts when the joint margin was
sitting in the same file was the error.

Rows become the 24 joint cells; columns are unchanged. Everything else in the pipeline is
identical, which is the point: this is a change of row definition, not of method.

WHAT COMES BACK
  - the crosstab itself, which is the object anyone actually wants ("is BN's Malay support
    age-graded?" rather than "what is 70+ support?", which was never really the question)
  - the marginals, free, by summing the joint cells over the other dimension -- so nothing
    the separate model produced is lost
  - identification: at SE-16, 196 salurans (4.0%) are >90% a single joint cell, 13.6% >75%.

WHAT DOES NOT COME BACK
The Indian and Other cells stay unidentified -- their joint shares have sd 0.017-0.054 and no
saluran is anywhere near pure on them. The joint model does not rescue them and does not
pretend to. That is the same verdict as before, for the same reason.

THE TEST
The holdout from 09 is re-run with joint rows. The difference is that the joint model can now
predict a *specific stratum's* composition rather than a state average:

    predicted rate for age band a in the pure stratum
        = sum_e  N_pure[e,a] * b[e,a]  /  sum_e N_pure[e,a]

i.e. it is asked what 70+ voters who are 86% Chinese should do, instead of what the average
70+ voter should do. If the diagnosis in 09 was right, this closes the gap. If it does not
close, the diagnosis was wrong and the age failure is something else -- which would be worth
knowing too.
"""

import os

import duckdb
import numpy as np
import pandas as pd

import common
from common import ESTIMAND_NAMES, ORDER

import models
import reconcile as rec

OUT = common.OUT
ETH, AGE = common.ETH, common.AGE
JOINT = [(e, a) for e in ETH for a in AGE]
JCOL = [f"{e}|{a}" for e, a in JOINT]
THR = 0.95


def joint_margins(el):
    """The 24-cell ethnicity x age registered count per saluran, counted off the roll.

    Same exclusions as data.py: the early+postal bloc is out.
    """
    c = common.cfg(el)
    yr = c["year"]
    q = duckdb.query(f"""
        SELECT dm, pm, CAST(saluran AS INTEGER) AS saluran,
          CASE WHEN ethnicity IN ('Malay','Chinese','Indian') THEN ethnicity
               ELSE 'Other' END AS eth,
          CASE {" ".join(f"WHEN {common.age_sql(b, yr)} THEN '{b}'" for b in AGE[:-1])}
               ELSE '{AGE[-1]}' END AS age,
          COUNT(*) AS n
        FROM read_parquet('{c["roll"]}')
        WHERE state = '{common.STATE}' AND dm NOT LIKE '%/UP %' AND dm NOT LIKE '%/00 %'
        GROUP BY 1,2,3,4,5
    """).df()
    q["cell"] = q.eth + "|" + q.age
    w = (q.pivot_table(index=["dm", "pm", "saluran"], columns="cell", values="n",
                       fill_value=0).reset_index())
    for cc in JCOL:
        if cc not in w.columns:
            w[cc] = 0
    return w[["dm", "pm", "saluran"] + JCOL]


def load_joint(el):
    df = common.load(el).merge(joint_margins(el), on=["dm", "pm", "saluran"], how="left")
    assert df[JCOL].notna().all().all(), "saluran missing joint margins"
    # the joint cells must reproduce the electorate exactly, or the margins do not line up
    bad = (df[JCOL].sum(axis=1) != df.V).sum()
    assert bad == 0, f"{el}: joint margins != electorate on {bad} salurans"
    return df


def jshares(df):
    X = df[JCOL].to_numpy(float)
    return X / X.sum(axis=1, keepdims=True)


def jcounts(df):
    return df[JCOL].to_numpy(float)


def fit_joint(df, el):
    """Seed on 24 rows, then rake every saluran. Returns total 24 x C counts."""
    X, V = jshares(df), df.V.to_numpy(float)
    CH = common.choices(el)
    seed = np.zeros((len(JCOL), len(CH)))
    for k, name in enumerate(CH):
        seed[:, k], _ = models.fit_link(X, df[name].to_numpy(float), V, "logit")
    N, Y = jcounts(df), df[CH].to_numpy(float)
    per = rec.ipf(seed, N, Y)
    return per.sum(axis=0), seed


def jrates(total, N_tot, el, estimand):
    CH = common.choices(el)
    cols = [CH.index(c) for c in common.estimands(el)[estimand]]
    return total[:, cols].sum(axis=1) / np.maximum(N_tot, 1e-9)


def marginal(total, N_tot, el, estimand, dim):
    """Collapse the 24 joint cells down to a one-dimension marginal, weighting by registration."""
    groups = common.DIMS[dim]
    num = np.zeros(len(groups))
    den = np.zeros(len(groups))
    CH = common.choices(el)
    cols = [CH.index(c) for c in common.estimands(el)[estimand]]
    for k, (e, a) in enumerate(JOINT):
        j = groups.index(e if dim == "ethnicity" else a)
        num[j] += total[k, cols].sum()
        den[j] += N_tot[k]
    return num / np.maximum(den, 1e-9)


# ---------------------------------------------------------------- the holdout test
def holdout(df, el, dim):
    """Fit on MIXED salurans, predict the PURE stratum using its own joint composition."""
    Xd = common.shares(df, dim)
    pure = Xd.max(axis=1) >= THR - 1e-9
    arg = Xd.argmax(axis=1)
    groups = common.DIMS[dim]
    fit_df = df[~pure].reset_index(drop=True)

    total, _ = fit_joint(fit_df, el)
    Nf = jcounts(fit_df).sum(axis=0)
    b = {e: jrates(total, Nf, el, e) for e in ESTIMAND_NAMES}      # 24 joint-cell rates

    # the separate-model comparator, fitted on the same mixed salurans
    Xs, Vs = common.shares(fit_df, dim), fit_df.V.to_numpy(float)
    sep_seed = rec.build_seed(fit_df, el, dim)
    sep_tot, _ = rec.reconcile(fit_df, el, dim, sep_seed)
    sep_N = common.counts(fit_df, dim).sum(axis=0)
    sep = {e: rec.rates(sep_tot, sep_N, el, e) for e in ESTIMAND_NAMES}

    Njoint = jcounts(df)
    V = df.V.to_numpy(float)
    N4 = common.counts(df, dim)
    rows = []
    for j, g in enumerate(groups):
        m = pure & (arg == j)
        if m.sum() == 0 or N4[m, j].sum() / N4[:, j].sum() < 0.02:
            continue
        # cells of the joint table that belong to this marginal group
        idx = [k for k, (e, a) in enumerate(JOINT) if (e if dim == "ethnicity" else a) == g]
        wts = Njoint[m][:, idx].sum(axis=0)          # the pure stratum's own composition
        for e in ESTIMAND_NAMES:
            Y = common.outcome(df, el, e)[m]
            lo = np.maximum(0.0, Y - (V[m] - N4[m, j])).sum() / N4[m, j].sum()
            hi = np.minimum(Y, N4[m, j]).sum() / N4[m, j].sum()
            # JOINT: what should 70+ voters who are 86% Chinese do?
            pj = float((wts * b[e][idx]).sum() / max(wts.sum(), 1e-9))
            # SEPARATE: what should the average 70+ voter do?
            ps = float(sep[e][j])
            for model, p in [("joint", pj), ("separate", ps)]:
                rows.append(dict(election=el, dim=dim, group=g, estimand=e, model=model,
                                 pred=p, truth_lo=lo, truth_hi=hi,
                                 miss=max(lo - p, p - hi, 0.0),
                                 inside=bool(lo - 1e-9 <= p <= hi + 1e-9)))
    return rows


# ---------------------------------------------------------------- main
def main():
    els = common.arg_elections()
    est_rows, hold_rows = [], []

    for el in els:
        df = load_joint(el)
        total, seed = fit_joint(df, el)
        N_tot = jcounts(df).sum(axis=0)

        for k, (e, a) in enumerate(JOINT):
            for est in ESTIMAND_NAMES:
                est_rows.append(dict(election=el, eth=e, age=a, estimand=est,
                                     registered=int(N_tot[k]),
                                     rate=float(jrates(total, N_tot, el, est)[k]),
                                     share_of_electorate=N_tot[k] / N_tot.sum()))
        for dim in common.DIMS:
            hold_rows += holdout(df, el, dim)
        print(f"  {el}: joint model fitted, holdout done")

    J = pd.DataFrame(est_rows)
    H = pd.DataFrame(hold_rows)
    J.to_csv(f"{OUT}/joint_estimates.csv", index=False)
    H.to_csv(f"{OUT}/joint_holdout.csv", index=False)

    # ---- the test that matters
    print("\n" + "=" * 92)
    print("DID THE JOINT MODEL FIX THE AGE FAILURE?  miss vs real ground truth, pp")
    print("=" * 92)
    piv = H.pivot_table(index=["dim", "group", "estimand"], columns="model",
                        values="miss", aggfunc="mean") * 100
    piv["improvement"] = piv.separate - piv.joint
    print(piv.round(2).to_string())
    print("\nmean miss by dimension:")
    s = H.pivot_table(index="dim", columns="model", values="miss", aggfunc="mean") * 100
    s["improvement"] = s.separate - s.joint
    print(s.round(2).to_string())
    print("\ninside the truth interval, %:")
    print((H.pivot_table(index="dim", columns="model", values="inside",
                         aggfunc="mean") * 100).round(1).to_string())

    # ---- the crosstab
    print("\n" + "=" * 92)
    print(f"THE CROSSTAB -- {common.cfg(els[-1])['label']}, % of that cell's registered voters")
    print("Cells with <1.5% of the electorate are unidentified; shown in brackets, do not cite.")
    print("=" * 92)
    last = J[J.election == els[-1]]
    for est in ESTIMAND_NAMES:
        print(f"\n### {est}\n")
        print(f"{'':10s}" + "".join(f"{a:>12s}" for a in AGE))
        for e in ETH:
            cells = []
            for a in AGE:
                r = last[(last.eth == e) & (last.age == a) & (last.estimand == est)].iloc[0]
                v = f"{100 * r.rate:.1f}"
                cells.append(f"[{v}]" if r.share_of_electorate < 0.015 else v)
            print(f"{e:10s}" + "".join(f"{c:>12s}" for c in cells))

    # ---- marginals: joint vs separate
    print("\n" + "=" * 92)
    print("MARGINALS: joint model vs the separate model already reported")
    print("=" * 92)
    ts = pd.read_csv(f"{OUT}/timeseries_estimates.csv")
    cmp_rows = []
    for el in els:
        df = load_joint(el)
        total, _ = fit_joint(df, el)
        N_tot = jcounts(df).sum(axis=0)
        for dim, gs in common.DIMS.items():
            for est in ESTIMAND_NAMES:
                mm = marginal(total, N_tot, el, est, dim)
                for j, g in enumerate(gs):
                    old = ts[(ts.election == el) & (ts.dim == dim) & (ts.group == g) &
                             (ts.estimand == est)]
                    cmp_rows.append(dict(election=el, dim=dim, group=g, estimand=est,
                                         joint=mm[j],
                                         separate=float(old.reconciled.iloc[0]) if len(old) else np.nan))
    C = pd.DataFrame(cmp_rows)
    C["shift"] = (C.joint - C.separate) * 100
    C.to_csv(f"{OUT}/joint_vs_separate.csv", index=False)
    last_c = C[C.election == els[-1]]
    print(f"\n{common.cfg(els[-1])['label']}: joint vs separate marginal (pp shift)\n")
    print(last_c.pivot_table(index=["dim", "group"], columns="estimand",
                             values="shift")[ESTIMAND_NAMES].round(2).to_string())
    # NB: C.shift is the DataFrame method, not the column -- index it explicitly.
    print("\nlargest marginal shifts across all elections:")
    print(C.reindex(C["shift"].abs().sort_values(ascending=False).index)
          .head(8)[["election", "dim", "group", "estimand", "separate", "joint", "shift"]]
          .round(3).to_string(index=False))
    print(f"\nmarginal shift, joint vs separate: mean |shift| "
          f"{C['shift'].abs().mean():.2f}pp, max {C['shift'].abs().max():.2f}pp")
    print(f"\nwrote {OUT}/joint_estimates.csv, joint_holdout.csv, joint_vs_separate.csv")


if __name__ == "__main__":
    main()
