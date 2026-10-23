"""Direct validation of the joint ethnicity-by-age cells against near-pure JOINT salurans.
Usage: python3 jointholdout.py [election ...]

WHY THIS EXISTS. The crosstab is the paper's headline object, but its cells previously carried
the larger of the two MARGINAL validation errors (ethnicity, age) as a proxy. That proxy is
not conservative: errors in different joint cells can cancel inside either marginal, so a
cell can be wrong by more than both marginals. The only honest error for a joint cell is one
measured against that cell's own behaviour -- and the sorting rule plus residential
geography make that possible for a useful number of cells: a stream that is 95% Malay AND
95% one age band is 95% a single joint cell.

DESIGN. Identical to errors.py, with the joint cell in place of the marginal group:

    pure_i   : max over the 24 joint shares of saluran i is >= thr
    fit      : the 24-row model on every non-pure saluran
    target   : the exact interval [L, U] the margins permit for the dominant cell, aggregated
               over its pure stratum (equation aggmiss in the manuscript)
    predict  : the model's rate for THAT cell -- no composition weighting is needed, since
               the target is a single cell
    score    : max(L - p, p - U, 0)   (lower envelope)
               max(|p - L|, |p - U|)   (upper envelope)
    error bar: RMS of each envelope across 2,000 district-block bootstrap replicates, for
               cells whose pure stratum spans >= MIN_CLUSTERS polling districts at thr=0.95

THRESHOLD GRID. The holdout is run at 0.75, 0.80, 0.90 and 0.95 so that the trade-off between
sharpness of the target (interval width) and the number of validatable cells is visible.
Bars in the manuscript's crosstab come from the 0.95 grid, the same threshold as the marginal
bars; cells with no adequate joint stratum carry NO bar.

Outputs
    output/joint_validation.csv         every eligible cell x estimand x threshold
    output/joint_validation_errors.csv  the 0.95 bars with bootstrap envelopes, per cell
"""

import numpy as np
import pandas as pd

import common
from common import ESTIMAND_NAMES

import joint as jm
import reconcile as rec
import holdout as hold

OUT = common.OUT
JOINT, JCOL = jm.JOINT, jm.JCOL
THRESHOLDS = [0.75, 0.80, 0.90, 0.95]
REPORT_THR = 0.95
MIN_CLUSTERS = hold.MIN_CLUSTERS
NBOOT = 2000
SEED = 20260711


def stream_tables(df, el, thr):
    """Per-cell frames of the pure stratum: Y, V, N (cell count), prediction, district."""
    X = jm.jshares(df)
    pure = X.max(axis=1) >= thr - 1e-9
    arg = X.argmax(axis=1)
    fit_df = df[~pure].reset_index(drop=True)
    per, _ = jm.fit_joint_tables(fit_df, el)
    # each held-out stream predicted with its own seat's rates (rec.seat_rates)
    R = rec.seat_rates(per, jm.jcounts(fit_df), fit_df.cluster.to_numpy(),
                       df.cluster.to_numpy(), el)

    Nj, V = jm.jcounts(df), df.V.to_numpy(float)
    out = {}
    for k, (eth, age) in enumerate(JOINT):
        m = pure & (arg == k)
        if m.sum() == 0:
            continue
        t = dict(dm=df.dm.to_numpy()[m], V=V[m], N=Nj[m, k],
                 cell_electorate=Nj[:, k].sum())
        for e in ESTIMAND_NAMES:
            t[f"Y_{e}"] = common.outcome(df, el, e)[m]
            t[f"p_{e}"] = R[e][m, k]
        out[(eth, age)] = pd.DataFrame(t)
    return out, int((~pure).sum())


def envelope(t, rows=None):
    """(lower miss, upper distance, L, U, p) per estimand over a (resampled) stratum."""
    s = t if rows is None else t.iloc[rows]
    N, V = s.N.to_numpy(), s.V.to_numpy()
    Ns = N.sum()
    out = {}
    for e in ESTIMAND_NAMES:
        Y = s[f"Y_{e}"].to_numpy()
        L = np.maximum(0.0, Y - (V - N)).sum() / Ns
        U = np.minimum(Y, N).sum() / Ns
        p = float((N * s[f"p_{e}"].to_numpy()).sum() / Ns)   # cell-weighted, own-seat rates
        out[e] = (max(L - p, p - U, 0.0), max(abs(p - L), abs(p - U)), L, U, p)
    return out


def bootstrap(t, rng):
    blocks = t.groupby("dm").indices
    keys = list(blocks)
    lo = {e: [] for e in ESTIMAND_NAMES}
    hi = {e: [] for e in ESTIMAND_NAMES}
    for _ in range(NBOOT):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([blocks[keys[k]] for k in pick])
        m = envelope(t, rows)
        for e in ESTIMAND_NAMES:
            lo[e].append(m[e][0])
            hi[e].append(m[e][1])
    return ({e: float(np.sqrt((np.array(v) ** 2).mean())) for e, v in lo.items()},
            {e: float(np.sqrt((np.array(v) ** 2).mean())) for e, v in hi.items()})


def main():
    els = common.arg_elections()
    rng = np.random.default_rng(SEED)
    grid, bars = [], []
    for el in els:
        df = jm.load_joint(el)
        for thr in THRESHOLDS:
            tabs, n_fit = stream_tables(df, el, thr)
            for (eth, age), t in tabs.items():
                n_dm = int(t.dm.nunique())
                point = envelope(t)
                for e, (miss, sup, L, U, p) in point.items():
                    grid.append(dict(
                        election=el, thr=thr, eth=eth, age=age, estimand=e,
                        pred=p, lo=L, hi=U, width=U - L, miss=miss, upper=sup,
                        signed=p - (L + U) / 2, n_pure=len(t), n_dm=n_dm, n_fit=n_fit,
                        coverage=100 * t.N.sum() / t.cell_electorate.iloc[0],
                        eligible=n_dm >= MIN_CLUSTERS))
                if thr == REPORT_THR and n_dm >= MIN_CLUSTERS:
                    blo, bhi = bootstrap(t, rng)
                    for e in ESTIMAND_NAMES:
                        bars.append(dict(
                            election=el, eth=eth, age=age, estimand=e,
                            err_lower=blo[e], err_upper=bhi[e],
                            point_miss=point[e][0], width=point[e][3] - point[e][2],
                            n_pure=len(t), n_dm=n_dm,
                            coverage=100 * t.N.sum() / t.cell_electorate.iloc[0]))
        print(f"  {el}: done")

    G = pd.DataFrame(grid)
    B = pd.DataFrame(bars)
    G.to_csv(f"{OUT}/joint_validation.csv", index=False)
    B.to_csv(f"{OUT}/joint_validation_errors.csv", index=False)

    print("\nEligible joint cells (>= %d districts) by threshold and election\n" % MIN_CLUSTERS)
    e0 = G[(G.estimand == "turnout")]
    print(e0.pivot_table(index=["eth", "age"], columns=["thr"], values="n_dm",
                         aggfunc="min").fillna(0).astype(int).to_string())
    print("\nAggregate miss (pp) over eligible cell-estimands, by threshold")
    el_ = G[G.eligible]
    s = el_.groupby("thr").agg(mean_miss=("miss", "mean"), max_miss=("miss", "max"),
                               mean_width=("width", "mean"), cells=("miss", "size"))
    s[["mean_miss", "max_miss", "mean_width"]] *= 100
    print(s.round(2).to_string())
    print(f"\nDirect bars at {REPORT_THR}: {len(B)} cell-estimands; "
          f"mean lower {100 * B.err_lower.mean():.2f}, mean upper {100 * B.err_upper.mean():.2f}")
    print("\nWorst five cell-estimands at 0.95:")
    w = el_[el_.thr == REPORT_THR].sort_values("miss", ascending=False).head(5)
    print((w[["election", "eth", "age", "estimand", "pred", "lo", "hi", "miss"]]
           .assign(**{c: lambda d, c=c: (100 * d[c]).round(2) for c in ["pred", "lo", "hi", "miss"]})
           .to_string(index=False)))
    print(f"\nwrote {OUT}/joint_validation.csv, {OUT}/joint_validation_errors.csv")


if __name__ == "__main__":
    main()
