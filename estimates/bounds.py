"""
Deterministic bounds on every estimand (the "ecology constraint").

For a saluran with electorate V, a demographic group of size N, and an outcome (turnout,
or votes for one party) with total Y, the number of that outcome contributed by the group,
y, is pinned by accounting alone:

    max(0, Y - (V - N))  <=  y  <=  min(Y, N)

The lower bound is the pigeonhole: even if every non-member picked the outcome, Y - (V - N)
of it still has to come from members. The upper bound is that members cannot supply more of
the outcome than there are members, or than there is outcome.

These are the Frechet-Hoeffding bounds for a 2-way table with known margins, and they are
*sharp*: both ends are attained by an actual feasible table. Nothing tightens them without
an assumption. Collapsing the other parties into a single "not this outcome" column loses
nothing -- the sharp bound on a cell of the full R x C table is the same expression.

Aggregating over salurans, the statewide rate for group g is bounded by

    sum_i max(0, Y_i - V_i + N_ig)          sum_i min(Y_i, N_ig)
    ------------------------------   <=  b_g  <=  --------------------
    sum_i N_ig                              sum_i N_ig

which is far tighter than the bound computed on the state as one aggregate, because each
saluran contributes its own accounting. This interval is assumption-free: it is where every
model in this project is required to land, and any model estimate outside it is
arithmetically impossible rather than merely unlikely.

Writes per-saluran bounds (consumed by reconcile.py) and a statewide/per-dun summary.
"""

import os

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES, ncol


def saluran_bounds(df, el, dim, estimand):
    """Per-saluran (lo, hi) count bounds. Shape: salurans x groups."""
    V = df.V.to_numpy(float)[:, None]
    N = common.counts(df, dim)
    Y = common.outcome(df, el, estimand)[:, None]
    lo = np.maximum(0.0, Y - (V - N))
    hi = np.minimum(Y, N)
    return lo, hi


def aggregate(df, el, dim, estimand, mask=None):
    """Statewide (or subset) rate bounds per group, plus the group's registered size."""
    lo, hi = saluran_bounds(df, el, dim, estimand)
    N = common.counts(df, dim)
    if mask is not None:
        lo, hi, N = lo[mask], hi[mask], N[mask]
    tot = N.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return lo.sum(axis=0) / tot, hi.sum(axis=0) / tot, tot


def run(el):
    df = common.load(el)
    OUT = common.outdir(el)
    rows, cells = [], []

    for dim, gs in DIMS.items():
        for estimand in ESTIMAND_NAMES:
            lo, hi = saluran_bounds(df, el, dim, estimand)
            N = common.counts(df, dim)
            alo, ahi, tot = aggregate(df, el, dim, estimand)

            for j, g in enumerate(gs):
                rows.append(dict(
                    dim=dim, group=g, estimand=estimand, registered=int(tot[j]),
                    lo=alo[j], hi=ahi[j], width=ahi[j] - alo[j],
                ))
                # A saluran only informs a group it actually contains.
                sel = N[:, j] > 0
                cells.append(pd.DataFrame(dict(
                    dim=dim, group=g, estimand=estimand,
                    cluster=df.cluster[sel], dm=df.dm[sel], saluran=df.saluran[sel],
                    n_group=N[sel, j], lo=lo[sel, j], hi=hi[sel, j],
                )))

    summary = pd.DataFrame(rows)
    percell = pd.concat(cells, ignore_index=True)
    percell["rate_lo"] = percell.lo / percell.n_group
    percell["rate_hi"] = percell.hi / percell.n_group
    summary.to_csv(f"{OUT}/bounds_statewide.csv", index=False)
    percell.to_parquet(f"{OUT}/bounds_saluran.parquet", index=False)

    # Per-seat bounds: one separate ecological problem per contested seat.
    seat_rows = []
    for cl, idx in df.groupby("cluster").indices.items():
        m = np.zeros(len(df), bool)
        m[idx] = True
        for dim, gs in DIMS.items():
            for estimand in ESTIMAND_NAMES:
                alo, ahi, tot = aggregate(df, el, dim, estimand, mask=m)
                for j, g in enumerate(gs):
                    if tot[j] > 0:
                        seat_rows.append(dict(cluster=cl, dim=dim, group=g, estimand=estimand,
                                              registered=int(tot[j]), lo=alo[j], hi=ahi[j]))
    pd.DataFrame(seat_rows).to_csv(f"{OUT}/bounds_seat.csv", index=False)

    print(f"\n=== {common.cfg(el)['label']} -- deterministic bound width (pp) ===")
    print("How much the arithmetic alone already settles. No model may leave these.")
    w = summary.pivot_table(index=["dim", "group"], columns="estimand", values="width") * 100
    print(w[ESTIMAND_NAMES].round(1).to_string())


if __name__ == "__main__":
    for e in common.arg_elections():
        run(e)
