"""Sharp, assumption-free bounds on the paper's headline CONTRASTS.
Usage: python3 contrasts.py

WHY. Table tab_bounds gives the Duncan-Davis interval for one cell at a time. But every
substantive claim is a linear contrast -- a change between two elections, BN minus PN, one
age band minus another -- and the sharp bound on a contrast can be tighter than the
difference of two cellwise bounds, because within a saluran the cells share margins: a Malay
BN ballot and a Malay PN ballot draw on the same Malay row and cannot both be at their
maximum. This script solves the transportation problem that enforces every constraint at
once, per saluran, and reports where arithmetic alone lands on each claim.

THE PROBLEM. For saluran i with row margins N (groups) and column margins Y (choices), the
feasible tables are  T >= 0,  T 1 = N,  1'T = Y.  For a statewide contrast with coefficient
matrix W (groups x choices, saluran-independent because denominators are statewide
electorates), the sharp bound is

    min / max  sum_i sum_gc W_gc T_igc      =  sum_i  ( min / max  sum_gc W_gc T_igc )

since salurans are independent tables. Each saluran is a small LP (4 x 8) solved with
HiGHS. Single-cell objectives reproduce the Frechet bounds exactly (a check the script
runs). A between-election change is the difference of two elections' interiors, which are
unlinked, so its sharp bound is [lo_to - hi_from, hi_to - lo_from].

TERRITORY. Every cross-election contrast is computed statewide AND on the 33 seats PN
contested at SE-16 (polling districts are identical across GE-15 and SE-16, so the
restriction is exact), since the stand-down makes fixed contested territory the
substantively cleaner comparison.

Outputs: output/contrast_bounds.csv and a printed claim ladder.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import common
from common import DIMS

OUT, REPO = common.OUT, common.REPO


def contested_seats():
    bal = pd.read_parquet(f"{REPO}/data/jhr_se16_ballots.parquet")
    bal = bal[~bal["dm"].str.contains("/00 |/UP ")]
    coal = bal.groupby("seat")["coalition"].apply(set)
    return sorted(s for s, c in coal.items() if "PN" in c)


def lp_bounds(df, el, dim, W):
    """Sharp [lo, hi] of sum_i sum_gc W_gc T_igc over feasible saluran tables."""
    CH = common.choices(el)
    N = common.counts(df, dim)
    Y = df[CH].to_numpy(float)
    G, C = W.shape
    # equality constraints: row sums (G) and column sums (C); one is redundant, drop last col
    A = np.zeros((G + C - 1, G * C))
    for g in range(G):
        A[g, g * C:(g + 1) * C] = 1
    for c in range(C - 1):
        A[G + c, c::C] = 1
    w = W.ravel()
    lo = hi = 0.0
    for i in range(len(df)):
        b = np.concatenate([N[i], Y[i, :-1]])
        if b.sum() == 0:
            continue
        r1 = linprog(w, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        r2 = linprog(-w, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        assert r1.status == 0 and r2.status == 0, f"LP infeasible at row {i}"
        lo += r1.fun
        hi += -r2.fun
    return lo, hi


def objective(el, dim, terms, N_tot):
    """W for sum_k coef_k * rate(group_k, estimand_k), rates as shares of the group's
    electorate over the territory (N_tot = group electorate totals in that territory)."""
    CH = common.choices(el)
    W = np.zeros((len(DIMS[dim]), len(CH)))
    for coef, g, e in terms:
        j = DIMS[dim].index(g)
        for c in common.estimands(el)[e]:
            W[j, CH.index(c)] += coef / N_tot[j]
    return W


def bounds_for(df, el, dim, terms):
    N_tot = common.counts(df, dim).sum(axis=0)
    return lp_bounds(df, el, dim, objective(el, dim, terms, N_tot))


def main():
    pn = contested_seats()
    data = {el: common.load(el) for el in ["ge15", "jhr_se16"]}
    terr = {"statewide": lambda d: d, "PN-contested 33 seats": lambda d: d[d.dun.isin(pn)]}

    # sanity: single-cell LP reproduces the closed-form Frechet bound
    d = data["jhr_se16"]
    lo, hi = bounds_for(d, "jhr_se16", "ethnicity", [(1, "Malay", "BN")])
    N = d.n_Malay.to_numpy(float); V = d.V.to_numpy(float); Y = d.BN.to_numpy(float)
    flo = np.maximum(0, Y - (V - N)).sum() / N.sum(); fhi = np.minimum(Y, N).sum() / N.sum()
    assert abs(lo - flo) < 1e-6 and abs(hi - fhi) < 1e-6, (lo, flo, hi, fhi)
    print(f"check: LP reproduces Frechet for Malay BN SE-16: [{100*lo:.2f}, {100*hi:.2f}]")

    claims = [
        ("Malay PN", "ethnicity", [(1, "Malay", "PN")]),
        ("Malay BN", "ethnicity", [(1, "Malay", "BN")]),
        ("Malay BN - PN", "ethnicity", [(1, "Malay", "BN"), (-1, "Malay", "PN")]),
        ("Malay turnout", "ethnicity", [(1, "Malay", "turnout")]),
        ("Chinese PH", "ethnicity", [(1, "Chinese", "PH")]),
        ("Chinese turnout", "ethnicity", [(1, "Chinese", "turnout")]),
        ("Chinese BN", "ethnicity", [(1, "Chinese", "BN")]),
    ]
    rows = []
    for name, dim, terms in claims:
        for tname, f in terr.items():
            lv = {el: bounds_for(f(data[el]), el, dim, terms) for el in data}
            rows.append(dict(claim=name, territory=tname, kind="level",
                             ge15_lo=lv["ge15"][0], ge15_hi=lv["ge15"][1],
                             se16_lo=lv["jhr_se16"][0], se16_hi=lv["jhr_se16"][1],
                             change_lo=lv["jhr_se16"][0] - lv["ge15"][1],
                             change_hi=lv["jhr_se16"][1] - lv["ge15"][0]))
            print(f"  {name:14s} {tname:22s} change in [{100*rows[-1]['change_lo']:6.1f}, "
                  f"{100*rows[-1]['change_hi']:6.1f}]")

    # within-election age contrasts at SE-16 (life-cycle profile)
    d16 = data["jhr_se16"]
    for a, b in [("60-69", "50-59"), ("60-69", "70+"), ("60-69", "30-39")]:
        lo, hi = bounds_for(d16, "jhr_se16", "age", [(1, a, "turnout"), (-1, b, "turnout")])
        rows.append(dict(claim=f"turnout {a} - {b}", territory="statewide", kind="within SE-16",
                         se16_lo=lo, se16_hi=hi))
        print(f"  turnout {a} - {b} at SE-16 in [{100*lo:.1f}, {100*hi:.1f}]")

    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/contrast_bounds.csv", index=False)
    print(f"\nwrote {OUT}/contrast_bounds.csv")


if __name__ == "__main__":
    main()
