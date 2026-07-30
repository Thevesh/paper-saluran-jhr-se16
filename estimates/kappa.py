"""The kappa frontier: exact bounds under a single bounded-heterogeneity dial.
Usage: python3 kappa.py

THE IDEA. Every estimate in this project is bounds + an assumption. The Duncan-Davis bounds
assume nothing about cross-stream behaviour; the models assume rates are constant everywhere.
This script makes the space between them the reported object. One parameter:

    kappa: group g's rate for outcome c varies across streams by at most +/- kappa
           around a common central rate m, on the logit scale.

kappa = 0 is Goodman's constancy assumption. kappa = infinity is Duncan-Davis. For any kappa
in between, the statewide bound is EXACT and has a closed form:

  - per stream, the group's feasible rate is Frechet [L_i/N_i, U_i/N_i] intersected with the
    band [expit(m-k) - s_i, expit(m+k) + s_i], where s_i = z*sqrt(0.25/N_i) is binomial slack
    (a small stream's realised rate can sit off its propensity; slack also auto-neutralises
    tiny streams);
  - the only cross-stream coupling is m, and the statewide sum is monotone in m, so the upper
    bound is attained at the largest feasible m and the lower at the smallest. No LP needed.

Feasibility itself is informative: the band must intersect every stream's Frechet interval,
so streams whose arithmetic pins their rate (the near-pure ones) force m to sit within kappa
of each of their observed rates. Their dispersion therefore LOWER-BOUNDS kappa: constancy is
not just implausible here, it is arithmetically infeasible, and kappa_min says by how much.

The near-pure streams then discipline the dial from above as well: their observed cross-stream
dispersion, after deconvolving binomial noise, measures within-group heterogeneity directly.
The single remaining assumption is one sentence: streams where behaviour is invisible are no
more heterogeneous than streams where it is visible.

Findings are reported as breakdown values: the smallest kappa at which the claim fails,
against the kappa actually measured.

Outputs: out/kappa_frontier.csv, out/kappa_breakdown.csv, and a printed report.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

OUT = common.OUT
Z_SLACK = 2.0          # binomial slack multiplier
EPS = 1e-9

# (label, group column, outcome builder). Turnout = ballots issued = V - NONVOTE.
TARGETS = [
    ("Malay BN", "n_Malay", lambda d: d.BN),
    ("Malay PN", "n_Malay", lambda d: d.PN),
    ("Malay turnout", "n_Malay", lambda d: d.V - d.NONVOTE),
    ("Chinese PH", "n_Chinese", lambda d: d.PH),
    ("Chinese BN", "n_Chinese", lambda d: d.BN),
    ("Chinese turnout", "n_Chinese", lambda d: d.V - d.NONVOTE),
]

KAPPAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]


def stream_arrays(df, gcol, ycol_fn):
    """Per-stream N, Frechet rate bounds, and slack, for streams with any group presence."""
    N = df[gcol].to_numpy(float)
    V = df.V.to_numpy(float)
    Y = ycol_fn(df).to_numpy(float)
    keep = N > 0
    N, V, Y = N[keep], V[keep], Y[keep]
    L = np.maximum(0.0, Y - (V - N))
    U = np.minimum(Y, N)
    r_lo, r_hi = L / N, U / N
    s = Z_SLACK * np.sqrt(0.25 / N)
    return N, r_lo, r_hi, s


def m_range(r_lo, r_hi, s, kappa):
    """Feasible range of the central logit m at a given kappa; (nan, nan) if infeasible.

    Stream i is feasible iff the band intersects its Frechet interval:
      expit(m - k) - s_i <= r_hi_i   and   expit(m + k) + s_i >= r_lo_i.
    """
    lo_c = r_lo - s
    hi_c = r_hi + s
    A = logit(np.clip(lo_c[lo_c > 0], EPS, 1 - EPS)).max() if (lo_c > 0).any() else -np.inf
    B = logit(np.clip(hi_c[hi_c < 1], EPS, 1 - EPS)).min() if (hi_c < 1).any() else np.inf
    m_min, m_max = A - kappa, B + kappa
    if m_min > m_max + 1e-9:
        return np.nan, np.nan
    return min(m_min, m_max), m_max


def kappa_min(r_lo, r_hi, s):
    """The smallest kappa at which any common central rate is feasible at all."""
    lo_c = r_lo - s
    hi_c = r_hi + s
    A = logit(np.clip(lo_c[lo_c > 0], EPS, 1 - EPS)).max() if (lo_c > 0).any() else -np.inf
    B = logit(np.clip(hi_c[hi_c < 1], EPS, 1 - EPS)).min() if (hi_c < 1).any() else np.inf
    return max(0.0, (A - B) / 2)


def bounds_at(N, r_lo, r_hi, s, kappa):
    """Exact statewide bounds at a given kappa. Frechet at kappa = inf."""
    if np.isinf(kappa):
        return (N * r_lo).sum() / N.sum(), (N * r_hi).sum() / N.sum()
    m_min, m_max = m_range(r_lo, r_hi, s, kappa)
    if np.isnan(m_min):
        return np.nan, np.nan
    band_hi = 1.0 if np.isinf(m_max) else expit(m_max + kappa)
    band_lo = 0.0 if np.isinf(m_min) else expit(m_min - kappa)
    hi = (N * np.minimum(r_hi, band_hi + s)).sum() / N.sum()
    lo = (N * np.maximum(r_lo, band_lo - s)).sum() / N.sum()
    return lo, hi


def measure_kappa(df, gcol, ycol_fn, thr=0.90):
    """Observed heterogeneity from near-pure streams: weighted logit-scale dispersion of
    their Frechet-midpoint rates, with the binomial component deconvolved."""
    pure = (df[gcol] / df.V) >= thr
    d = df[pure]
    if len(d) < 30:
        return dict(n=len(d), sd=np.nan, half95=np.nan)
    N, r_lo, r_hi, _ = stream_arrays(d, gcol, ycol_fn)
    r = np.clip((r_lo + r_hi) / 2, EPS, 1 - EPS)
    x = logit(r)
    w = N / N.sum()
    mu = (w * x).sum()
    var_tot = (w * (x - mu) ** 2).sum()
    p_bar = np.clip((w * r).sum(), EPS, 1 - EPS)
    var_bin = (w / (N * p_bar * (1 - p_bar))).sum()      # delta-method binomial noise
    sd = np.sqrt(max(0.0, var_tot - var_bin))
    # weighted central 95% half-spread, noise-deconvolved proportionally
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    q = np.interp([0.025, 0.975], cw, x[order])
    shrink = sd / np.sqrt(var_tot) if var_tot > 0 else 0.0
    return dict(n=len(d), sd=sd, half95=0.5 * (q[1] - q[0]) * shrink)


def frontier_table(df, label, gcol, ycol_fn):
    N, r_lo, r_hi, s = stream_arrays(df, gcol, ycol_fn)
    kmin = kappa_min(r_lo, r_hi, s)
    rows = []
    for k in KAPPAS:
        kk = k if np.isinf(k) else max(k, kmin + 1e-6)
        lo, hi = bounds_at(N, r_lo, r_hi, s, kk)
        rows.append(dict(target=label, kappa=k, kappa_used=kk, lo=100 * lo, hi=100 * hi,
                         width=100 * (hi - lo)))
    return kmin, pd.DataFrame(rows)


def swing_lower(bounds_by_el, kappa):
    """Worst-case (smallest) GE-15 -> SE-16 movement at a given kappa: lo16 - hi15."""
    lo16, _ = bounds_by_el["jhr_se16"](kappa)
    _, hi15 = bounds_by_el["ge15"](kappa)
    return 100 * (lo16 - hi15)


def breakdown(bounds_by_el, threshold, k_entry, k_hi=6.0):
    """Smallest kappa at which the worst-case swing drops below the threshold.

    The search starts at k_entry, the smallest kappa at which BOTH elections' constraints
    are jointly feasible; below that no world exists to test the claim in. Returns
    (kappa_star, status): status 'fails_at_entry' if the claim is already dead at k_entry,
    'never' if it holds all the way out to (near) the Frechet limit.
    """
    if swing_lower(bounds_by_el, k_entry) < threshold:
        return k_entry, "fails_at_entry"
    if swing_lower(bounds_by_el, k_hi) >= threshold:
        return np.inf, "never"
    lo_k, hi_k = k_entry, k_hi
    for _ in range(50):
        mid = (lo_k + hi_k) / 2
        if swing_lower(bounds_by_el, mid) >= threshold:
            lo_k = mid
        else:
            hi_k = mid
    return (lo_k + hi_k) / 2, "breaks"


def main():
    data = {el: common.load(el) for el in ["ge15", "jhr_se16"]}
    # PN at SE-16 only exists where it stood: restrict PN targets to contested duns.
    pn_duns = data["jhr_se16"].groupby("dun").PN.sum()
    contested = set(pn_duns[pn_duns > 0].index)
    pn16 = data["jhr_se16"][data["jhr_se16"].dun.isin(contested)]

    frontiers, meas_rows = [], []
    print("=" * 100)
    for el in ["ge15", "jhr_se16"]:
        df = data[el]
        for label, gcol, fn in TARGETS:
            d = pn16 if (el == "jhr_se16" and "PN" in label) else df
            kmin, tab = frontier_table(d, f"{common.cfg(el)['short']} {label}", gcol, fn)
            m = measure_kappa(d, gcol, fn)
            frontiers.append(tab)
            meas_rows.append(dict(election=common.cfg(el)["short"], target=label,
                                  kappa_min=kmin, n_pure=m["n"], kappa_sd=m["sd"],
                                  kappa_half95=m["half95"]))
            print(f"\n--- {common.cfg(el)['short']} {label} ---")
            print(f"  kappa_min (forced by arithmetic): {kmin:.3f}")
            print(f"  measured on >=90% pure streams (n={m['n']}): "
                  f"sd = {m['sd']:.3f}, 95% half-spread = {m['half95']:.3f}"
                  if not np.isnan(m["sd"]) else "  no usable pure stratum")
            print(tab[["kappa", "lo", "hi", "width"]].to_string(index=False,
                  float_format=lambda v: f"{v:8.2f}"))

    pd.concat(frontiers).to_csv(f"{OUT}/kappa_frontier.csv", index=False)
    pd.DataFrame(meas_rows).to_csv(f"{OUT}/kappa_measured.csv", index=False)

    # ---- breakdown values for the headline claims -------------------------------------
    print("\n" + "=" * 100)
    print("BREAKDOWN VALUES (smallest kappa at which the claim fails)")

    def make_fn(gcol, fn, el, restrict_pn=False):
        d = pn16 if restrict_pn else data[el]
        N, r_lo, r_hi, s = stream_arrays(d, gcol, fn)
        return lambda k: bounds_at(N, r_lo, r_hi, s, k)

    claims = [
        ("Malay BN swing >= 20pp", "n_Malay", lambda d: d.BN, 20.0, False),
        ("Malay BN swing >= 10pp", "n_Malay", lambda d: d.BN, 10.0, False),
        ("Malay PN fall >= 20pp", "n_Malay", lambda d: d.PN, 20.0, True),
        ("Malay PN fall >= 10pp", "n_Malay", lambda d: d.PN, 10.0, True),
    ]
    def entry_kappa(gcol, fn, restrict_pn):
        ks = []
        for el in ["ge15", "jhr_se16"]:
            d = pn16 if (restrict_pn and el == "jhr_se16") else data[el]
            _, r_lo, r_hi, s = stream_arrays(d, gcol, fn)
            ks.append(kappa_min(r_lo, r_hi, s))
        return max(ks) + 1e-6

    brows = []
    for name, gcol, fn, thr, is_pn in claims:
        if "PN" in name:
            # the fall: hi16 below lo15 by at least thr -> worst case lo15 - hi16
            b15 = make_fn(gcol, fn, "ge15")
            b16 = make_fn(gcol, fn, "jhr_se16", restrict_pn=True)
            by = {"ge15": b16, "jhr_se16": b15}   # reuse swing_lower via swap
        else:
            by = {"ge15": make_fn(gcol, fn, "ge15"),
                  "jhr_se16": make_fn(gcol, fn, "jhr_se16")}
        k_entry = entry_kappa(gcol, fn, is_pn)
        k_star, status = breakdown(by, thr, k_entry)
        worst_at_entry = swing_lower(by, k_entry)
        brows.append(dict(claim=name, kappa_entry=k_entry, worst_case_at_entry=worst_at_entry,
                          kappa_star=k_star, status=status))
        print(f"  {name:28s} entry kappa = {k_entry:.3f}, worst-case movement there = "
              f"{worst_at_entry:+.1f}pp -> {status}"
              + (f" at kappa* = {k_star:.3f}" if status == "breaks" else ""))
    pd.DataFrame(brows).to_csv(f"{OUT}/kappa_breakdown.csv", index=False)

    print(f"\nwrote {OUT}/kappa_frontier.csv, kappa_measured.csv, kappa_breakdown.csv")


if __name__ == "__main__":
    main()
