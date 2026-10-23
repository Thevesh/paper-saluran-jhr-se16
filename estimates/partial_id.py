"""
Slope-bounded partial identification: an interval for every group, including those no
near-pure stratum can validate. Usage: python3 partial_id.py      (prototype; not a stage)

THE MODEL (after Jiang, King, Schmaltz & Tanner 2020, "Ecological regression with partial
identification"). In saluran i, a group's realised rate for an outcome is

    b_i = m + c * (x_i - xbar) + u_i

  m      the group's rate over the territory -- the estimand
  x_i    the group's share of saluran i; xbar its mean over the group's own members
  c      the CONTEXTUAL SLOPE: how the group's rate moves with its own concentration
  u_i    saluran-specific deviation, averaging to zero over the group's members

The data say only that each b_i lies in saluran i's Frechet interval [L_i/N_i, U_i/N_i]. The
assumptions are |c| <= BETA and |u_i| <= RHO + s_i (s_i: finite-count slack for a realised
rate). The interval for m is the min/max of a linear programme. Because sum_i N_i u_i = 0 and
sum_i N_i (x_i - xbar) = 0, the group's realised rate over the territory equals m exactly.

WHERE BETA AND RHO COME FROM -- nothing is chosen by hand:
  BETA  measured contextual slopes of the VALIDATED groups (Malay, Chinese, age bands), read
        off their own held-out tests: (what the near-pure streams did - what the mixed
        fit predicted) / (how much more concentrated the near-pure streams are). The
        benchmark is a high quantile of those measured slopes.
  RHO   the dispersion of Malay rates across near-pure Malay streams, where composition is
        ~constant and the contextual term therefore drops out.
So an Indian or Other interval reads: what the data allow if these voters' contextual
effects and local variation are no larger than the validated groups' demonstrably are.

If the assumptions cannot hold at all for a cell (the LP is infeasible), both dials are
relaxed by the smallest factor that restores feasibility, and the factor is reported.

Fitted INSIDE each seat (own rate, own bounded slope) and summed: the pooled form, one rate
for the whole state, forced real seat-to-seat differences into the noise term and excluded
validated estimates (Malay turnout SE-16 [63.7, 67.1] against a validated 76.2).
Writes output/partial_id_slopes.csv and output/partial_id.csv.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

import common
import errors as err
import holdout as hold
import joint as jm

OUT = common.OUT
EST = ["turnout", "BN", "PH", "PN"]
Z = 2.0                                   # finite-count slack multiplier, as in kappa.py
RELAX = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def outcome(df, est):
    if est == "turnout":
        return df.V - df.NONVOTE
    if est == "valid":                       # valid votes cast: the vote-share denominator
        return df[[c for c in ["BN", "PH", "PN", "BERSAMA", "OTHER"] if c in df]].sum(axis=1)
    return df[est]


# ---------------------------------------------------------------- the benchmarks
def measure_slopes():
    """Contextual slope implied by every validated held-out test, at its own tier."""
    v = pd.read_csv(f"{OUT}/validation_errors.csv")
    tiers = v.drop_duplicates(["election", "dim", "group"])[["election", "dim", "group", "thr"]]
    rows = []
    for el in common.ORDER:
        df = jm.load_joint(el)
        V = df.V.to_numpy(float)
        for (dim, thr), sub in tiers[tiers.election == el].groupby(["dim", "thr"]):
            pure, mixed, arg = hold.design_masks(df, dim, "A", thr=thr)
            tabs = err.stream_table(df, el, dim, list(sub.group), thr=thr)
            Nall = common.counts(df, dim)
            for g, t in tabs.items():
                j = common.DIMS[dim].index(g)
                m = pure & (arg == j)
                x = Nall[:, j] / V
                # member-weighted concentration: the near-pure streams vs the mixed streams of
                # the same seats, which is what the own-seat prediction was fitted on
                seats = set(df.cluster[m])
                mm = mixed & df.cluster.isin(seats).to_numpy()
                dx = (np.average(x[m], weights=Nall[m, j])
                      - np.average(x[mm], weights=Nall[mm, j]))
                N, Vs, wt = t.N.to_numpy(), t.V.to_numpy(), t.wt.to_numpy()
                for est in EST:
                    Y = t[f"Y_{est}"].to_numpy()
                    lo = np.maximum(0, Y - (Vs - N)).sum() / N.sum()
                    hi = np.minimum(Y, N).sum() / N.sum()
                    pred = (wt * t[f"p_{est}"].to_numpy()).sum() / wt.sum()
                    rows.append(dict(election=el, dim=dim, group=g, estimand=est, thr=thr,
                                     dx=dx, pred=pred, lo=lo, hi=hi,
                                     slope_mid=((lo + hi) / 2 - pred) / dx,
                                     slope_min=max(lo - pred, pred - hi, 0.0) / dx))
    return pd.DataFrame(rows)


def measure_rho(thr=0.90):
    """95% half-spread of Malay rates across near-pure Malay streams (probability scale),
    binomial noise deconvolved -- local variation at ~constant composition."""
    out = []
    for el in common.ORDER:
        df = common.load(el)
        for est in EST:
            d = df
            if est in common.PARTIES:
                d = df[df.groupby("cluster")[est].transform("sum") > 0]
            d = d[d.n_Malay / d.V >= thr]
            N, V, Y = d.n_Malay.to_numpy(float), d.V.to_numpy(float), outcome(d, est).to_numpy(float)
            r = (np.maximum(0, Y - (V - N)) + np.minimum(Y, N)) / (2 * N)
            w = N / N.sum()
            mu = (w * r).sum()
            var = (w * (r - mu) ** 2).sum()
            var_bin = (w * mu * (1 - mu) / N).sum()
            sd = np.sqrt(max(0.0, var - var_bin))
            out.append(dict(election=el, estimand=est, rho=1.96 * sd, n=len(d)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- the interval
def interval(d, gcol, est, beta, rho, N_total=None):
    """[lo, hi] for the group's rate over the salurans in d, or (nan, nan) if infeasible.
    Party outcomes are evaluated over the seats the party contested and scaled to the whole
    group (zero elsewhere by construction)."""
    N_total = d[gcol].sum() if N_total is None else N_total
    if est in common.PARTIES:
        d = d[d.groupby("cluster")[est].transform("sum") > 0]
    d = d[d[gcol] > 0]
    if d.empty or N_total == 0:
        return 0.0, 0.0
    share = d[gcol].sum() / N_total
    N, V, Y = d[gcol].to_numpy(float), d.V.to_numpy(float), outcome(d, est).to_numpy(float)
    rlo = np.maximum(0, Y - (V - N)) / N
    rhi = np.minimum(Y, N) / N
    s = Z * np.sqrt(0.25 / N)
    x = N / V
    xc = x - np.average(x, weights=N)
    n = len(N)
    # variables: [m, c, u_1..u_n]
    I = np.arange(n)
    rows = np.r_[I, I, I]
    cols = np.r_[np.zeros(n, int), np.ones(n, int), I + 2]
    A = coo_matrix((np.r_[np.ones(n), xc, np.ones(n)], (rows, cols)), shape=(n, n + 2))
    A_ub = vstack([A, -A]).tocsr()
    b_ub = np.r_[rhi, -rlo]                       # m + c xc_i + u_i in [rlo_i, rhi_i]
    A_eq = coo_matrix((N, (np.zeros(n, int), I + 2)), shape=(1, n + 2)).tocsr()
    ub_u = rho + s
    bounds = [(0, 1), (-beta, beta)] + list(zip(-ub_u, ub_u))
    out = []
    for sign in (1, -1):
        cvec = np.zeros(n + 2)
        cvec[0] = sign
        r = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[0.0], bounds=bounds,
                    method="highs")
        if r.status != 0:
            return np.nan, np.nan
        out.append(r.x[0])
    return out[0] * share, out[1] * share


def relaxed(d, gcol, est, beta, rho, N_total=None):
    for f in RELAX:
        lo, hi = interval(d, gcol, est, f * beta, f * rho, N_total)
        if np.isfinite(lo):
            return lo, hi, f
    return np.nan, np.nan, np.nan


def by_seat(df, gcol, est, beta, rho, seat="cluster"):
    """The model inside each seat (own rate, own bounded slope), seats summed: unrestricted
    seat-to-seat differences, which the pooled form wrongly forces into the noise term.
    Returns statewide (lo, hi, max relaxation factor used) and the per-seat frame."""
    N_total = df[gcol].sum()
    lo = hi = 0.0
    fmax, seats = 1.0, []
    for s_, d in df.groupby(seat):
        l, h, f = relaxed(d, gcol, est, beta, rho, N_total)
        if not np.isfinite(l):                      # never feasible: fall back to Frechet
            l, h = frechet(d, gcol, est, N_total)
            f = np.inf
        lo, hi, fmax = lo + l, hi + h, max(fmax, f)
        Ns = d[gcol].sum()
        seats.append(dict(seat=s_, lo=l * N_total / Ns if Ns else np.nan,
                          hi=h * N_total / Ns if Ns else np.nan, relax=f, registered=Ns))
    return lo, hi, fmax, pd.DataFrame(seats)


# ---------------------------------------------------------------- vote shares, directly
VALIDCOLS = ["BN", "PH", "PN", "BERSAMA", "OTHER"]


def share_lp(d, gcol, party, beta, rho, seat="cluster"):
    """[lo, hi] of the group's share of ITS OWN valid votes going to `party`, over the salurans
    in d, under the same seat-by-seat bounded-slope model -- solved directly, not by dividing
    two separately computed ranges (which stacks their worst cases).

    Per saluran i the unknowns are the group's party votes y_i and its valid votes v_i. Both
    follow the model on the registered scale, seat by seat:
        y_i = N_i (mp_s + cp_s xc_i + up_i),   v_i = N_i (mv_s + cv_s xc_i + uv_i),
        |cp|, |cv| <= beta,  |up_i|, |uv_i| <= rho + s_i,  sum_{i in s} N_i up_i = 0 (same for uv)
    and the saluran's table must add up: y_i, v_i in their Frechet intervals, y_i <= v_i, and the
    other voters' party votes cannot exceed their valid votes (P_i - y_i <= valid_i - v_i).
    The objective sum y / sum v is linear-fractional; the Charnes-Cooper transformation makes
    it an ordinary LP. Returns (nan, nan) if the assumptions cannot hold."""
    d = d[d[gcol] > 0]
    if d.empty or d[party].sum() == 0:
        return np.nan, np.nan
    N = d[gcol].to_numpy(float)
    V = d.V.to_numpy(float)
    P_ = d[party].to_numpy(float)
    W = d[[c for c in VALIDCOLS if c in d]].sum(axis=1).to_numpy(float)
    sid = pd.factorize(d[seat])[0]
    S = sid.max() + 1
    n = len(N)
    x = N / V
    xc = x.copy()
    for k in range(S):
        m = sid == k
        xc[m] = x[m] - np.average(x[m], weights=N[m])
    slack = Z * np.sqrt(0.25 / N)
    # variable layout
    iy, iv, iup, iuv = 0, n, 2 * n, 3 * n
    imp, icp, imv, icv = 4 * n, 4 * n + S, 4 * n + 2 * S, 4 * n + 3 * S
    nv = 4 * n + 4 * S
    lo = np.r_[np.maximum(0, P_ - (V - N)), np.maximum(0, W - (V - N)),
               -(rho + slack), -(rho + slack), np.zeros(S), -beta * np.ones(S),
               np.zeros(S), -beta * np.ones(S)]
    hi = np.r_[np.minimum(P_, N), np.minimum(W, N), rho + slack, rho + slack,
               np.ones(S), beta * np.ones(S), np.ones(S), beta * np.ones(S)]
    I = np.arange(n)
    r, c, v = [], [], []                       # equality rows
    def put(row, col, val):
        r.append(row); c.append(col); v.append(val)
    for i in I:                                # y_i - N_i(mp + cp xc_i + up_i) = 0
        put(i, iy + i, 1.0); put(i, imp + sid[i], -N[i]); put(i, icp + sid[i], -N[i] * xc[i])
        put(i, iup + i, -N[i])
        j = n + i                              # v_i - N_i(mv + cv xc_i + uv_i) = 0
        put(j, iv + i, 1.0); put(j, imv + sid[i], -N[i]); put(j, icv + sid[i], -N[i] * xc[i])
        put(j, iuv + i, -N[i])
    for i in I:                                # per-seat mean-zero deviations
        put(2 * n + sid[i], iup + i, N[i])
        put(2 * n + S + sid[i], iuv + i, N[i])
    A_eq = coo_matrix((v, (r, c)), shape=(2 * n + 2 * S, nv)).tocsr()
    b_eq = np.zeros(2 * n + 2 * S)
    # inequalities: y - v <= 0 ; -y + v <= W - P
    A1 = coo_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[I, I], np.r_[iy + I, iv + I])),
                    shape=(n, nv))
    A2 = coo_matrix((np.r_[-np.ones(n), np.ones(n)], (np.r_[I, I], np.r_[iy + I, iv + I])),
                    shape=(n, nv))
    A_ub = vstack([A1, A2]).tocsr()
    b_ub = np.r_[np.zeros(n), W - P_]
    # Charnes-Cooper: w = tau * x, tau >= 0; constraints homogenised; sum v (scaled) = 1
    from scipy.sparse import eye, hstack, csr_matrix
    tcol = lambda b: csr_matrix(-np.asarray(b, float).reshape(-1, 1))
    Aub = vstack([hstack([A_ub, tcol(b_ub)]),
                  hstack([eye(nv), tcol(hi)]),                 #  w - hi tau <= 0
                  hstack([-eye(nv), csr_matrix(np.asarray(lo, float).reshape(-1, 1))])])  # lo tau - w <= 0
    Aeq = vstack([hstack([A_eq, tcol(b_eq)]),
                  csr_matrix((np.ones(n), (np.zeros(n, int), iv + I)), shape=(1, nv + 1))])
    beq = np.r_[np.zeros(A_eq.shape[0]), 1.0]
    obj = np.zeros(nv + 1)
    obj[iy + I] = 1.0
    bounds = [(None, None)] * nv + [(0, None)]
    out = []
    for sign in (1, -1):
        res = linprog(sign * obj, A_ub=Aub, b_ub=np.zeros(Aub.shape[0]), A_eq=Aeq, b_eq=beq,
                      bounds=bounds, method="highs")
        if res.status != 0:
            return np.nan, np.nan
        out.append(sign * res.fun)
    return out[0], out[1]


def share_relaxed(d, gcol, party, beta, rho, seat="cluster"):
    """share_lp at the smallest relaxation of (beta, rho) that is feasible; Frechet otherwise."""
    for f in RELAX:
        lo, hi = share_lp(d, gcol, party, f * beta, f * rho, seat)
        if np.isfinite(lo):
            return lo, hi, f
    # assumption-free: the same LP with the model switched off
    lo, hi = share_lp(d, gcol, party, 1e6, 1e6, seat)
    return lo, hi, np.inf


def frechet(d, gcol, est, N_total=None):
    N_total = d[gcol].sum() if N_total is None else N_total
    N, V, Y = d[gcol].to_numpy(float), d.V.to_numpy(float), outcome(d, est).to_numpy(float)
    return (np.maximum(0, Y - (V - N)).sum() / N_total, np.minimum(Y, N).sum() / N_total)


def main():
    S = measure_slopes()
    S.to_csv(f"{OUT}/partial_id_slopes.csv", index=False)
    a = S.slope_mid.abs()
    BETA = float(a.quantile(0.9))
    print("Measured contextual slopes (|rate change| per unit of own-group share), validated "
          f"groups, {len(S)} cells:")
    print(f"  median {a.median():.3f}   90th pct {BETA:.3f}   max {a.max():.3f}")
    print(S.assign(abs_slope=a).groupby("group").abs_slope.median().round(3).to_string())
    R0 = measure_rho()
    RHO = float(R0.rho.median())
    print(f"\nMalay local variation (95% half-spread, near-pure streams): median rho = {RHO:.3f}")
    print(f"\n-> BETA = {BETA:.3f} (90th pct of measured slopes), RHO = {RHO:.3f}\n")

    C = pd.read_csv(f"{OUT}/joint_cells.csv").fillna(0)
    rows = []
    for el in common.ORDER:
        df = common.load(el)
        for g in common.ETH + common.AGE:
            key = "eth" if g in common.ETH else "age"
            cell = C[(C.election == el) & (C[key] == g)].sum(numeric_only=True)
            for est in EST:
                p = 1 - cell.NONVOTE / cell.registered if est == "turnout" else cell[est] / cell.registered
                lo, hi, f, _ = by_seat(df, f"n_{g}", est, BETA, RHO)
                flo, fhi = frechet(df, f"n_{g}", est)
                rows.append(dict(election=el, group=g, estimand=est, point=100 * p,
                                 lo=100 * lo, hi=100 * hi, width=100 * (hi - lo), relax=f,
                                 bounds_lo=100 * flo, bounds_hi=100 * fhi,
                                 bounds_width=100 * (fhi - flo),
                                 point_inside=bool(lo - 1e-9 <= p <= hi + 1e-9)))
        print(f"  {el}: done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/partial_id.csv", index=False)

    pd.set_option("display.width", 220)
    order = common.ETH + common.AGE
    W = R.groupby("group")[["width", "bounds_width"]].mean().reindex(order)
    W["% of bounds"] = 100 * W.width / W.bounds_width
    W["relaxed cells"] = R.groupby("group").relax.apply(lambda f: int((f > 1).sum())).reindex(order)
    print("\nMean interval width (pp, registered scale) by group")
    print(W.round(1).to_string())
    print(f"\npoint estimate inside its interval: {int(R.point_inside.sum())} of {len(R)}")
    print("\nSE-16 detail:")
    print(R[R.election == common.ORDER[-1]][["group", "estimand", "point", "lo", "hi", "relax",
                                             "bounds_lo", "bounds_hi"]].round(1).to_string(index=False))
    print(f"\nwrote {OUT}/partial_id_slopes.csv, {OUT}/partial_id.csv")


if __name__ == "__main__":
    main()
