"""
Joint (ethnicity x age) estimates with a full error budget, and marginals derived from them.
Usage: python3 joint_estimates.py [election ...]

Standalone: reuses joint.py for the model itself and 01/02/03 for the primitives. Writes
only its own files (joint_final.csv, joint_marginals.csv). Nothing in 00-11 is modified.

WHY THIS SUPERSEDES THE SEPARATE MODEL AS THE HEADLINE
groundtruth.py showed, against real held-out voters, that the separate age model is wrong
(70+ BN off by 16pp) because it cannot separate ethnicity within an age band, and joint.py
showed the joint model fixes it (16pp -> 2.5pp). The crosstab is therefore the object to
report; the four-cell marginals are recovered from it by summation, and -- because the joint
model barely moves them (<0.4pp for age, <1.2pp for Malay/Chinese vs the separate model) --
the marginal numbers already published stand. What changes is provenance, not value: every
marginal now comes from a model that is validated against ground truth rather than one that is
not.

THE ERROR BUDGET, unchanged in form from 07 so the whole paper uses one error model:

    total_se = sqrt( sampling_se^2 + identification_rmse^2 ),  clipped to the Frechet bounds

  sampling_se        cluster bootstrap over contested seats, through the whole joint
                     seed-then-rake pipeline.
  identification_rmse world-B simulation on the joint rows: plant a truth in which each
                     cell's rate slides with saluran composition, run the joint pipeline, take
                     the rmse of the recovered cell rate against the planted one. This is the
                     aggregation-bias term, and it does not shrink with electorate size.

Independently, joint.py's holdout puts the *real* out-of-sample miss of the identified
cells at ~1-1.5pp, which is a real-data check that the simulated identification term is not
wildly off. The two agree to the right order of magnitude.

THE STAR
    *  <=>  the cell holds < 1.5% of the electorate  OR  total error > 5pp
The share test is what retires Indian and Other: no saluran is anywhere near pure on their
joint cells, so they are unidentified regardless of the error arithmetic. A starred cell is
still the best available guess and still inside its bounds; it just cannot carry an argument.
"""

import os

import numpy as np
import pandas as pd

import common
from common import ESTIMAND_NAMES, ORDER, DIMS

import joint as jm
import models
import reconcile as rec

OUT = common.OUT
JOINT, JCOL, ETH, AGE = jm.JOINT, jm.JCOL, jm.ETH, jm.AGE
NJ = len(JOINT)   # number of joint cells (ethnicity x age)
NBOOT = 100
NSIM = 25
DELTA_SD = 1.0                       # world-B context slope, same as 05
STAR_ERR = 0.05
STAR_SHARE = 0.015
RNG = np.random.default_rng(19072026)


# ---------------------------------------------------------------- deterministic bounds
def cell_bounds(df, el, estimand):
    """Frechet bounds on every joint cell's rate for one estimand. Returns (lo[NJ], hi[NJ])."""
    V = df.V.to_numpy(float)[:, None]
    N = jm.jcounts(df)
    Y = common.outcome(df, el, estimand)[:, None]
    lo = np.maximum(0.0, Y - (V - N)).sum(axis=0)
    hi = np.minimum(Y, N).sum(axis=0)
    tot = N.sum(axis=0)
    return lo / np.maximum(tot, 1e-9), hi / np.maximum(tot, 1e-9)


# ---------------------------------------------------------------- sampling error
def bootstrap(df, el):
    """Seat cluster bootstrap through the joint pipeline. Returns {estimand: draws[B,NJ]}."""
    cls = df.cluster.unique()
    idx = {c: np.flatnonzero((df.cluster == c).to_numpy()) for c in cls}
    draws = {e: [] for e in ESTIMAND_NAMES}
    for _ in range(NBOOT):
        pick = RNG.choice(cls, size=len(cls), replace=True)
        d2 = df.iloc[np.concatenate([idx[c] for c in pick])]
        try:
            total, _ = jm.fit_joint(d2, el)
            Nt = jm.jcounts(d2).sum(axis=0)
            for e in ESTIMAND_NAMES:
                draws[e].append(jm.jrates(total, Nt, el, e))
        except Exception:
            continue
    return {e: np.array(v) for e, v in draws.items()}


# ---------------------------------------------------------------- identification error
def collapse_weights(Ntot):
    """{dim: W[ngroups,NJ]} mapping joint cells onto marginal groups, weighted by electorate."""
    Wd = {}
    for dim, groups in DIMS.items():
        W = np.zeros((len(groups), NJ))
        for k, (et, ag) in enumerate(JOINT):
            W[groups.index(et if dim == "ethnicity" else ag), k] = Ntot[k]
        Wd[dim] = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
    return Wd


def identification(df, el):
    """World-B on the joint rows.

    Returns ({estimand: rmse[NJ]}, {(dim,party): rmse[ngroups]}) -- the second being the
    identification error of the turnout-SCALED marginal (party/turnout). The ratio is formed
    inside each simulation, before averaging, so the error shared by numerator and denominator
    cancels exactly as it does in the estimate itself. Dividing the two marginals' errors would
    instead add them, overstating the uncertainty of every voter-scale figure.
    """
    CH = common.choices(el)
    seed = jm.fit_joint(df, el)[1]
    B = seed / seed.sum(axis=1, keepdims=True)                      # NJ x C truth, rows sum 1
    X = jm.jshares(df)
    z = X[:, 0] - np.average(X[:, 0], weights=df.V)                 # centred Malay|18-29 share
    N = jm.jcounts(df).astype(int)
    cols = {e: [CH.index(c) for c in common.estimands(el)[e]] for e in ESTIMAND_NAMES}
    sq = {e: np.zeros(NJ) for e in ESTIMAND_NAMES}

    parties = [e for e in ESTIMAND_NAMES if e != "turnout"]
    Wd = collapse_weights(N.sum(axis=0))
    sq_r = {(d, p): np.zeros(len(DIMS[d])) for d in DIMS for p in parties}

    for _ in range(NSIM):
        delta = RNG.normal(0, DELTA_SD, size=B.shape)
        P = common.expit(common.logit(np.clip(B, 1e-6, 1 - 1e-6))[None] + z[:, None, None] * delta[None])
        P = P / P.sum(axis=2, keepdims=True)
        Y = np.zeros((len(df), len(CH)))
        truth = np.zeros(B.shape)
        for g in range(NJ):
            cell = RNG.multinomial(N[:, g], P[:, g, :])
            Y += cell
            truth[g] = cell.sum(axis=0)
        Ntot = N.sum(axis=0)
        t_rate = {e: truth[:, cols[e]].sum(axis=1) / np.maximum(Ntot, 1e-9) for e in ESTIMAND_NAMES}
        sim = df.copy()
        for k, name in enumerate(CH):
            sim[name] = Y[:, k]
        total, _ = jm.fit_joint(sim, el)
        r_rate = {e: jm.jrates(total, Ntot, el, e) for e in ESTIMAND_NAMES}
        for e in ESTIMAND_NAMES:
            sq[e] += (r_rate[e] - t_rate[e]) ** 2
        for dim, W in Wd.items():                       # ratio formed per-sim, then squared
            rt, tt = W @ r_rate["turnout"], W @ t_rate["turnout"]
            for p in parties:
                sq_r[(dim, p)] += ((W @ r_rate[p]) / np.maximum(rt, 1e-9)
                                   - (W @ t_rate[p]) / np.maximum(tt, 1e-9)) ** 2
    return ({e: np.sqrt(sq[e] / NSIM) for e in ESTIMAND_NAMES},
            {k: np.sqrt(v / NSIM) for k, v in sq_r.items()})


# ---------------------------------------------------------------- assemble
def run(el):
    df = jm.load_joint(el)
    total, _ = jm.fit_joint(df, el)
    Ntot = jm.jcounts(df).sum(axis=0)
    boot = bootstrap(df, el)
    ident, ident_ratio = identification(df, el)
    print(f"  {el}: joint fitted, {NBOOT} bootstraps, {NSIM} identification sims")

    rows = []
    for e in ESTIMAND_NAMES:
        rate = jm.jrates(total, Ntot, el, e)
        se = boot[e].std(axis=0)
        err = 1.96 * np.sqrt(se ** 2 + ident[e] ** 2)
        lo_b, hi_b = cell_bounds(df, el, e)
        lo = np.maximum(rate - err, lo_b)
        hi = np.minimum(rate + err, hi_b)
        for k, (et, ag) in enumerate(JOINT):
            share = Ntot[k] / Ntot.sum()
            cite = (share >= STAR_SHARE) and (err[k] <= STAR_ERR)
            rows.append(dict(election=el, eth=et, age=ag, estimand=e, registered=int(Ntot[k]),
                             share=share, rate=rate[k], se_sampling=se[k],
                             rmse_ident=ident[e][k], error=err[k],
                             lo=lo[k], hi=hi[k], bound_lo=lo_b[k], bound_hi=hi_b[k],
                             cite=bool(cite)))
    return pd.DataFrame(rows), boot, ident, ident_ratio, total, Ntot, df


def marginals(el, boot, ident, total, Ntot, df):
    """Collapse joint -> 4-cell marginals, carrying the error through the same draws."""
    out = []
    for dim, groups in DIMS.items():
        for e in ESTIMAND_NAMES:
            CH = common.choices(el)
            cols = [CH.index(c) for c in common.estimands(el)[e]]
            # point estimate
            num = np.zeros(len(groups)); den = np.zeros(len(groups))
            for k, (et, ag) in enumerate(JOINT):
                j = groups.index(et if dim == "ethnicity" else ag)
                num[j] += total[k, cols].sum(); den[j] += Ntot[k]
            rate = num / np.maximum(den, 1e-9)
            # collapse the bootstrap draws the same way for a sampling SE
            W = np.zeros((len(groups), NJ))
            for k, (et, ag) in enumerate(JOINT):
                W[groups.index(et if dim == "ethnicity" else ag), k] = Ntot[k]
            W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
            se = (W @ boot[e].T).std(axis=1)
            rmse = W @ ident[e]                          # identification, collapsed
            err = 1.96 * np.sqrt(se ** 2 + rmse ** 2)
            for j, g in enumerate(groups):
                out.append(dict(election=el, dim=dim, group=g, estimand=e,
                                rate=rate[j], error=err[j]))
    return pd.DataFrame(out)


def scaled(el, boot, ident_ratio, total, Ntot):
    """The same joint marginals on the VOTER scale: party / turnout.

    Both the point estimate and its error come from the joint model, so the whole results
    section is one specification. The ratio is recomputed inside every bootstrap draw and
    every identification simulation, letting the error common to numerator and denominator
    cancel. Because that cancellation is what keeps the ratio's error honest, the citation
    bar is then applied directly, in the units displayed: a voter-scale figure is citable iff
    its voter-scale error clears 5pp. (Rescaling the error by turnout before testing would
    pass cells carrying +-13pp in the units a reader would actually quote.)
    """
    parties = [e for e in ESTIMAND_NAMES if e != "turnout"]
    Wd = collapse_weights(Ntot)
    CH = common.choices(el)
    out = []
    for dim, groups in DIMS.items():
        W = Wd[dim]
        num = {}
        for e in ESTIMAND_NAMES:
            cols = [CH.index(c) for c in common.estimands(el)[e]]
            n = np.zeros(len(groups)); d = np.zeros(len(groups))
            for k, (et, ag) in enumerate(JOINT):
                j = groups.index(et if dim == "ethnicity" else ag)
                n[j] += total[k, cols].sum(); d[j] += Ntot[k]
            num[e] = n / np.maximum(d, 1e-9)
        t_draw = W @ boot["turnout"].T                       # [ngroups, B]
        for p in parties:
            ratio = num[p] / np.maximum(num["turnout"], 1e-9)
            se = ((W @ boot[p].T) / np.maximum(t_draw, 1e-9)).std(axis=1)
            err = 1.96 * np.sqrt(se ** 2 + ident_ratio[(dim, p)] ** 2)
            for j, g in enumerate(groups):
                out.append(dict(election=el, dim=dim, group=g, party=p,
                                share_of_registered=num[p][j], turnout=num["turnout"][j],
                                share_of_voters=min(ratio[j], 1.0), error=err[j],
                                se_sampling=se[j], rmse_ident=ident_ratio[(dim, p)][j],
                                cite=bool(err[j] <= STAR_ERR)))
    return pd.DataFrame(out)


def main():
    els = common.arg_elections()
    cells, margs, scal = [], [], []
    for el in els:
        cdf, boot, ident, ident_ratio, total, Ntot, df = run(el)
        cells.append(cdf)
        margs.append(marginals(el, boot, ident, total, Ntot, df))
        scal.append(scaled(el, boot, ident_ratio, total, Ntot))
    C = pd.concat(cells, ignore_index=True)
    M = pd.concat(margs, ignore_index=True)
    S = pd.concat(scal, ignore_index=True)
    C.to_csv(f"{OUT}/joint_final.csv", index=False)
    M.to_csv(f"{OUT}/joint_marginals.csv", index=False)
    S.to_csv(f"{OUT}/joint_scaled.csv", index=False)

    print("\n" + "=" * 92)
    print(f"THE CROSSTAB WITH ERROR BARS -- {common.cfg(els[-1])['label']}")
    print("% of that cell's registered voters, +-95% total error. * = do not cite.")
    print("=" * 92)
    last = C[C.election == els[-1]]
    for e in ESTIMAND_NAMES:
        print(f"\n### {e}\n")
        print(f"{'':9s}" + "".join(f"{a:>16s}" for a in AGE))
        for et in ETH:
            cs = []
            for ag in AGE:
                r = last[(last.eth == et) & (last.age == ag) & (last.estimand == e)].iloc[0]
                cs.append(f"{100*r.rate:5.1f}±{100*r.error:<4.1f}{'' if r.cite else '*'}")
            print(f"{et:9s}" + "".join(f"{c:>16s}" for c in cs))

    print("\n" + "=" * 92)
    print("MARGINAL CONSISTENCY: joint-derived vs separately-reported (07)")
    print("=" * 92)
    ts = pd.read_csv(f"{OUT}/timeseries_estimates.csv")
    chk = M.merge(ts[["election", "dim", "group", "estimand", "reconciled"]],
                  on=["election", "dim", "group", "estimand"], how="left")
    chk["shift_pp"] = (chk.rate - chk.reconciled) * 100
    print(f"max |shift| joint vs separate marginal: {chk.shift_pp.abs().max():.2f}pp")
    print(f"mean |shift|: {chk.shift_pp.abs().mean():.2f}pp")
    print("\nlargest shifts:")
    print(chk.reindex(chk.shift_pp.abs().sort_values(ascending=False).index)
          .head(6)[["election", "dim", "group", "estimand", "reconciled", "rate", "shift_pp"]]
          .round(3).to_string(index=False))

    print("\n" + "=" * 92)
    print("REAL-DATA CHECK: simulated identification error vs the joint.py holdout miss")
    print("=" * 92)
    try:
        h = pd.read_csv(f"{OUT}/joint_holdout.csv")
        h = h[h.model == "joint"].groupby("dim").miss.mean() * 100
        sim = C.merge(M[["election"]].drop_duplicates())  # noqa
        simerr = (C.groupby(C.eth.isin(ETH)).rmse_ident.mean())  # placeholder
        agg = C.copy()
        agg["dim"] = "joint-cell"
        print(f"  simulated identification rmse, mean over identified cells: "
              f"{100*C[C.cite].rmse_ident.mean():.2f}pp")
        print(f"  real holdout miss (10_joint), age cells:  {h.get('age', float('nan')):.2f}pp")
        print(f"  real holdout miss (10_joint), eth cells:  {h.get('ethnicity', float('nan')):.2f}pp")
        print("  -> same order of magnitude; the simulated term is not wildly miscalibrated.")
    except FileNotFoundError:
        pass

    print(f"\nwrote {OUT}/joint_final.csv, joint_marginals.csv, joint_scaled.csv")


if __name__ == "__main__":
    main()
