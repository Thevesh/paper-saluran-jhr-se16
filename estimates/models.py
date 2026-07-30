"""
Ecological regression models for group-level turnout and party support.

THE ESTIMAND
    b_g = (registered voters in group g who did X) / (registered voters in group g)
where X is "turned out" or "voted BN/PH/PN". The denominator is *registration*, not
turnout, so abstention is one of the choices being modelled rather than a filter applied
before modelling. This matches the demographic margins, which cover the whole electorate.

THE MODELS
All share the accounting identity that makes ecological inference possible. If group g
does X at rate b_g inside saluran i, the saluran's observed rate is the composition-
weighted average of the group rates:

    p_i = sum_g x_ig * b_ig          x_ig = group g's share of saluran i's electorate

The models differ only in how b_ig is parameterised and how it is fitted:

  M1  Goodman LPM        b_ig = beta_g                     WLS, weights V_i
  M2  Ecological logit   b_ig = expit(beta_g)              binomial ML
  M3  Ecological probit  b_ig = Phi(beta_g)                binomial ML
  M4  Logit + context    b_ig = expit(beta_g + gamma_g z_i)  binomial ML
  M5  Logit, within-seat M2 fitted separately in each contested seat, then aggregated

A point worth being explicit about, because it is easy to get wrong: in M1-M3 the link is
applied to *each group's rate*, not to the saluran's linear predictor. p_i stays linear in
the group rates no matter which link is used, because the mixture weights x_ig are data.
So M2 and M3 are reparameterisations of M1 that force every rate into (0,1); they can only
differ from M1 where M1 wants to leave the unit interval, and from each other essentially
not at all. Running all three is a check on the constraint binding, not a menu of rival
theories. Fitting expit(sum_g x_ig beta_g) instead -- the link on the linear predictor --
would be a different and wrong model here: its coefficients are not group rates and it has
no interpretation as a decomposition. It is not fitted.

Where the link genuinely earns its keep is M4 and M5, which relax the assumption that does
the real work in M1-M3: that a group's rate does not vary with where its members live.

AGGREGATION BIAS
The assumption b_ig = b_g for all i is the known weak point of ecological regression
(Robinson 1950; Goodman 1953; King 1997). If Malays in heavily-Malay salurans behave
differently from Malays in mixed ones, b_g is not recoverable from aggregates and M1-M3
absorb the difference into the coefficients. M4 tests this directly by letting each group's
rate slide with saluran composition; M5 restricts identification to within-seat contrasts,
purging the between-seat component of the bias. Divergence between M2, M4 and M5 measures
how much the answer is being driven by that assumption rather than by the data.

Standard errors are a cluster bootstrap over contested seats (56 duns at SE-15/SE-16, 26
parlimens at GE-15), the level at which a campaign runs and therefore the level at which the
ecological correlation lives; iid saluran-level SEs would be far too small.

Writes tidy per-model estimates for reconcile.py and workbook.py.
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import common
from common import DIMS, ESTIMAND_NAMES, expit, logit

RNG = np.random.default_rng(20260711)
NBOOT = 400
EPS = 1e-9

LINKS = {
    "logit": (expit, lambda b: expit(b) * (1 - expit(b))),
    "probit": (norm.cdf, norm.pdf),
}


# Weakly-informative N(0, PRIOR_SD^2) prior on each beta, per Gelman et al. (2008).
# Rationale: at SE-16 PN takes 3.7% of the electorate and is near-absent among non-Malays, so
# unpenalised likelihood is maximised by driving those betas to -inf -- a corner solution
# that reports a rate of exactly 0.000% with an infinite coefficient and a meaningless
# standard error. On the logit scale PRIOR_SD = 2.5 spans rates from ~0.7% to ~99.3%, so
# against ~2.7m individuals it is negligible for any group the data actually identify; it
# only bites where the likelihood is flat out to infinity, pulling the corner back to a
# finite interior point. Set PRIOR_SD = None to recover pure ML.
PRIOR_SD = 2.5


def _nll(beta, X, Y, V, f, fp, prior_sd=PRIOR_SD, offset=0.0):
    b = f(beta)
    p = np.clip(X @ b + offset, EPS, 1 - EPS)
    nll = -(Y * np.log(p) + (V - Y) * np.log(1 - p)).sum()
    grad = -(X * ((Y / p - (V - Y) / (1 - p)))[:, None]).sum(axis=0) * fp(beta)
    if prior_sd is not None:
        nll += (beta ** 2).sum() / (2 * prior_sd ** 2)
        grad += beta / prior_sd ** 2
    return nll, grad


def fit_link(X, Y, V, link, start=None, prior_sd=PRIOR_SD, offset=0.0):
    """Penalised binomial ML for p_i = sum_g x_ig f(beta_g) + offset_i. Returns group rates.

    `offset` carries any part of the mixture held fixed rather than fitted -- see
    fit_within_cluster, where groups dropped from a seat's design contribute through it.
    """
    f, fp = LINKS[link]
    rate = np.clip((Y.sum() / V.sum()), 0.02, 0.98)
    b0 = start if start is not None else np.full(X.shape[1], logit(rate) if link == "logit"
                                                 else norm.ppf(rate))
    r = minimize(_nll, b0, args=(X, Y, V, f, fp, prior_sd, offset), jac=True,
                 method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-12))
    return f(r.x), r


def fit_lpm(X, Y, V):
    """Goodman regression: WLS of the saluran rate on group shares, no intercept.

    Weights are V_i, the binomial precision up to the p(1-p) factor. Coefficients are the
    group rates directly and are free to leave [0, 1] -- that freedom is the diagnostic.
    """
    y = Y / V
    w = np.sqrt(V)
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return beta


def fit_context(X, Y, V, z):
    """Logit with a composition slope: b_ig = expit(beta_g + gamma_g * z_i).

    z is the saluran's centred share of the reference group, so gamma_g = 0 recovers M2.
    A gamma_g far from zero says group g's rate moves with the makeup of its saluran,
    which is aggregation bias made visible.
    """
    k = X.shape[1]

    def obj(th):
        beta, gamma = th[:k], th[k:]
        lin = beta[None, :] + np.outer(z, gamma)
        b = expit(lin)
        p = np.clip((X * b).sum(axis=1), EPS, 1 - EPS)
        nll = -(Y * np.log(p) + (V - Y) * np.log(1 - p)).sum()
        r = (Y / p - (V - Y) / (1 - p))
        d = X * b * (1 - b) * r[:, None]
        return nll, -np.concatenate([d.sum(axis=0), (d * z[:, None]).sum(axis=0)])

    rate = np.clip(Y.sum() / V.sum(), 0.02, 0.98)
    th0 = np.concatenate([np.full(k, logit(rate)), np.zeros(k)])
    r = minimize(obj, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=4000, ftol=1e-12))
    beta, gamma = r.x[:k], r.x[k:]
    # Report each group's rate at the electorate-weighted mean context, i.e. the
    # statewide average saluran a member of that group actually lives in.
    wz = (X * V[:, None] * z[:, None]).sum(axis=0) / (X * V[:, None]).sum(axis=0)
    return expit(beta + gamma * wz), gamma


def fit_within_cluster(df, X, Y, V, link="logit"):
    """M2 fitted inside each contested seat, then pooled by registered counts.

    Identification comes only from saluran-to-saluran contrasts within a seat, so any
    between-seat confounding of composition with behaviour drops out. A seat whose salurans
    are compositionally flat cannot identify its own rates; it is skipped for that group
    and contributes no weight.
    """
    k = X.shape[1]
    num, den = np.zeros(k), np.zeros(k)
    skipped = 0
    for _, idx in df.groupby("cluster").indices.items():
        Xi, Yi, Vi = X[idx], Y[idx], V[idx]
        Ni = (Xi * Vi[:, None]).sum(axis=0)
        # need real spread in a group's share for its rate to be identified here
        ok = (Ni > 0) & (Xi.std(axis=0) > 0.02)
        if ok.sum() == 0:
            skipped += 1
            continue
        # Groups dropped from the design still cast the ballots sitting in Yi. Carry them as
        # a fixed offset at the seat's overall rate, so the fitted groups' coefficients are
        # not forced to absorb the dropped groups' votes.
        off = Xi[:, ~ok].sum(axis=1) * (Yi.sum() / Vi.sum())
        b, _ = fit_link(Xi[:, ok], Yi, Vi, link, offset=off)
        num[ok] += b * Ni[ok]
        den[ok] += Ni[ok]
    return np.where(den > 0, num / np.maximum(den, 1), np.nan), den, skipped


# ---------------------------------------------------------------- driver
def context_var(df, dim):
    """Centred saluran composition, used as the context regressor in M4."""
    X = common.shares(df, dim)
    z = X[:, 0]  # Malay share / 18-29 share -- the leading group of the partition
    return z - np.average(z, weights=df.V)


def fit_all(df, el, dim, estimand):
    X = common.shares(df, dim)
    Y = common.outcome(df, el, estimand)
    V = df.V.to_numpy(float)
    z = context_var(df, dim)
    out = {}
    out["M1_lpm"] = fit_lpm(X, Y, V)
    out["M2_logit"], _ = fit_link(X, Y, V, "logit")
    out["M3_probit"], _ = fit_link(X, Y, V, "probit")
    # Pure ML, no prior: kept so the prior's effect is visible rather than buried. Where
    # M2 and M2ml agree the prior is doing nothing; where they diverge the likelihood was
    # running to a corner and the gap is the prior alone holding it back.
    out["M2ml_logit_noprior"], _ = fit_link(X, Y, V, "logit", prior_sd=None)
    out["M4_context"], gamma = fit_context(X, Y, V, z)
    out["M5_within_seat"], _, _ = fit_within_cluster(df, X, Y, V)
    return out, gamma


def bootstrap(df, el, dim, estimand, nboot=NBOOT):
    """Cluster bootstrap over contested seats for M1/M2/M5 (M3 tracks M2; M4 is diagnostic)."""
    cls = df.cluster.unique()
    idx_by = {d: np.flatnonzero((df.cluster == d).to_numpy()) for d in cls}
    draws = {m: [] for m in ["M1_lpm", "M2_logit", "M5_within_seat"]}
    for _ in range(nboot):
        pick = RNG.choice(cls, size=len(cls), replace=True)
        d2 = df.iloc[np.concatenate([idx_by[d] for d in pick])].copy()
        # resampled seats repeat; relabel so within-seat fitting sees them as distinct
        d2["cluster"] = np.concatenate([[f"{d}#{j}"] * len(idx_by[d]) for j, d in enumerate(pick)])
        X, Y, V = common.shares(d2, dim), common.outcome(d2, el, estimand), d2.V.to_numpy(float)
        try:
            draws["M1_lpm"].append(fit_lpm(X, Y, V))
            draws["M2_logit"].append(fit_link(X, Y, V, "logit")[0])
            draws["M5_within_seat"].append(fit_within_cluster(d2, X, Y, V)[0])
        except Exception:
            continue
    return {m: np.array(v) for m, v in draws.items()}


def run(el):
    df = common.load(el)
    OUT = common.outdir(el)
    bounds = pd.read_csv(f"{OUT}/bounds_statewide.csv")
    rows, gammas = [], []

    for dim, gs in DIMS.items():
        for estimand in ESTIMAND_NAMES:
            est, gamma = fit_all(df, el, dim, estimand)
            boot = bootstrap(df, el, dim, estimand)
            for j, g in enumerate(gs):
                for m, vals in est.items():
                    b = boot.get(m)
                    se = np.nanstd(b[:, j]) if b is not None and len(b) else np.nan
                    lo = np.nanpercentile(b[:, j], 2.5) if b is not None and len(b) else np.nan
                    hi = np.nanpercentile(b[:, j], 97.5) if b is not None and len(b) else np.nan
                    rows.append(dict(dim=dim, group=g, estimand=estimand, model=m,
                                     estimate=vals[j], se=se, ci_lo=lo, ci_hi=hi))
                gammas.append(dict(dim=dim, group=g, estimand=estimand, gamma=gamma[j]))
            print(f"  {dim:9s} {estimand:8s}  " +
                  "  ".join(f"{g}={100 * est['M2_logit'][j]:.1f}" for j, g in enumerate(gs)))

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/model_estimates.csv", index=False)
    pd.DataFrame(gammas).to_csv(f"{OUT}/context_slopes.csv", index=False)

    # Which model estimates are arithmetically impossible? This is the whole point of
    # having computed the bounds, and it is the diagnostic that motivates 03_reconcile.
    chk = res.merge(bounds[["dim", "group", "estimand", "lo", "hi"]],
                    on=["dim", "group", "estimand"])
    # 1e-4 = 0.01pp. A logit fit that saturates lands exactly *on* a bound to within
    # optimiser tolerance; that is the constraint binding, not a violation, and flagging
    # it at 1e-6 just reports floating-point noise as an impossibility.
    TOL = 1e-4
    chk["violates"] = (chk.estimate < chk.lo - TOL) | (chk.estimate > chk.hi + TOL)
    chk["excess"] = np.maximum(chk.lo - chk.estimate, chk.estimate - chk.hi).clip(lower=0)
    chk.to_csv(f"{OUT}/model_vs_bounds.csv", index=False)

    print(f"\n[{el}] UNCONSTRAINED FITS vs DETERMINISTIC BOUNDS")
    print(f"{'model':16s} {'n_est':>6s} {'n_outside':>10s} {'worst_pp':>9s}")
    for m, gdf in chk.groupby("model"):
        print(f"{m:16s} {len(gdf):6d} {int(gdf.violates.sum()):10d} "
              f"{100 * gdf.excess.max():9.1f}")

    v = chk[chk.violates]
    if not v.empty:
        print("\nOutside the bounds (arithmetically impossible, must be reconciled):")
        print(v[["dim", "group", "estimand", "model", "estimate", "lo", "hi", "excess"]]
              .assign(**{c: lambda d, c=c: (100 * d[c]).round(1)
                         for c in ["estimate", "lo", "hi", "excess"]})
              .to_string(index=False))


if __name__ == "__main__":
    for e in common.arg_elections():
        print(f"\n{'=' * 70}\n{e}  --  {common.cfg(e)['label']}\n{'=' * 70}")
        run(e)
