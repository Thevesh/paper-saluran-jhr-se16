"""
Impose the ecology constraint -- properly.

THE PROBLEM WITH CLIPPING
The obvious way to honour the bounds is to fit a model, then clip the statewide estimate
into [lo, hi]. That is what 03 also reports, as `capped`, because it was asked for. But it
is weak, for three reasons:

  1. It is slack almost everywhere. The statewide bound is an average of saluran bounds, so
     an estimate can sit comfortably inside it while implying impossible cell counts in
     hundreds of individual salurans -- e.g. more Chinese BN voters than there are Chinese
     registered in that saluran. Clipping never looks at the salurans.
  2. It breaks the accounting. Clip turnout for Malays and nothing re-balances the other
     groups, so the implied group turnouts no longer reproduce each saluran's actual
     ballots issued. The estimates stop being a decomposition of anything.
  3. It only ever binds when the model is already badly wrong, so it acts as a cosmetic
     patch on the symptom rather than using the constraint as information.

THE FIX
The bounds are not an extra assumption bolted on after estimation. They are exactly the
statement "cells are non-negative and both margins are respected" -- the Frechet bounds are
a *consequence* of the margins, not an independent fact. So imposing the margins on the
interior at the saluran level imposes the bounds automatically, everywhere, by construction.

For each saluran build the full group x choice table and rake it (iterative proportional
fitting) to its two known margins, starting from a seed supplied by the model:

    seed_gc = model's statewide rate of choice c in group g       (strictly positive)
    IPF ->  T_gc >= 0,  sum_c T_gc = N_g,  sum_g T_gc = Y_c

IPF converges to the unique table matching both margins while preserving the seed's odds
ratios (Deming & Stephan 1940; Ireland & Kullback 1968: it is the minimum-discrimination-
information table subject to the margins). This is structure-preserving estimation --
SPREE, Purcell & Kish 1980 -- and it is the standard small-area tool for exactly this
shape of problem: known margins, borrowed interior structure.

The division of labour is the attractive part, and it is automatic:
  - In a near-homogeneous saluran the margins nearly pin the interior, the Frechet bounds
    are nearly a point, and the seed barely matters. The data speak.
  - In a balanced saluran the margins say little and the seed carries the answer. The
    model speaks.
Estimates therefore lean on the assumption exactly in proportion to how little the
arithmetic settles, which is the behaviour you want and cannot get from post-hoc clipping.

The seed needs a rate for every choice column, so M2 is refitted across all 8 columns
rather than only the 4 headline estimands. The seeded rates need not sum to 1 within a
group -- the row margin forces that during raking.

Outputs the final bounds-consistent estimates, plus the naive capped variant, plus a
verification that every saluran-level Frechet bound is satisfied.
"""

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES

import models

SEED_FLOOR = 1e-6
MAXIT, TOL = 200, 1e-8
NBOOT = 200


def _safe_div(num, den):
    return np.divide(num, den, out=np.zeros_like(den), where=den > 0)


def ipf(seed, N, Y, maxit=MAXIT, tol=TOL):
    """Rake every saluran at once. Vectorised over salurans so this can be bootstrapped.

    seed : (G, C) statewide rate matrix, shared by all salurans -- it carries the model's
           association structure and nothing saluran-specific.
    N    : (n, G) row margins, registered by group.
    Y    : (n, C) column margins, the choice counts.
    Returns (n, G, C) raked tables.

    Margins agree in total by construction: both sum to the saluran's electorate. Zero
    margins are fine -- a zero row or column comes out zero and stays zero. The seed floor
    keeps every cell reachable, since IPF can never revive a structural zero.
    """
    T = np.broadcast_to(np.clip(seed, SEED_FLOOR, None), (len(N), *seed.shape)).copy()
    T *= (N[:, :, None] > 0) * (Y[:, None, :] > 0)
    for _ in range(maxit):
        T *= _safe_div(N, T.sum(axis=2))[:, :, None]
        T *= _safe_div(Y, T.sum(axis=1))[:, None, :]
        if np.abs(T.sum(axis=2) - N).max() < tol:
            break
    return T


def build_seed(df, el, dim, link="logit"):
    """Statewide group x choice rate matrix from M2, fitted column by column."""
    X = common.shares(df, dim)
    V = df.V.to_numpy(float)
    CHOICES = common.choices(el)
    seed = np.zeros((X.shape[1], len(CHOICES)))
    for c, name in enumerate(CHOICES):
        Y = df[name].to_numpy(float)
        seed[:, c], _ = models.fit_link(X, Y, V, link)
    return seed


def reconcile(df, el, dim, seed):
    """Rake every saluran and sum the interiors. Returns (total G x C, per-saluran tables)."""
    N = common.counts(df, dim)
    Y = df[common.choices(el)].to_numpy(float)
    per_saluran = ipf(seed, N, Y)
    return per_saluran.sum(axis=0), per_saluran


def rates(total, N_tot, el, estimand):
    CHOICES = common.choices(el)
    cols = [CHOICES.index(c) for c in common.estimands(el)[estimand]]
    return total[:, cols].sum(axis=1) / N_tot


def bootstrap(df, el, dim, nboot=NBOOT, rng=None):
    """Cluster bootstrap over contested seats, through the whole seed-then-rake pipeline.

    Resamples seats, refits the seed on the resample, rakes the resample to its own margins.
    This carries seed uncertainty into the final answer. It does NOT carry the deeper
    uncertainty -- that the seed's constant-odds-ratio structure might be wrong -- which no
    resampling of these data can measure, because it is an identification problem rather
    than a sampling one. That is what the bounds and validate.py are for, and it is why
    these intervals should be read as the narrower of the two uncertainties, not the total.
    """
    rng = rng or np.random.default_rng(11072026)
    cls = df.cluster.unique()
    idx = {d: np.flatnonzero((df.cluster == d).to_numpy()) for d in cls}
    draws = []
    for _ in range(nboot):
        pick = rng.choice(cls, size=len(cls), replace=True)
        d2 = df.iloc[np.concatenate([idx[d] for d in pick])]
        seed = build_seed(d2, el, dim)
        total, _ = reconcile(d2, el, dim, seed)
        N_tot = common.counts(d2, dim).sum(axis=0)
        draws.append({e: rates(total, N_tot, el, e) for e in ESTIMAND_NAMES})
    return draws


def run(el):
    df = common.load(el)
    OUT = common.outdir(el)
    CHOICES = common.choices(el)
    bounds = pd.read_csv(f"{OUT}/bounds_statewide.csv")
    est = pd.read_csv(f"{OUT}/model_estimates.csv")
    out_rows, checks, draw_rows = [], [], []

    for dim, gs in DIMS.items():
        seed = build_seed(df, el, dim)
        total, per = reconcile(df, el, dim, seed)
        N = common.counts(df, dim)
        N_tot = N.sum(axis=0)
        boot = bootstrap(df, el, dim)
        print(f"  [{dim}] seed fitted, {len(df):,} salurans raked, {NBOOT} bootstraps")

        # Keep the raw draws, not just their summary. Any quantity derived from more than
        # one estimand -- e.g. BN/turnout, support among those who actually voted -- has to
        # be recomputed inside each draw, because numerator and denominator are strongly
        # correlated here (BN votes are a subset of turnout). Combining their separate SEs
        # after the fact would assume an independence that does not hold and would inflate
        # the error. Cheap to store; impossible to reconstruct later.
        for b, d in enumerate(boot):
            for estimand, vals in d.items():
                for j, g in enumerate(gs):
                    draw_rows.append(dict(dim=dim, group=g, estimand=estimand,
                                          draw=b, value=vals[j]))

        # --- verification: every saluran's Frechet bound must hold, by construction
        V = df.V.to_numpy(float)
        for estimand in ESTIMAND_NAMES:
            cols = [CHOICES.index(c) for c in common.estimands(el)[estimand]]
            y = per[:, :, cols].sum(axis=2)                       # salurans x groups
            Y = common.outcome(df, el, estimand)[:, None]
            lo = np.maximum(0.0, Y - (V[:, None] - N))
            hi = np.minimum(Y, N)
            checks.append(dict(dim=dim, estimand=estimand,
                               n_cells=int((N > 0).sum()),
                               viol_lo=int((y < lo - 1e-6).sum()),
                               viol_hi=int((y > hi + 1e-6).sum()),
                               max_abs_margin_err=float(
                                   np.abs(per.sum(axis=2).sum(axis=1) - V).max())))

        for estimand in ESTIMAND_NAMES:
            r = rates(total, N_tot, el, estimand)
            bd = np.array([d[estimand] for d in boot])
            for j, g in enumerate(gs):
                b = bounds[(bounds.dim == dim) & (bounds.group == g) &
                           (bounds.estimand == estimand)].iloc[0]
                m2 = est[(est.dim == dim) & (est.group == g) & (est.estimand == estimand) &
                         (est.model == "M2_logit")].iloc[0]
                out_rows.append(dict(
                    dim=dim, group=g, estimand=estimand, registered=int(N_tot[j]),
                    bound_lo=b.lo, bound_hi=b.hi,
                    m2_raw=m2.estimate,
                    capped=float(np.clip(m2.estimate, b.lo, b.hi)),   # the naive version
                    reconciled=r[j],                                   # the real answer
                    ci_lo=float(np.percentile(bd[:, j], 2.5)),
                    ci_hi=float(np.percentile(bd[:, j], 97.5)),
                    se=float(bd[:, j].std()),
                ))

    res = pd.DataFrame(out_rows)
    res.to_csv(f"{OUT}/reconciled.csv", index=False)
    pd.DataFrame(draw_rows).to_parquet(f"{OUT}/bootstrap_draws.parquet", index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(f"{OUT}/reconcile_checks.csv", index=False)

    assert chk.viol_lo.sum() == 0 and chk.viol_hi.sum() == 0, \
        f"{el}: raking left a saluran outside its Frechet bounds -- this is a bug"
    print(f"  verified: 0 bound violations in {int(chk.n_cells.sum()):,} saluran cells; "
          f"margins matched to {chk.max_abs_margin_err.max():.0e}")
    d = 100 * (res.reconciled - res.capped).abs()
    print(f"  reconciled vs naive capped: mean |diff| {d.mean():.2f}pp, max {d.max():.2f}pp")


if __name__ == "__main__":
    for e in common.arg_elections():
        print(f"\n{'=' * 70}\n{e}  --  {common.cfg(e)['label']}\n{'=' * 70}")
        run(e)
