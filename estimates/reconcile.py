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

    seed_igc = model's rate of choice c in group g, for saluran i's contest  (strictly positive)
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

The seed is CONTEST-AWARE (seed_tables): each coalition column is fitted only on the seats it
contested, separately for each component party that stood (BN-UMNO and BN-MCA seats get their
own profiles), and the local bundles (OTHER, BERSAMA) are fitted seat by seat. The original
Johor pipeline used one statewide seed per column. That is wrong wherever coalitions stand down
or field different component parties, and seatcheck.py shows how wrong.

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

    seed : (G, C) rate matrix shared by all salurans, or (n, G, C) one per saluran -- the
           contest-aware seed (seed_tables) is per saluran, because who was on the ballot,
           and so the profile of a column, differs from seat to seat.
    N    : (n, G) row margins, registered by group.
    Y    : (n, C) column margins, the choice counts.
    Returns (n, G, C) raked tables.

    Margins agree in total by construction: both sum to the saluran's electorate. Zero
    margins are fine -- a zero row or column comes out zero and stays zero. The seed floor
    keeps every cell reachable, since IPF can never revive a structural zero.
    """
    seed = np.clip(seed, SEED_FLOOR, None)
    if seed.ndim == 2:
        seed = np.broadcast_to(seed, (len(N), *seed.shape))
    T = seed.copy()
    T *= (N[:, :, None] > 0) * (Y[:, None, :] > 0)
    for _ in range(maxit):
        T *= _safe_div(N, T.sum(axis=2))[:, :, None]
        T *= _safe_div(Y, T.sum(axis=1))[:, None, :]
        if np.abs(T.sum(axis=2) - N).max() < tol:
            break
    return T


# ---------------------------------------------------------------- the contest-aware seed
# A column's seed must be fitted only where that column was actually on the ballot, and on a
# profile that describes the candidates behind it:
#
#   CONTEST  a coalition's column is structurally zero in every seat it did not contest (BN
#            stood down from 23 of Johor's 56 duns at SE-16; in Negeri Sembilan BN and PN never
#            met at SE-16, and BN and PH divided the seats at SE-15). Fitting its
#            seed on those zeros mistakes a contest decision for behaviour and distorts the
#            odds ratios carried into every seat it did contest. -> fit on contested seats.
#   LOCAL    OTHER (and SE-16's BERSAMA) bundle different candidates in different seats -- in
#            Negeri Sembilan, Berjasa in Nilai, PSM, ASLI, independents elsewhere -- so no statewide
#            profile describes any one of them. There a statewide seed put Nilai's Berjasa vote (40% among
#            near-pure Malay streams, 2% among near-pure non-Malay) onto Indian and Other
#            voters, and raking, which preserves the seed's odds ratios, could not undo it.
#            -> fit separately within each contested seat, on ethnicity only (the
#            dimension a local candidate's appeal runs along, and few enough parameters for
#            one seat's ~45 salurans), each joint cell taking its ethnicity's rate.
#   the rest REJECTED / NOTRETURNED / NONVOTE are not contests: statewide, as before.
#
# Every column's seed then still feeds one rake per saluran onto both known margins, so the
# bounds, exact totals and the rest of the method are unchanged.
CONTEST = {"BN", "PH", "PN", "BERSAMA", "OTHER"}
LOCAL = {"BERSAMA", "OTHER"}


def contested(df, name):
    """Salurans in seats (clusters) where column `name` received any vote."""
    tot = df.groupby("cluster")[name].transform("sum").to_numpy()
    return tot > 0


SEEDS = ("candidate", "contest", "statewide")
# see seatcheck.py for how the three compare; SALURAN_SEED overrides, for comparisons only
SEED = __import__("os").environ.get("SALURAN_SEED", "candidate")


def candidate_party(el):
    """(seat, coalition column) -> the component party that stood there, e.g. BN -> MCA."""
    c = common.cfg(el)
    b = pd.read_parquet(c["ballots"]).query("state == @common.STATE")
    b = b[b.coalition.isin(common.PARTIES)].drop_duplicates(["seat", "coalition"])
    return {(r.seat, r.coalition): r.party for r in b.itertuples()}


def seed_tables(df, el, X, M=None, link="logit", seed=None):
    """Contest-aware seed. X: (n, G) row shares. M: (G, K) map from rows to the ethnic
    groups a LOCAL column is fitted on (identity when the rows are themselves the groups to
    use). Returns (per-saluran seed (n, G, C), statewide summary (G, C)); the summary is
    only for the simulation studies, which plant one statewide truth.

    seed: "candidate" (default) -- coalition columns are fitted per component party actually
          standing (BN-UMNO and BN-MCA seats get different profiles), on contested seats only;
          "contest" -- coalition columns fitted once over the seats they contested;
          "statewide" -- the original: every column fitted on every saluran.
    LOCAL columns are fitted per seat under both "candidate" and "contest"."""
    seed = seed or SEED
    assert seed in SEEDS, seed
    V = df.V.to_numpy(float)
    CH = common.choices(el)
    n, G = X.shape
    M = np.eye(G) if M is None else M
    XK = X @ M
    seat = df.cluster.to_numpy()
    cand = candidate_party(el) if seed == "candidate" else {}
    S = np.full((n, G, len(CH)), SEED_FLOOR)
    B = np.zeros((G, len(CH)))
    for c, name in enumerate(CH):
        y = df[name].to_numpy(float)
        on = (contested(df, name) if (name in CONTEST and seed != "statewide")
              else np.ones(n, bool))
        if not on.any():
            continue
        b, _ = models.fit_link(X[on], y[on], V[on], link)
        B[:, c] = b
        if name in LOCAL and seed != "statewide":
            for s in np.unique(seat[on]):
                i = seat == s
                bk, _ = models.fit_link(XK[i], y[i], V[i], link)
                S[i, :, c] = M @ bk
        elif name in common.PARTIES and seed == "candidate":
            party = np.array([cand.get((st, name), "") for st in seat])
            for p in np.unique(party[on]):
                i = on & (party == p)
                bp, _ = models.fit_link(X[i], y[i], V[i], link, start=np.log(b / (1 - b))
                                        if link == "logit" else None)
                S[i, :, c] = bp
        else:
            S[on, :, c] = b
    return S, B


MIN_SEAT_N = 30   # below this a seat's own cell count is too thin; fall back to statewide


def seat_rates(per, N_fit, seats_fit, seats_target, el):
    """Per-target-saluran group rates, read from the fit IN THE TARGET'S OWN SEAT.

    per: (n_fit, G, C) raked tables of the fitted (mixed) salurans; N_fit: (n_fit, G) their
    row counts. A held-out stream must be predicted with the rates of the contest it sat in --
    under a stand-down, a statewide BN rate averages BN's own seats with seats where BN was
    not on the ballot, which is the wrong prediction for any stream under any model. Falls
    back to the statewide rate for a group thinner than MIN_SEAT_N in the seat's mixed set.
    Returns {estimand: (n_target, G)}."""
    CH = common.choices(el)
    cols = {e: [CH.index(c) for c in common.estimands(el)[e]] for e in ESTIMAND_NAMES}
    Tall, Nall = per.sum(axis=0), N_fit.sum(axis=0)
    state = {e: Tall[:, cols[e]].sum(axis=1) / np.maximum(Nall, 1e-9) for e in ESTIMAND_NAMES}
    out = {e: np.tile(state[e], (len(seats_target), 1)) for e in ESTIMAND_NAMES}
    for s in np.unique(seats_target):
        mf, mt = seats_fit == s, seats_target == s
        if not mf.any():
            continue
        T, N = per[mf].sum(axis=0), N_fit[mf].sum(axis=0)
        ok = N >= MIN_SEAT_N
        for e in ESTIMAND_NAMES:
            r = np.where(ok, T[:, cols[e]].sum(axis=1) / np.maximum(N, 1e-9), state[e])
            out[e][mt] = r
    return out


def build_seed(df, el, dim, link="logit", seed=None):
    """Per-saluran contest-aware seed for a one-dimension model (see seed_tables)."""
    return seed_tables(df, el, common.shares(df, dim), link=link, seed=seed)[0]


def statewide_seed(df, el, dim, link="logit"):
    """The statewide G x C summary of the same fit -- a planted truth for simulations."""
    return seed_tables(df, el, common.shares(df, dim), link=link)[1]


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
