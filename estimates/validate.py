"""
Does any of this actually work? A simulation study on real Johor geography.

Every number in 02-04 rests on an assumption that cannot be checked against the Johor data
itself, because the individual-level truth is unobservable -- that is the whole reason for
doing ecological inference. What *can* be done is to manufacture a world where the truth is
known, run the identical pipeline on it, and measure the error. If the method cannot
recover a truth we planted, it should not be believed on a truth we did not.

DESIGN
Real saluran electorates and real demographic margins from the Johor roll are kept exactly
as they are, so the identifying variation, the homogeneity of the salurans and the
correlation between ethnicity and age are all the genuine article. Only behaviour is
synthetic: each registered voter is assigned a choice by a multinomial draw, then the
individual data is thrown away and only the two margins -- which is all the real data ever
gives us -- are handed to the pipeline.

Three worlds, in increasing order of hostility:

  A  homogeneous      b_igc = b_gc. The model's assumption is exactly true. This is the
                      best case and measures floor error: sampling noise only.
  B  context effects  b_igc = expit(beta_gc + delta_gc * z_i), z_i = centred Malay share.
                      A group's behaviour shifts with the makeup of its saluran, so the
                      constant-rate assumption is false in precisely the way ecological
                      regression is famous for failing. This is aggregation bias, planted.
  C  dun heterogen.   b_igc = expit(beta_gc + u_dun). Rates vary by dun at random, i.e.
                      real geographic variation uncorrelated with composition -- the case
                      M5 (within-dun) is designed for.

World B is the important one. Robinson (1950) is not a warning about noise, it is a warning
about bias, and bias does not shrink with 2.7 million voters. If the pipeline survives B
with tolerable error, the Johor estimates are credible; if it does not, the honest output is
the bounds and nothing more.

Reported per world: bias (signed), MAE, and -- the key column -- whether the deterministic
bounds still cover the truth. Bounds cover truth by arithmetic in every world, so a failure
there is a bug in the code rather than a fact about the method.
"""

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES, expit, logit

import bounds as bnd
import models
import reconcile as rec

RNG = np.random.default_rng(8112026)
NSIM = 30
DELTA_SD = 1.0   # logit-scale context slope in world B
U_SD = 0.5       # logit-scale dun effect in world C


def true_rates(df, el, dim):
    """A realistic G x C truth: the fitted seed, with rows normalised to sum to 1."""
    seed = rec.build_seed(df, el, dim)
    return seed / seed.sum(axis=1, keepdims=True)


def simulate(df, dim, B, world, rng):
    """Draw individual choices, return only the observable column margins Y (salurans x C).

    The row margins N are the real roll counts and are already known to the pipeline.
    """
    N = common.counts(df, dim).astype(int)
    G, C = B.shape
    z = models.context_var(df, dim)
    duns = pd.factorize(df.cluster)[0]

    if world == "A":
        P = np.repeat(B[None, :, :], len(df), axis=0)
    elif world == "B":
        delta = rng.normal(0, DELTA_SD, size=(G, C))
        lin = logit(np.clip(B, 1e-6, 1 - 1e-6))[None, :, :] + z[:, None, None] * delta[None, :, :]
        P = expit(lin)
    elif world == "C":
        u = rng.normal(0, U_SD, size=(duns.max() + 1, 1, C))
        lin = logit(np.clip(B, 1e-6, 1 - 1e-6))[None, :, :] + u[duns]
        P = expit(lin)
    P = P / P.sum(axis=2, keepdims=True)

    # Generator.multinomial broadcasts over salurans, so only the 4 groups need a loop.
    Y = np.zeros((len(df), C))
    truth = np.zeros((G, C))
    for g in range(G):
        cell = rng.multinomial(N[:, g], P[:, g, :])   # (n, C)
        Y += cell
        truth[g] = cell.sum(axis=0)
    return Y, truth / N.sum(axis=0)[:, None]


def run_pipeline(df, el, dim, Y):
    """Exactly what 02+03 do, but on simulated columns: fit seed, rake, read off rates."""
    sim = df.copy()
    for c, name in enumerate(common.choices(el)):
        sim[name] = Y[:, c]
    seed = rec.build_seed(sim, el, dim)
    total, _ = rec.reconcile(sim, el, dim, seed)
    N_tot = common.counts(sim, dim).sum(axis=0)
    m2 = {}
    X, V = common.shares(sim, dim), sim.V.to_numpy(float)
    for e in ESTIMAND_NAMES:
        Ye = sim[common.estimands(el)[e]].sum(axis=1).to_numpy(float)
        m2[e], _ = models.fit_link(X, Ye, V, "logit")
    return {e: rec.rates(total, N_tot, el, e) for e in ESTIMAND_NAMES}, m2, sim


def run(el):
    df = common.load(el)
    OUT = common.outdir(el)
    CHOICES = common.choices(el)
    rows = []

    for dim, gs in DIMS.items():
        B = true_rates(df, el, dim)
        for world in ["A", "B", "C"]:
            for s in range(NSIM):
                rng = np.random.default_rng(RNG.integers(1 << 32))
                Y, truth = simulate(df, dim, B, world, rng)
                recon, m2, sim = run_pipeline(df, el, dim, Y)
                for e in ESTIMAND_NAMES:
                    cols = [CHOICES.index(c) for c in common.estimands(el)[e]]
                    t = truth[:, cols].sum(axis=1)
                    lo, hi = bnd.aggregate(sim, el, dim, e)[:2]
                    for j, g in enumerate(gs):
                        rows.append(dict(dim=dim, world=world, sim=s, group=g, estimand=e,
                                         truth=t[j], reconciled=recon[e][j], m2=m2[e][j],
                                         bound_lo=lo[j], bound_hi=hi[j],
                                         covered=bool(lo[j] - 1e-6 <= t[j] <= hi[j] + 1e-6)))
            print(f"  {dim:9s} world {world}: {NSIM} sims")

    r = pd.DataFrame(rows)
    r["err_recon"] = r.reconciled - r.truth
    r["err_m2"] = r.m2 - r.truth
    r.to_csv(f"{OUT}/validation_sims.csv", index=False)

    assert r.covered.all(), "bounds failed to cover a planted truth -- this is a bug"
    print(f"  bounds covered the planted truth in 100% of cells")
    t = r.groupby(["dim", "world"]).err_recon.apply(lambda x: 100 * x.abs().mean()).unstack()
    print("  reconciled MAE (pp) by world:")
    print("    " + t.round(2).to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    for e in common.arg_elections():
        print(f"\n{'=' * 70}\n{e}  --  {common.cfg(e)['label']}\n{'=' * 70}")
        run(e)
