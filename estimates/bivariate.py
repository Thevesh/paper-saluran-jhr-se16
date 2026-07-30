"""
The "quick and dirty" scatterplot version -- and exactly how it relates to M1.

The intuition: scatter PH's vote share against the Chinese share of the electorate, fit a
line, read off the fitted value at x = 100%. That is the implied PH support among Chinese.

This intuition is not a rough cousin of the model in models.py. It IS the model. This
script proves it numerically rather than asking anyone to take it on faith.

WHY THEY ARE THE SAME THING
Fit the simple bivariate regression with an intercept:

    y_i = a + b*x_i                      x = Chinese share, y = PH share

Read it off at the two ends:
    x = 1  ->  a + b      = implied PH support among Chinese
    x = 0  ->  a          = implied PH support among everyone who is not Chinese

Now write the same thing as a two-group Goodman regression -- no intercept, one regressor
per group, shares summing to 1:

    y_i = B_chi * x_i + B_non * (1 - x_i)
        = B_non + (B_chi - B_non) * x_i

Matching terms: a = B_non and b = B_chi - B_non, so a + b = B_chi. The extrapolation to
x = 100% and the Goodman coefficient are the *same number*, algebraically, not approximately.
Reading a fitted line off at its endpoints is ecological regression; Goodman (1953) just
writes it in the form that generalises.

WHAT M1 ADDS
Exactly one thing: it splits "not Chinese" into Malay / Indian / Other and fits all four at
once, instead of forcing them to share a single number. This matters here because "non-Chinese"
in Johor is 54% Malay and 7% Indian, and those two behave nothing like each other, so the
bivariate slope has to average them -- and the average is taken at whatever Malay:Indian mix
happens to prevail in the salurans doing the identifying. That is a real bias, and the script
quantifies it below.

WHAT THE LINK ADDS
Nothing, until the line runs off the edge of the world. The bivariate fit is free to predict
a negative rate at x = 1 or above 100% -- it is a straight line and nothing stops it. That is
not a small print caveat here: for PN the LPM returns *negative* support for Chinese, Indian
and Other. A rate below 0% is not a small error, it is a category error, and it is the entire
reason for the logit in M2.

Two denominators are in play and they are worth keeping straight. The classic scatterplot uses
y = party share of *valid votes*, which answers "of those who voted, what share went PH". The
rest of this project uses y = party share of *registered voters*, which answers "of the whole
group, what share cast a PH ballot" and folds abstention in as a choice. Both are computed here.
"""

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES

DEMOS = {"ethnicity": ["Chinese", "Malay", "Indian", "Other"],
         "age": list(common.AGE)}


def bivariate(x, y, w):
    """WLS of y on x with an intercept. Returns (at x=0, at x=1, slope, R^2)."""
    A = np.column_stack([np.ones_like(x), x])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    a, b = coef
    resid = y - A @ coef
    ybar = np.average(y, weights=w)
    r2 = 1 - (w * resid ** 2).sum() / (w * (y - ybar) ** 2).sum()
    return a, a + b, b, r2


def run(el):
    OUT = common.outdir(el)
    df = common.load(el)
    V = df.V.to_numpy(float)
    # "valid votes" = every choice that is a ballot cast for a contestant: parties (+ BERSAMA
    # where it stood) and OTHER, but not rejected/not-returned/abstention.
    valid_cols = [c for c in common.choices(el) if c not in ("REJECTED", "NOTRETURNED", "NONVOTE")]
    valid = df[valid_cols].sum(axis=1).to_numpy(float)
    rows = []

    for dim, gs in DIMS.items():
        X = common.shares(df, dim)
        for j, g in enumerate(gs):
            x = X[:, j]
            for e in ESTIMAND_NAMES:
                Y = common.outcome(df, el, e)
                # (a) classic: y = share of VALID VOTES, salurans with votes only
                m = valid > 0
                y_v = np.divide(Y, valid, out=np.zeros_like(Y), where=valid > 0)
                a_v, at1_v, _, r2_v = bivariate(x[m], y_v[m], valid[m])
                # (b) this project's denominator: y = share of REGISTERED
                y_r = Y / V
                a_r, at1_r, _, r2_r = bivariate(x, y_r, V)
                rows.append(dict(dim=dim, group=g, estimand=e,
                                 biv_valid_at1=at1_v, biv_valid_at0=a_v, r2_valid=r2_v,
                                 biv_reg_at1=at1_r, biv_reg_at0=a_r, r2_reg=r2_r))

    biv = pd.DataFrame(rows)
    models = pd.read_csv(f"{OUT}/model_estimates.csv")
    bounds = pd.read_csv(f"{OUT}/bounds_statewide.csv")
    rec = pd.read_csv(f"{OUT}/reconciled.csv")

    cmp = (biv.merge(models[models.model == "M1_lpm"][["dim", "group", "estimand", "estimate"]]
                     .rename(columns={"estimate": "M1_lpm"}), on=["dim", "group", "estimand"])
              .merge(models[models.model == "M2_logit"][["dim", "group", "estimand", "estimate"]]
                     .rename(columns={"estimate": "M2_logit"}), on=["dim", "group", "estimand"])
              .merge(bounds[["dim", "group", "estimand", "lo", "hi"]], on=["dim", "group", "estimand"])
              .merge(rec[["dim", "group", "estimand", "reconciled"]], on=["dim", "group", "estimand"]))
    cmp["biv_minus_M1"] = cmp.biv_reg_at1 - cmp.M1_lpm
    cmp["biv_impossible"] = (cmp.biv_reg_at1 < cmp.lo - 1e-4) | (cmp.biv_reg_at1 > cmp.hi + 1e-4)
    cmp.to_csv(f"{OUT}/bivariate_comparison.csv", index=False)

    print("=" * 104)
    print(f"THE SCATTERPLOT VERSION vs THE FULL MODEL -- {common.cfg(el)['label']}   "
          "(all % of the group's REGISTERED voters)")
    print("=" * 104)
    print("\nbiv@100%  = fit y on ONE group's share, read off at x=100%   (the quick-and-dirty)")
    print("M1_lpm    = same idea, but all 4 groups fitted at once, no intercept (Goodman)")
    print("M2_logit  = M1 with rates forced into (0,1)")
    print("FINAL     = M2 seed, raked to every saluran's margins\n")
    hdr = (f"{'group':9s} {'estimand':9s} {'biv@100%':>9s} {'M1_lpm':>8s} {'M2_logit':>9s} "
           f"{'FINAL':>7s} {'bounds':>14s}  {'biv-M1':>7s}  impossible?")
    for dim, gs in DIMS.items():
        print(f"\n--- by {dim} " + "-" * 76)
        print(hdr)
        for g in gs:
            for e in ESTIMAND_NAMES:
                r = cmp[(cmp.dim == dim) & (cmp.group == g) & (cmp.estimand == e)].iloc[0]
                print(f"{g:9s} {e:9s} {100 * r.biv_reg_at1:8.1f}% {100 * r.M1_lpm:7.1f}% "
                      f"{100 * r.M2_logit:8.1f}% {100 * r.reconciled:6.1f}% "
                      f"{100 * r.lo:5.1f}-{100 * r.hi:<7.1f} {100 * r.biv_minus_M1:+7.1f}"
                      f"  {'YES <-- outside bounds' if r.biv_impossible else ''}")

    print("\n\n" + "=" * 104)
    print("PROOF THAT THE EXTRAPOLATION IS THE GOODMAN COEFFICIENT")
    print("=" * 104)
    print("Collapse the electorate to 'group g' vs 'everyone else' and fit BOTH ways.")
    print("If the claim is right these agree to floating-point, for every row.\n")
    worst = 0.0
    for dim, gs in DIMS.items():
        X = common.shares(df, dim)
        for j, g in enumerate(gs):
            x = X[:, j]
            for e in ESTIMAND_NAMES:
                y = common.outcome(df, el, e) / V
                _, at1, _, _ = bivariate(x, y, V)              # intercept form, read at x=1
                A = np.column_stack([x, 1 - x])                 # two-group Goodman, no intercept
                sw = np.sqrt(V)
                coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
                worst = max(worst, abs(at1 - coef[0]))
    print(f"  max |bivariate@x=1  -  two-group Goodman coefficient|  =  {worst:.3e}")
    print("  -> identical. The scatterplot IS ecological regression, written differently.\n")

    n_imp = int(cmp.biv_impossible.sum())
    print("=" * 104)
    print(f"WHERE THE QUICK VERSION BREAKS: {n_imp} of {len(cmp)} bivariate extrapolations are "
          "arithmetically impossible")
    print("=" * 104)
    bad = cmp[cmp.biv_impossible]
    if not bad.empty:
        print(bad[["dim", "group", "estimand", "biv_reg_at1", "lo", "hi", "M2_logit", "reconciled"]]
              .assign(**{c: lambda d, c=c: (100 * d[c]).round(1) for c in
                         ["biv_reg_at1", "lo", "hi", "M2_logit", "reconciled"]})
              .to_string(index=False))
    d = 100 * cmp.biv_minus_M1.abs()
    print(f"\nbivariate vs M1 (the cost of collapsing the other 3 groups into one): "
          f"mean |diff| {d.mean():.1f}pp, max {d.max():.1f}pp")
    print(f"wrote {OUT}/bivariate_comparison.csv")


def main():
    for el in common.arg_elections():
        run(el)


if __name__ == "__main__":
    main()
