"""
Support among those who actually voted, i.e. rescaled by each group's turnout.

THE ESTIMAND
Everything else in this project uses registration as the denominator, so "Chinese PH 52.5%"
means 52.5% of *all registered* Chinese cast a PH ballot. The conventional vote share instead
asks: of the Chinese who turned out, what share went PH? Since the registered denominator is
common to both, it cancels and the answer is just the quotient:

    share_among_voters(g, p) = support(g, p) / turnout(g)        e.g. 52.5 / 62.3 = 84.3%

That really is simple division and there is nothing subtle about the point estimate.

Two facts make the quotient well-behaved here, and both come free from the raking in 03:
  - Party votes are one of the turnout choice columns, so the numerator is a strict subset of
    the denominator inside every group of every saluran. The ratio therefore cannot exceed
    100%, which a ratio of two separately-fitted models could.
  - Both come from the same reconciled table, so they are mutually consistent by construction.

THE ERROR IS THE ONLY SUBTLE PART
The naive move is to take the +-3.2pp on the numerator and the +-4.8pp on the denominator and
push them through the delta method. That assumes the two errors are independent. They are not,
and strongly not: PH votes are a subset of turnout, so a bootstrap draw that pushes Chinese
turnout up pushes Chinese PH up with it. Most of the error is common and cancels in the
quotient. Assuming independence would report an interval materially wider than the truth.

So the ratio is recomputed *inside* each replicate, before anything is summarised:
  - sampling error      -> ratio within each of the 200 bootstrap draws, then take the sd
  - identification error -> ratio within each simulated world-B run, against the ratio of the
                            planted truth, then take the rmse of the difference
and the two combine in quadrature exactly as in 07. This lets the correlation cancel on its
own rather than being assumed away in either direction.

Outputs scaled_estimates.csv and prints both tables.
"""

import os

import numpy as np
import pandas as pd

import common
from common import DIMS, ELECTIONS, ORDER, REPO

OUT = common.OUT
PARTIES = ["BN", "PH", "PN"]
STAR_ERR = 0.05


def ratio_errors(el, dim, group, party):
    """(sampling se, identification rmse) for party/turnout, both taken per replicate."""
    d = common.outdir(el)

    draws = pd.read_parquet(f"{d}/bootstrap_draws.parquet")
    q = draws[(draws.dim == dim) & (draws.group == group)]
    piv = q.pivot_table(index="draw", columns="estimand", values="value")
    r = piv[party] / piv["turnout"].where(piv["turnout"] > 0)
    se = float(r.std())

    sims = pd.read_csv(f"{d}/validation_sims.csv")
    s = sims[(sims.world == "B") & (sims.dim == dim) & (sims.group == group)]
    est = s.pivot_table(index="sim", columns="estimand", values="reconciled")
    tru = s.pivot_table(index="sim", columns="estimand", values="truth")
    er = est[party] / est["turnout"].where(est["turnout"] > 0)
    tr = tru[party] / tru["turnout"].where(tru["turnout"] > 0)
    rmse = float(np.sqrt(((er - tr) ** 2).mean()))
    return se, rmse


def build():
    base = pd.read_csv(f"{OUT}/timeseries_estimates.csv")
    rows = []
    for el in ORDER:
        for dim, gs in DIMS.items():
            for g in gs:
                sub = base[(base.election == el) & (base.dim == dim) & (base.group == g)]
                t = sub[sub.estimand == "turnout"].iloc[0]
                for p in PARTIES:
                    v = sub[sub.estimand == p].iloc[0]
                    est = v.reconciled / t.reconciled if t.reconciled > 0 else np.nan
                    se, rmse = ratio_errors(el, dim, g, p)
                    err = 1.96 * float(np.sqrt(se ** 2 + rmse ** 2))
                    rows.append(dict(
                        election=el, short=ELECTIONS[el]["short"], dim=dim, group=g, party=p,
                        share_of_registered=v.reconciled, turnout=t.reconciled,
                        share_of_voters=min(est, 1.0), error=err,
                        lo=max(est - err, 0.0), hi=min(est + err, 1.0),
                        se_sampling=se, rmse_ident=rmse,
                        # The ratio's error is recomputed per replicate (above), so the error
                        # shared by numerator and denominator has already cancelled; parent
                        # stars are therefore NOT inherited -- that would double-count. The
                        # bar is applied in the units displayed, as everywhere else.
                        cite=bool(err <= STAR_ERR),
                    ))
    return pd.DataFrame(rows)


def table(df, dim, value, errcol, gs, cols, header):
    out = [header, "|---|---|" + "---|" * len(cols)]
    for g in gs:
        for k, el in enumerate(ORDER):
            cells = []
            for c in cols:
                if value == "reg":
                    r = df[(df.dim == dim) & (df.group == g) & (df.estimand == c) &
                           (df.election == el)].iloc[0]
                    v, e, ok = r.reconciled, r.error, r.cite
                else:
                    r = df[(df.dim == dim) & (df.group == g) & (df.party == c) &
                           (df.election == el)].iloc[0]
                    v, e, ok = r.share_of_voters, r.error, r.cite
                cells.append(f"{100 * v:.1f} ±{100 * e:.1f}{'' if ok else '*'}")
            name = f"**{g}**" if k == 0 else ""
            out.append(f"| {name} | {ELECTIONS[el]['short']} | " + " | ".join(cells) + " |")
        out.append("| | | | | |" + ("|" if len(cols) > 3 else ""))
    out.pop()
    return "\n".join(out)


def main():
    df = build()
    df.to_csv(f"{OUT}/scaled_estimates.csv", index=False)
    base = pd.read_csv(f"{OUT}/timeseries_estimates.csv")

    print("=" * 96)
    print("TABLE 2 -- SUPPORT AMONG THOSE WHO VOTED  (party / turnout, both per replicate)")
    print("=" * 96)
    for dim, gs in DIMS.items():
        print(f"\n### by {dim}\n")
        print(f"{'group':9s} {'election':9s} " + " ".join(f"{p:>15s}" for p in PARTIES))
        for g in gs:
            for el in ORDER:
                cs = []
                for p in PARTIES:
                    r = df[(df.dim == dim) & (df.group == g) & (df.party == p) &
                           (df.election == el)].iloc[0]
                    cs.append(f"{100 * r.share_of_voters:5.1f}±{100 * r.error:<4.1f}"
                              f"{'' if r.cite else '*'}")
                print(f"{g:9s} {ELECTIONS[el]['short']:9s} " + " ".join(f"{c:>15s}" for c in cs))
            print()

    # Does taking the ratio per replicate actually beat the delta method? Compare like with
    # like: both must price the same two error sources, so the naive version uses each
    # estimand's TOTAL se (sampling + identification), not just its sampling se.
    print("\nDOES THE PER-REPLICATE RATIO ACTUALLY BEAT THE DELTA METHOD?")
    print("delta method on total se (assumes numerator/denominator errors independent)")
    print("vs per-replicate (lets whatever correlation exists cancel on its own):\n")
    print(f"{'group':9s} {'party':6s} {'el':6s} {'ratio':>7s} {'delta':>7s} {'per-rep':>8s} "
          f"{'ratio of errors':>16s}")
    comp = []
    for dim, gs in DIMS.items():
        for g in gs:
            for el in ORDER:
                for p in PARTIES:
                    r = df[(df.dim == dim) & (df.group == g) & (df.party == p) &
                           (df.election == el)].iloc[0]
                    b = base[(base.election == el) & (base.dim == dim) & (base.group == g)]
                    sn = b[b.estimand == p].iloc[0].total_se
                    sd = b[b.estimand == "turnout"].iloc[0].total_se
                    n, d = r.share_of_registered, r.turnout
                    if n <= 1e-6 or d <= 1e-6:
                        continue
                    delta = 1.96 * abs(n / d) * np.sqrt((sn / n) ** 2 + (sd / d) ** 2)
                    comp.append(dict(dim=dim, group=g, party=p, el=el, delta=delta,
                                     actual=r.error, k=delta / max(r.error, 1e-9)))
                    if dim == "ethnicity" and g in ("Malay", "Chinese") and p in ("BN", "PH"):
                        print(f"{g:9s} {p:6s} {ELECTIONS[el]['short']:6s} "
                              f"{100 * r.share_of_voters:6.1f}% {100 * delta:6.1f} "
                              f"{100 * r.error:7.1f} {delta / max(r.error, 1e-9):15.2f}x")
    c = pd.DataFrame(comp)
    print(f"\nover all {len(c)} cells: delta/per-replicate median {c.k.median():.2f}x, "
          f"mean {c.k.mean():.2f}x, min {c.k.min():.2f}x, max {c.k.max():.2f}x")
    print(f"delta method is WIDER than per-replicate in {100 * (c.k > 1).mean():.0f}% of cells, "
          f"NARROWER in {100 * (c.k < 1).mean():.0f}%")

    with open(f"{OUT}/TABLES.md", "w") as f:
        f.write(md(df, base))
    print(f"\nwrote {OUT}/scaled_estimates.csv, TABLES.md")


def md(df, base):
    hdr1 = "| group | election | turnout | BN | PH | PN |"
    hdr2 = "| group | election | BN | PH | PN |"
    s = ["# Johor — two views of the same estimates\n",
         "> **Superseded — pipeline diagnostic, not the paper's estimates.** These tables come",
         "> from the separate-dimension model with simulation-based errors, an intermediate",
         "> stage the paper's validation test rejects in favour of the joint 24-cell model",
         "> (`joint_*.csv`, `validation_errors.csv`). Cite the paper's tables, not these.\n",
         "`*` = do not cite (95% total error > 5pp). All errors combine sampling and",
         "identification error in quadrature.\n",
         "## Table 1 — share of the group's REGISTERED voters\n",
         "Not voting is one of the choices being measured, so these four columns are shares of",
         "the whole group and `turnout ≈ BN + PH + PN + OTHER + REJECTED`.\n"]
    for dim, gs in DIMS.items():
        s.append(f"\n### By {dim}\n")
        s.append(table(base, dim, "reg", "error", gs, ["turnout", "BN", "PH", "PN"], hdr1))
    s += ["\n## Table 2 — share of the group's VOTERS (rescaled by turnout)\n",
          "Each cell is Table 1's party number divided by Table 1's turnout for the same group",
          "and election — e.g. Chinese PH at SE-16 = 52.5 / 62.3 = 84.3%. This is the",
          "conventional vote share: of those who showed up, who did they back.\n",
          "The errors are *not* Table 1's errors divided through. Numerator and denominator",
          "move together (party votes are a subset of turnout), so the ratio is recomputed",
          "inside every bootstrap draw and every simulation, letting the shared error cancel.",
          "The result is tighter than naive propagation would give.\n"]
    for dim, gs in DIMS.items():
        s.append(f"\n### By {dim}\n")
        s.append(table(df, dim, "vot", "error", gs, PARTIES, hdr2))
    return "\n".join(s)


if __name__ == "__main__":
    main()
