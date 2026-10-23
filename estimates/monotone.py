"""Monotone ecological inference (Elzayn, Goldin, Guage, Ho & Morton 2026) on the headline
group rates: sharp bounds under sign restrictions on the two conditional associations.
Usage: python3 monotone.py

WHAT IT IS. For a binary group indicator X (member of group g or not) and binary outcome Y
(did c or not), with saluran n carrying shares (X_n, Y_n) and population weight P_n, the paper
shows that the two canonical point estimators bracket the truth once the SIGNS of two
associations are known:

    delta_B = E[Cov(Y, X | N)]     between-group: do members differ from non-members INSIDE
                                    a saluran?  (Prop. 2: NM is biased by -delta_B / Var X)
    delta_W = E[Cov(Y, X_N | X)]   within-group: does a member's rate move with the saluran's
                                    composition?  (Prop. 1: ER is biased by +delta_W / Var X_N)

Their Tables 2-3 give the sharp identified set for the group mean under every combination of
sign assumptions, as intersections of the Duncan-Davis (method-of-bounds, MOB) interval
with the ecological-regression (ER, Goodman with intercept) and neighbourhood-model (NM,
composition-weighted saluran mean) points. Under CONTEXTUAL REINFORCEMENT -- delta_B and
delta_W share a sign -- the set is [NM, min(ER, MOB+)] or its mirror, with the shared sign
read off sign(ER - NM) (Prop. 6, Cor. 5).

WHY IT BELONGS HERE. It is the intermediate rung of the claim ladder: strictly weaker than the
constancy assumption behind the point model (constancy implies delta_W = 0 exactly), strictly
stronger than the margins alone. And unlike most EI settings, the assumptions are partly
CHECKABLE here: the split-sample and near-pure holdouts of robustness.py measure how a group's
rate moves with composition (the sign of delta_W), and the direct joint-cell holdouts observe
within-stream group differences (delta_B).

IMPLEMENTATION NOTES.
  * All moments are population moments over individuals, i.e. saluran quantities weighted by
    electorate V_n (the paper's P_n); the margins are a census, so there is no sampling step.
  * Each group is treated as binary against the rest, which is the paper's own setup; the
    supplementary R x C extension is not needed for a single group's rate.
  * MOB endpoints reproduce contrasts.py's Frechet bounds exactly (asserted).
  * ER here is the two-group Goodman regression WITH intercept -- algebraically the
    'bivariate scatterplot' row of the manuscript's Table of specifications.
  * Between-election changes are bounded by the difference of the two elections' extremes,
    since the interiors are unlinked.

Outputs: output/monotone_bounds.csv, tex/tab_monotone.tex, and a printed summary.
"""

import numpy as np
import pandas as pd

import common

OUT, TEX = common.OUT, f"{common.REPO}/tex"
ELS = ["ge15", "jhr_se16"]
CLAIMS = [("Malay", "BN"), ("Malay", "PN"), ("Malay", "turnout"),
          ("Chinese", "PH"), ("Chinese", "turnout"), ("Chinese", "BN")]
# Substantive sign of delta_B for each claim: is the group's rate above everyone else's INSIDE
# a saluran? These are the paper's Malaysian priors, stated as assumptions and not estimated:
# Malays vote BN and PN more, and turn out more, than their non-Malay neighbours; Chinese
# voters vote PH more, BN less, and turn out less, than theirs.
DB_SIGN = {("Malay", "BN"): "+", ("Malay", "PN"): "+", ("Malay", "turnout"): "+",
           ("Chinese", "PH"): "+", ("Chinese", "turnout"): "-", ("Chinese", "BN"): "-"}
DW_MIN_GAP = 0.01   # split-sample own-share gap below 1pp is read as no evidence on sign(delta_W)

CELLS = {  # (dW, dB) -> label; Table 2 of the paper
    ("?", "?"): "MOB", ("?", "+"): "dB>=0", ("?", "-"): "dB<=0",
    ("+", "?"): "dW>=0", ("+", "+"): "dW>=0 & dB>=0", ("+", "-"): "dW>=0 & dB<=0",
    ("-", "?"): "dW<=0", ("-", "+"): "dW<=0 & dB>=0", ("-", "-"): "dW<=0 & dB<=0",
}
ASSUMPTIONS = list(CELLS.values()) + ["reinforcement", "preferred", "election-specific"]


def moments(df, el, group, est):
    V = df.V.to_numpy(float)
    P = V / V.sum()
    X = df[f"n_{group}"].to_numpy(float) / V                 # X_n
    Y = common.outcome(df, el, est) / V                       # Y_n
    EX, EY = P @ X, P @ Y
    varX = EX * (1 - EX)                                      # binary X
    varXN = P @ (X - EX) ** 2
    cov = P @ ((X - EX) * (Y - EY))
    gamma = varXN / varX
    D_ER = cov / varXN
    Y1_ER = EY + D_ER * (1 - EX)
    Y0_ER = EY - D_ER * EX
    Y1_NM = (P @ (X * Y)) / EX
    Y0_NM = (P @ ((1 - X) * Y)) / (1 - EX)
    Y1_lo = (P @ np.maximum(Y - (1 - X), 0)) / EX
    Y1_hi = (P @ np.minimum(Y, X)) / EX
    Y0_lo = (P @ np.maximum(Y - X, 0)) / (1 - EX)
    Y0_hi = (P @ np.minimum(Y, 1 - X)) / (1 - EX)
    return dict(EX=EX, EY=EY, gamma=gamma, D_ER=D_ER, D_NM=Y1_NM - Y0_NM,
                Y1_ER=Y1_ER, Y0_ER=Y0_ER, Y1_NM=Y1_NM, Y0_NM=Y0_NM,
                Y1_lo=Y1_lo, Y1_hi=Y1_hi, Y0_lo=Y0_lo, Y0_hi=Y0_hi)


def table2(m, dW, dB):
    """Identified set for the group-1 mean, Table 2 of the paper, for signs in {'?','+','-'}."""
    lo, hi, er, nm = m["Y1_lo"], m["Y1_hi"], m["Y1_ER"], m["Y1_NM"]
    L = {"?": lo, "+": nm, "-": lo}[dB] if dW == "?" else None
    if dW == "?":
        a, b = {"?": (lo, hi), "+": (nm, hi), "-": (lo, nm)}[dB]
    elif dW == "+":
        a, b = {"?": (lo, min(er, hi)), "+": (nm, min(er, hi)), "-": (lo, min(nm, er))}[dB]
    else:
        a, b = {"?": (max(lo, er), hi), "+": (max(er, nm), hi), "-": (max(er, lo), nm)}[dB]
    return max(a, lo), min(b, hi)          # inside the MOB by construction


def reinforcement(m):
    """Prop. 6 / Cor. 5: delta_B and delta_W share a sign, read from sign(D_ER) = sign(D_NM)."""
    return table2(m, "+", "+") if m["D_ER"] >= 0 else table2(m, "-", "-")


def election_dW_sign(split, el, group, est):
    """Same diagnostic, but read separately for one election."""
    s = split[(split.election == el) & (split.group == group) & (split.estimand == est)]
    gap = float(s.gap.iloc[0]) / 100 * (1 if group == "Malay" else -1)
    return "?" if abs(gap) < DW_MIN_GAP else ("+" if gap > 0 else "-"), gap


def empirical_dW_sign(split, group, est):
    """Sign of delta_W as the split-sample check of robustness.py sees it: the group's own
    rate in the half of mixed streams where its share is high, minus the half where it is
    low (for Chinese, the Malay-share partition runs the other way). Averaged over the three
    elections; gaps under DW_MIN_GAP count as no evidence."""
    s = split[(split.group == group) & (split.estimand == est)]
    gap = s.gap.mean() / 100 * (1 if group == "Malay" else -1)
    return "?" if abs(gap) < DW_MIN_GAP else ("+" if gap > 0 else "-"), gap


def group_mean_bounds(m, dW_emp, dB_sub, dW_el):
    out = {label: table2(m, dW, dB) for (dW, dB), label in CELLS.items()}
    out["reinforcement"] = reinforcement(m)
    out["preferred"] = table2(m, dW_emp, dB_sub)          # common sign over elections
    out["election-specific"] = table2(m, dW_el, dB_sub)   # this election's own diagnostic sign
    return out


def main():
    data = {el: common.load(el) for el in ELS}
    marg = pd.read_csv(f"{OUT}/joint_marginals.csv").set_index(["election", "group", "estimand"]).rate
    fre = pd.read_csv(f"{OUT}/contrast_bounds.csv")
    split = pd.read_csv(f"{OUT}/robustness_split.csv")
    rows = []
    for g, e in CLAIMS:
        M = {el: moments(data[el], el, g, e) for el in ELS}
        dW_emp, dW_gap = empirical_dW_sign(split, g, e)
        dB_sub = DB_SIGN[(g, e)]
        dW_el = {el: election_dW_sign(split, el, g, e) for el in ELS}
        B = {el: group_mean_bounds(M[el], dW_emp, dB_sub, dW_el[el][0]) for el in ELS}
        # check MOB == contrasts.py Frechet level bounds
        key = f"{g} {e}"
        f = fre[(fre.claim == key) & (fre.territory == "statewide")]
        if len(f):
            f = f.iloc[0]
            assert abs(B["ge15"]["MOB"][0] - f.ge15_lo) < 1e-9 and abs(B["jhr_se16"]["MOB"][1] - f.se16_hi) < 1e-9
        for a in ASSUMPTIONS:
            lo15, hi15 = B["ge15"][a]
            lo16, hi16 = B["jhr_se16"][a]
            refuted = (lo15 > hi15 + 1e-12) or (lo16 > hi16 + 1e-12)
            rows.append(dict(group=g, estimand=e, assumption=a, refuted=refuted,
                             dW_empirical=dW_emp, dW_split_gap=dW_gap, dB_assumed=dB_sub,
                             dW_ge15=dW_el["ge15"][0], dW_se16=dW_el["jhr_se16"][0],
                             cor5_consistent=np.sign(M["ge15"]["D_ER"]) == np.sign(M["ge15"]["D_NM"])
                             and np.sign(M["jhr_se16"]["D_ER"]) == np.sign(M["jhr_se16"]["D_NM"]),
                             ge15_lo=lo15, ge15_hi=hi15, se16_lo=lo16, se16_hi=hi16,
                             change_lo=np.nan if refuted else lo16 - hi15,
                             change_hi=np.nan if refuted else hi16 - lo15,
                             point_ge15=marg[("ge15", g, e)], point_se16=marg[("jhr_se16", g, e)],
                             point_change=marg[("jhr_se16", g, e)] - marg[("ge15", g, e)],
                             gamma_ge15=M["ge15"]["gamma"], gamma_se16=M["jhr_se16"]["gamma"],
                             D_ER_ge15=M["ge15"]["D_ER"], D_NM_ge15=M["ge15"]["D_NM"],
                             D_ER_se16=M["jhr_se16"]["D_ER"], D_NM_se16=M["jhr_se16"]["D_NM"],
                             ER_ge15=M["ge15"]["Y1_ER"], NM_ge15=M["ge15"]["Y1_NM"],
                             ER_se16=M["jhr_se16"]["Y1_ER"], NM_se16=M["jhr_se16"]["Y1_NM"]))
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/monotone_bounds.csv", index=False)

    # point model inside each set?
    R["point_inside_ge15"] = (R.point_ge15 >= R.ge15_lo - 1e-9) & (R.point_ge15 <= R.ge15_hi + 1e-9)
    R["point_inside_se16"] = (R.point_se16 >= R.se16_lo - 1e-9) & (R.point_se16 <= R.se16_hi + 1e-9)
    R["change_signed"] = ((R.change_lo > 0) | (R.change_hi < 0)) & ~R.refuted

    pd.set_option("display.width", 200)
    print("Assumption checks: empirical sign of delta_W (split-sample) and Corollary-5 consistency")
    print(R.drop_duplicates(["group", "estimand"])[["group", "estimand", "dB_assumed", "dW_empirical", "dW_split_gap", "cor5_consistent"]].round(3).to_string(index=False))
    print("\ngamma (group aggregation ratio):")
    print(R.drop_duplicates(["group", "estimand"])[["group", "estimand", "gamma_ge15", "gamma_se16"]].round(3).to_string(index=False))
    print("\nER vs NM levels (pp), and our point estimate:")
    d = R.drop_duplicates(["group", "estimand"])
    print((d[["group", "estimand", "ER_ge15", "NM_ge15", "point_ge15", "ER_se16", "NM_se16", "point_se16"]]
           .assign(**{c: lambda x, c=c: (100 * x[c]).round(1) for c in ["ER_ge15", "NM_ge15", "point_ge15", "ER_se16", "NM_se16", "point_se16"]})
           .to_string(index=False)))
    print("\nBounds on the GE-15 -> SE-16 change (pp) by assumption:")
    p = R.pivot_table(index=["group", "estimand"], columns="assumption", values=["change_lo", "change_hi"]) * 100
    for a in ASSUMPTIONS:
        print(f"\n  {a}")
        for (g, e), r in R[R.assumption == a].set_index(["group", "estimand"]).iterrows():
            if r.refuted:
                print(f"    {g:8s} {e:8s}  REFUTED (empty identified set)"); continue
            print(f"    {g:8s} {e:8s}  change in [{100*r.change_lo:6.1f}, {100*r.change_hi:6.1f}]  "
                  f"signed={'yes' if r.change_signed else 'no ':3s}  point {100*r.point_change:+.1f}  "
                  f"inside(GE15,SE16)=({r.point_inside_ge15},{r.point_inside_se16})")

    # ---- the table
    def lab(a):
        return (a.replace("dB>=0", "$\\delta_B \\ge 0$").replace("dB<=0", "$\\delta_B \\le 0$")
                 .replace("dW>=0", "$\\delta_W \\ge 0$").replace("dW<=0", "$\\delta_W \\le 0$")
                 .replace(" & ", ", "))
    blocks = []
    for g, e in CLAIMS:
        sub = R[(R.group == g) & (R.estimand == e)].set_index("assumption")
        pt = sub.iloc[0]
        dB, dW = pt.dB_assumed, pt.dW_empirical
        show = ["MOB", CELLS[("?", dB)], CELLS[(dW, "?")] if dW != "?" else None,
                CELLS[(dW, dB)], "election-specific", "reinforcement"]
        show = [a for a in show if a]
        seen = []
        blocks.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{g} {e}}} --- point model: {100*pt.point_ge15:.1f}, "
                      f"{100*pt.point_se16:.1f}, change {100*pt.point_change:+.1f}; assumed $\\delta_B {'\\ge' if dB == '+' else '\\le'} 0$, "
                      f"diagnostic $\\delta_W$ (common sign): {'$\\ge 0$' if dW == '+' else ('$\\le 0$' if dW == '-' else 'no evidence')}}}\\\\[1pt]")
        for a in show:
            if a in seen:
                continue
            seen.append(a)
            r = sub.loc[a]
            name = "Margins only (MOB)" if a == "MOB" else ("Contextual reinforcement" if a == "reinforcement" else lab(a))
            if a == CELLS[(dW, dB)] and a != "MOB":
                name += " (common sign)"
            if a == "election-specific":
                sg = {"+": "$\\ge 0$", "-": "$\\le 0$", "?": "?"}
                name = f"per-election diagnostic ($\\delta_W$: {sg[pt.dW_ge15]}, {sg[pt.dW_se16]})"
            if r.refuted:
                blocks.append(f"\\quad {name} & \\multicolumn{{4}}{{c}}{{refuted: identified set empty}} \\\\")
                continue
            flag = "" if (r.point_inside_ge15 and r.point_inside_se16) else "$^{\\dagger}$"
            blocks.append(f"\\quad {name} & [{100*r.ge15_lo:.1f}, {100*r.ge15_hi:.1f}] & "
                          f"[{100*r.se16_lo:.1f}, {100*r.se16_hi:.1f}] & "
                          f"[{100*r.change_lo:+.1f}, {100*r.change_hi:+.1f}] & "
                          f"{'yes' if r.change_signed else 'no'}{flag} \\\\")
        blocks.append("\\addlinespace[3pt]")
    body = "\n".join(blocks)
    out = f"""\\begin{{table}}[!htb]
\\caption{{Monotone-EI bounds on the headline rates, as shares of the group's electorate, in percentage points}}
\\label{{tab:monotone}}
\\centering
\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l ccc c}}
\\toprule
Assumption & GE-15 & SE-16 & Change & Signed? \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}\\par\\vspace{{3pt}}
{{\\footnotesize Sharp identified sets of Elzayn et al.\\cite{{elzayn2026}}, Table~2, statewide. $\\delta_B$: within-saluran difference between the group and everyone else; $\\delta_W$: movement of the group's own rate with the group's share of the saluran. The sign of $\\delta_B$ is assumed, not observed. The `diagnostic' sign of $\\delta_W$ is read from the split-sample check of Section~\\ref{{sec:strategy-transport}}, which is itself model-based, either as one common sign over elections or election by election; both are conditional assumptions, not measurements. Contextual reinforcement: $\\delta_B$ and $\\delta_W$ share a sign, read from the ER--NM ordering. `Refuted': the two sign restrictions are jointly inconsistent with the margins. $^{{\\dagger}}$: the point model lies outside the set at one or both elections.}}
\\end{{table}}
"""
    with open(f"{TEX}/tab_monotone.tex", "w") as f:
        f.write(out)
    print(f"\nwrote {OUT}/monotone_bounds.csv, {TEX}/tab_monotone.tex")


if __name__ == "__main__":
    main()
