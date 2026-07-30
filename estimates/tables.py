"""Generate the manuscript's two data tables from the pipeline outputs.
Usage: python3 tables.py

Writes tex/tab_results.tex (marginals, both scales, all three elections),
tex/tab_crosstab.tex (the SE-16 joint ethnicity-by-age crosstab) and tex/tab_errors.tex
(the reported error bar for every group x election x estimand). Also prints every
between-election movement the manuscript claims against its conservative combined error. Everything is derived
from out/joint_marginals.csv, out/joint_scaled.csv, out/joint_final.csv and the measured
error bars in out/validation_errors.csv, so the tables can never drift from the estimates.

Error bars are the held-out validation errors of errors.py -- measured, and specific to
one group, one election AND one estimand, so Malay BN support carries the error of Malay BN
support rather than an average over four unlike quantities. They are not confidence intervals.
Groups with no near-pure saluran (Indian, Other) could not be validated, so they carry no
error bar: absence of a bar means "we could not check this", not "this is imprecise".

Shading samples matplotlib's Blues colormap (seaborn's Blues) over its full range on a
common 0-1 scale; text flips to white on dark fills, as seaborn heatmaps do. Unvalidated
groups (Indian, Other) carry no error bar, and the affected tables carry a note under the
title that point estimates without error bars are not citable.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

BLUES = matplotlib.colormaps["Blues"]

NOTE = "{\\footnotesize (point estimates without error bars are not citable)}\\par\\vspace{3pt}"


def shade(rate, text):
    """A cell filled from the full Blues colormap, flipping to white text on dark fills.
    Only validated cells are shaded: the fill itself encodes "carries an error bar", so
    unvalidated point estimates sit on plain white and draw no emphasis."""
    r, g, b, _ = BLUES(max(0.0, min(1.0, float(rate))))
    hexc = f"{round(255*r):02X}{round(255*g):02X}{round(255*b):02X}"
    if rate >= 0.5:                 # white text = majority, a semantic cue not a luminance one
        text = f"\\textcolor{{white}}{{{text}}}"
    return f"\\cellcolor[HTML]{{{hexc}}}{text}"

TEX = os.path.join(common.REPO, "tex")
OUT = common.OUT
ELS = [("jhr_se15", "SE-15"), ("ge15", "GE-15"), ("jhr_se16", "SE-16")]
GROUPS = common.ETH + common.AGE

VERR = (pd.read_csv(f"{common.OUT}/validation_errors.csv")
          .set_index(["election", "group", "estimand"]).err.to_dict())


def cell(rate, el, grp, est):
    """A rate with the error bar measured for that same estimand, or bare if unvalidated."""
    err = VERR.get((el, grp, est))
    if err is None:
        return f"{100 * rate:.1f}"
    # Never print +/-0.0: an error too small to round to a tenth is still not zero.
    bar = f"{{\\tiny$\\pm$}}{100 * err:.1f}" if 100 * err >= 0.05 else "{\\tiny$\\pm<$}0.1"
    return shade(rate, f"{100 * rate:.1f}\\,{bar}")


def results_table():
    marg = pd.read_csv(f"{common.OUT}/joint_marginals.csv")
    scaled = pd.read_csv(f"{common.OUT}/joint_scaled.csv")

    def m(el, grp, est):
        r = marg[(marg.election == el) & (marg.group == grp) & (marg.estimand == est)].iloc[0]
        return cell(r.rate, el, grp, est)

    def s(el, grp, party):
        r = scaled[(scaled.election == el) & (scaled.group == grp)
                   & (scaled.party == party)].iloc[0]
        return cell(r.share_of_voters, el, grp, f"{party}_v")

    lines = []
    for gi, grp in enumerate(GROUPS):
        if gi in (0, len(common.ETH)):
            lines.append("\\midrule" if gi else "")
        label = grp.replace("-", "--")
        for j, (el, short) in enumerate(ELS):
            head = f"\\multirow{{3}}{{*}}{{\\textbf{{{label}}}}}" if j == 0 else ""
            row = [m(el, grp, e) for e in ["turnout", "BN", "PH", "PN"]] + \
                  [s(el, grp, p) for p in ["BN", "PH", "PN"]]
            lines.append(f"{head} & {short} & " + " & ".join(row) + " \\\\")
        lines.append("\\addlinespace[2pt]")

    caption = ("Turnout and coalition support in Johor, 2022--2026, by ethnicity and age")
    body = "\n".join(lines)
    out = f"""\\begin{{table}}[!htb]
\\caption{{{caption}}}
\\label{{tab:results}}
\\centering{NOTE}
\\footnotesize\\setlength{{\\tabcolsep}}{{3.0pt}}
\\begin{{tabular}}{{ll cccc @{{\\hspace{{6pt}}}} ccc}}
\\toprule
 & & \\multicolumn{{4}}{{c}}{{\\% of the group's \\textbf{{electorate}}}} & \\multicolumn{{3}}{{c}}{{\\% of the group's \\textbf{{ballots}}}} \\\\
\\cmidrule(lr){{3-6}}\\cmidrule(lr){{7-9}}
 & & Turnout & BN & PH & PN & BN & PH & PN \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(f"{TEX}/tab_results.tex", "w") as f:
        f.write(out)


def crosstab_table():
    jf = pd.read_csv(f"{common.OUT}/joint_final.csv")
    jf = jf[jf.election == "jhr_se16"]

    def c(eth, age, est):
        r = jf[(jf.eth == eth) & (jf.age == age) & (jf.estimand == est)].iloc[0]
        # A joint cell has no pure stratum of its own, so we carry the LARGER of its two
        # marginal validation errors FOR THIS ESTIMAND -- a conservative proxy, stated as such.
        errs = [VERR.get(("jhr_se16", eth, est)), VERR.get(("jhr_se16", age, est))]
        if any(e is None for e in errs):
            return f"{100 * r.rate:.1f}"
        return shade(r.rate, f"{100 * r.rate:.1f}\\,{{\\tiny$\\pm$}}{100 * max(errs):.1f}")

    blocks = []
    for est, name in [("turnout", "Turnout"), ("BN", "BN"), ("PH", "PH"), ("PN", "PN")]:
        # every panel is on ONE scale -- the cell's registered electorate -- unlike
        # tab_malay, whose party panels are ballot shares; the label says so
        label = f"{name} (\\% of cell's electorate)"
        blocks.append(f"\\multicolumn{{{len(common.AGE) + 1}}}{{l}}{{\\textit{{{label}}}}}\\\\[1pt]")
        for eth in common.ETH:
            row = " & ".join(c(eth, a, est) for a in common.AGE)
            blocks.append(f"\\quad {eth} & {row} \\\\")
        blocks.append("\\addlinespace[3pt]")

    caption = ("The joint ethnicity-by-age crosstab, SE-16")
    head = " & " + " & ".join(a.replace("-", "--") for a in common.AGE) + " \\\\"
    body = "\n".join(blocks)
    out = f"""\\begin{{table}}[!htb]
\\caption{{{caption}}}
\\label{{tab:joint}}
\\centering{NOTE}
\\footnotesize\\setlength{{\\tabcolsep}}{{3.2pt}}
\\begin{{tabular}}{{l{'c' * len(common.AGE)}}}
\\toprule
{head}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(f"{TEX}/tab_crosstab.tex", "w") as f:
        f.write(out)


COLS = ["turnout", "BN", "PH", "PN", "BN_v", "PH_v", "PN_v"]
HEAD = ["Turnout", "BN", "PH", "PN", "BN", "PH", "PN"]
S = pd.read_csv(f"{OUT}/validation_errors.csv")
E = S.set_index(["election", "group", "estimand"])


def errors_table():
    lines = []
    groups = [g for g in common.ETH + common.AGE if g in set(S.group)]
    for gi, grp in enumerate(groups):
        if gi and grp in common.AGE and groups[gi - 1] in common.ETH:
            lines.append("\\midrule")
        for j, (el, short) in enumerate(ELS):
            head = f"\\multirow{{3}}{{*}}{{\\textbf{{{grp.replace('-', '--')}}}}}" if not j else ""
            row = []
            for c in COLS:
                v = E.err.get((el, grp, c))
                if v is None or not np.isfinite(v):
                    row.append("---")
                else:                       # never print 0.00; the measurement is not exact
                    row.append(f"{100 * v:.2f}" if 100 * v >= 0.005 else "$<$0.01")
            lines.append(f"{head} & {short} & " + " & ".join(row) + " \\\\")
        lines.append("\\addlinespace[2pt]")

    caption = ("Held-out validation error by group, election and estimand, in percentage points")
    body = "\n".join(lines)
    out = f"""\\begin{{table}}[!htb]
\\caption{{{caption}}}
\\label{{tab:errors}}
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{ll cccc @{{\\hspace{{6pt}}}} ccc}}
\\toprule
 & & \\multicolumn{{4}}{{c}}{{\\% of the group's \\textbf{{electorate}}}} & \\multicolumn{{3}}{{c}}{{\\% of the group's \\textbf{{ballots}}}} \\\\
\\cmidrule(lr){{3-6}}\\cmidrule(lr){{7-9}}
 & & {' & '.join(HEAD)} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(f"{TEX}/tab_errors.tex", "w") as f:
        f.write(out)


def bounds_table():
    """The statewide accounting bounds for every group: the assumption-free core.

    The Duncan--Davis bounds are the margins restated, so they hold with certainty and are
    the one thing any analysis of these elections must agree on. Bounds only, no point
    estimates: the table shows what the arithmetic settles before any modelling starts.
    """
    lines = []
    for gi, grp in enumerate(GROUPS):
        if gi == len(common.ETH):
            lines.append("\\midrule")
        dim = "ethnicity" if grp in common.ETH else "age"
        label = grp.replace("-", "--")
        for j, (el, short) in enumerate(ELS):
            b = pd.read_csv(f"{OUT}/{el}/bounds_statewide.csv")
            b = b[(b.dim == dim) & (b.group == grp)].set_index("estimand")
            head = f"\\multirow{{3}}{{*}}{{\\textbf{{{label}}}}}" if not j else ""
            cells = []
            for e in ["turnout", "BN", "PH", "PN"]:
                r = b.loc[e]
                cells.append(f"[{100 * r.lo:.1f}, {100 * r.hi:.1f}]")
            lines.append(f"{head} & {short} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")

    caption = ("The deterministic accounting bounds, statewide: every estimate must "
               "lie inside them")
    body = "\n".join(lines)
    out = f"""\\begin{{table}}[!htb]
\\caption{{{caption}}}
\\label{{tab:bounds}}
\\centering
\\footnotesize\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{ll cccc}}
\\toprule
 & & Turnout & BN & PH & PN \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(f"{TEX}/tab_bounds.tex", "w") as f:
        f.write(out)


def m1_table():
    """The raw Goodman (M1) ethnic estimates for all three elections (tab_m1).

    Printed in full so the failure of the constant-rate regression is shown, not
    asserted: cells outside the statewide deterministic bounds are set in red.
    The table doubles as a fingerprint -- a published breakdown matching these numbers
    (with impossible corners clipped to zero) came from this regression.
    """
    lines = []
    for grp in common.ETH:
        for j, (el, short) in enumerate(ELS):
            mv = pd.read_csv(f"{OUT}/{el}/model_vs_bounds.csv")
            mv = mv[(mv.model == "M1_lpm") & (mv.dim == "ethnicity")
                    & (mv.group == grp)].set_index("estimand")
            head = f"\\multirow{{3}}{{*}}{{\\textbf{{{grp}}}}}" if not j else ""

            def fmt(val, violates):
                txt = f"{100 * val:.1f}".replace("-", "$-$")
                return f"\\textcolor{{red}}{{{txt}}}" if violates else txt

            cells = [fmt(mv.loc[e].estimate, mv.loc[e].violates)
                     for e in ["turnout", "BN", "PH", "PN"]]
            # ballot scale: the same estimates divided by the group's M1 turnout --
            # the numbers informal breakdowns usually report. A dagger carries over
            # from the electorate-scale cell it is a rescaling of.
            t = mv.loc["turnout"].estimate
            cells += [fmt(mv.loc[e].estimate / t, mv.loc[e].violates)
                      for e in ["BN", "PH", "PN"]]
            lines.append(f"{head} & {short} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")

    caption = ("Goodman regression (M1) by ethnicity: the constant-rate estimates, "
               "as fitted")
    body = "\n".join(lines)
    out = f"""\\begin{{table}}[!htb]
\\caption{{{caption}}}
\\label{{tab:m1}}
\\centering
\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{ll wc{{3.4em}} *{{3}}{{wc{{2.7em}}}} @{{\\hspace{{10pt}}}} *{{3}}{{wc{{2.7em}}}}}}
\\toprule
 & & \\multicolumn{{4}}{{c}}{{\\% of the group's electorate}} & \\multicolumn{{3}}{{c}}{{\\% of the group's ballots}} \\\\
\\cmidrule(lr){{3-6}}\\cmidrule(lr){{7-9}}
 & & Turnout & BN & PH & PN & BN & PH & PN \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\par\\vspace{{3pt}}
{{\\footnotesize \\textcolor{{red}}{{red}}: outside the deterministic bounds of Table~\\ref{{tab:bounds}}}}
\\end{{table}}
"""
    with open(f"{TEX}/tab_m1.tex", "w") as f:
        f.write(out)


def malay_crosstab_table():
    """Malay turnout and coalition support by age band, all three elections (tab_malay).

    Turnout is % of registered Malays in the band; the coalition panels are % of the
    band's Malay voters (rate/turnout), the same voter scale as tab_results. Each cell
    carries the conservative max of its two marginal validation errors, propagated for
    the voter-scale panels exactly as in errors.py."""
    jf = pd.read_csv(f"{OUT}/joint_final.csv")
    jf = jf[jf.eth == "Malay"].set_index(["election", "age", "estimand"])

    def err(el, age, est):
        return max(VERR[(el, "Malay", est)], VERR[(el, age, est)])

    def cell_t(el, age):
        r = jf.loc[(el, age, "turnout")].rate
        e = err(el, age, "turnout")
        return shade(r, f"{100 * r:.1f}\\,{{\\tiny$\\pm$}}{100 * e:.1f}")

    def cell_p(el, age, party):
        t = jf.loc[(el, age, "turnout")].rate
        v = jf.loc[(el, age, party)].rate / t
        e = (err(el, age, party) + v * err(el, age, "turnout")) / t
        return shade(v, f"{100 * v:.1f}\\,{{\\tiny$\\pm$}}{100 * e:.1f}")

    blocks = []
    panels = [("turnout", "Turnout (\\% of Malay electorate)"),
              ("BN", "BN (\\% of Malay ballots)"),
              ("PN", "PN (\\% of Malay ballots)"),
              ("PH", "PH (\\% of Malay ballots)")]
    for est, name in panels:
        blocks.append(f"\\multicolumn{{{len(common.AGE) + 1}}}{{l}}{{\\textit{{{name}}}}}\\\\[1pt]")
        for el, short in ELS:
            cells = [cell_t(el, a) if est == "turnout" else cell_p(el, a, est) for a in common.AGE]
            blocks.append(f"\\quad {short} & " + " & ".join(cells) + " \\\\")
        blocks.append("\\addlinespace[3pt]")

    caption = ("The Malay reversal by age band, across the three elections")
    head = " & " + " & ".join(a.replace("-", "--") for a in common.AGE) + " \\\\"
    body = "\n".join(blocks)
    out = f"""\\begin{{table}}[!htb]
\\caption{{{caption}}}
\\label{{tab:malay}}
\\centering
\\footnotesize\\setlength{{\\tabcolsep}}{{3.2pt}}
\\begin{{tabular}}{{l{'c' * len(common.AGE)}}}
\\toprule
{head}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(f"{TEX}/tab_malay.tex", "w") as f:
        f.write(out)


def movements():
    """Every claimed movement against the conservative combined error, E_t + E_s."""
    marg = pd.read_csv(f"{OUT}/joint_marginals.csv")
    scaled = pd.read_csv(f"{OUT}/joint_scaled.csv")

    def rate(el, grp, est):
        if est.endswith("_v"):
            r = scaled[(scaled.election == el) & (scaled.group == grp)
                       & (scaled.party == est[:-2])]
            return float(r.share_of_voters.iloc[0])
        r = marg[(marg.election == el) & (marg.group == grp) & (marg.estimand == est)]
        return float(r.rate.iloc[0])

    print("Between-election movements against the conservative combined error\n")
    print(f"{'group':8} {'estimand':9} {'from':6} {'to':6} {'move':>7} {'E_sum':>7} "
          f"{'ratio':>6} {'E_rss':>7}")
    claims = [("Malay", "BN", "ge15", "jhr_se16"), ("Malay", "PN", "ge15", "jhr_se16"),
              ("Malay", "BN_v", "ge15", "jhr_se16"), ("Malay", "PN_v", "ge15", "jhr_se16"),
              ("Malay", "turnout", "ge15", "jhr_se16"),
              ("Chinese", "PH", "ge15", "jhr_se16"), ("Chinese", "BN", "ge15", "jhr_se16"),
              ("Chinese", "turnout", "ge15", "jhr_se16"),
              ("Chinese", "PH_v", "ge15", "jhr_se16"),
              ("Malay", "PN", "jhr_se15", "ge15"), ("Malay", "BN", "jhr_se15", "ge15")]
    for grp, est, a, b in claims:
        mv = 100 * (rate(b, grp, est) - rate(a, grp, est))
        ea, eb = 100 * E.err.get((a, grp, est), np.nan), 100 * E.err.get((b, grp, est), np.nan)
        s, q = ea + eb, np.sqrt(ea ** 2 + eb ** 2)
        print(f"{grp:8} {est:9} {common.cfg(a)['short']:6} {common.cfg(b)['short']:6} "
              f"{mv:7.1f} {s:7.2f} {abs(mv) / s:6.1f} {q:7.2f}")

    print("\nWithin SE-16 turnout contrasts (conservative sum of the two bands' errors)")
    for a, b in [("60-69", "30-39"), ("60-69", "70+"), ("60-69", "18-29")]:
        mv = 100 * (rate("jhr_se16", a, "turnout") - rate("jhr_se16", b, "turnout"))
        s = 100 * (E.err.get(("jhr_se16", a, "turnout"), np.nan)
                   + E.err.get(("jhr_se16", b, "turnout"), np.nan))
        print(f"  {a} vs {b}: {mv:.1f}pp against {s:.2f} (ratio {abs(mv) / s:.1f})")



if __name__ == "__main__":
    results_table()
    crosstab_table()
    errors_table()
    bounds_table()
    m1_table()
    malay_crosstab_table()
    movements()
    print(f"wrote {TEX}/tab_results.tex, {TEX}/tab_crosstab.tex")
