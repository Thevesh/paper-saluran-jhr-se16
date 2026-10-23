"""
The results document: the statewide table, then one page per seat. Usage:
    python3 results_tex.py                 statewide + all 36 duns
    python3 results_tex.py --sample        statewide + the sample seat (N.01) only

Ported from paper-saluran-nsn-se16.

Writes tex/results.tex and, if latexmk is on the PATH, compiles it to tex/results.pdf (build
files in tex/out/, gitignored). Every number is read from, or recomputed with, the pipeline:

  output/joint_cells.csv          the joint model's raked cell x choice counts (statewide points)
  output/validation_errors.csv    out-of-sample errors and the tier each was measured at
  output/partial_id_slopes.csv    the measured contextual slopes behind the partial-ID ranges

Scales: turnout is % of the group's registered voters; BN, PH, PN are % of the group's valid
votes.

Uncertainty, in two kinds, never mixed within a cell:
  +-x          out-of-sample error (errors.py), where a near-pure stratum allows the test;
               dagger: measured at >= 90% purity instead of 95%.
  ^hi_lo       partial-identification range (partial_id.py): what the data allow if the group's
               contextual effects are no larger than the validated groups' measured ones,
               fitted inside each seat; the estimate need not sit at its centre.
               Used statewide for the groups with no out-of-sample test (Indian, Other) and
               for every cell of the seat tables, where no out-of-sample test exists at all.
Vote-share ranges are solved directly on the vote-share scale (partial_id.share_lp), not by
dividing a party range by a turnout range.
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

import common
import joint as jm
import partial_id as P

TEX = os.path.join(common.REPO, "tex")
OUT = common.OUT
ELS = common.ORDER
SHORT = {el: common.cfg(el)["short"] for el in ELS}
EST = ["turnout", "BN", "PH", "PN"]
LABEL = {"turnout": "Turnout", "BN": "BN", "PH": "PH", "PN": "PN"}
VALID = ["BN", "PH", "PN", "BERSAMA", "OTHER"]
SAMPLE_SEATS = ["N.01 Buloh Kasap"]

_VE = pd.read_csv(f"{OUT}/validation_errors.csv").set_index(["election", "group", "estimand"])
VE, VTHR = _VE.err, _VE.thr
BETA = float(pd.read_csv(f"{OUT}/partial_id_slopes.csv").slope_mid.abs().quantile(0.9))
RHO = float(P.measure_rho().rho.median())


# ---------------------------------------------------------------- numbers
def points(T, N):
    """Registered-scale rates from a group's choice counts T (Series over choices) and N."""
    valid = sum(T.get(c, 0.0) for c in VALID)
    r = {"turnout": 1 - T["NONVOTE"] / N, "valid": valid / N}
    for p in common.PARTIES:
        r[p] = T[p] / N
    return r


def ranges(r, iv):
    """Display value and (down, up) errors in pp from registered-scale point r and intervals
    iv[est] = (lo, hi). Parties go to the valid-vote scale by propagation."""
    out = {}
    lo, hi = iv["turnout"]
    out["turnout"] = (100 * r["turnout"], 100 * max(r["turnout"] - lo, 0),
                      100 * max(hi - r["turnout"], 0))
    for p in common.PARTIES:
        if iv.get(p) is None:
            out[p] = None
            continue
        s = r[p] / r["valid"] if r["valid"] > 0 else np.nan
        slo, shi = iv[p]                         # already on the vote-share scale
        out[p] = (100 * s, 100 * max(s - slo, 0), 100 * max(shi - s, 0))
    return out


def seat_interval(d, gcol, est):
    """Partial-ID interval inside one territory, falling back to the Frechet bounds where the
    assumptions cannot hold at any relaxation."""
    lo, hi, _ = P.relaxed(d, gcol, est, BETA, RHO)
    if not np.isfinite(lo):
        lo, hi = P.frechet(d, gcol, est)
    return lo, hi


def statewide_cells():
    """{(election, group): {est: (value, down, up, kind)}}; kind 'oos', 'oos90' or 'pid'."""
    C = pd.read_csv(f"{OUT}/joint_cells.csv").fillna(0.0)
    out = {}
    for el in ELS:
        df = common.load(el)
        for g in common.ETH + common.AGE:
            key = "eth" if g in common.ETH else "age"
            T = C[(C.election == el) & (C[key] == g)].sum(numeric_only=True)
            r = points(T, T.registered)
            if (el, g, "turnout") in VE.index:
                cells = {"turnout": (100 * r["turnout"],) + (100 * VE[(el, g, "turnout")],) * 2}
                valid = T[[c for c in VALID if c in T.index]].sum()
                for p in common.PARTIES:
                    e = 100 * VE[(el, g, f"{p}_v")]
                    cells[p] = (100 * T[p] / valid, e, e)
                kind = "oos90" if VTHR[(el, g, "turnout")] < 0.95 else "oos"
            else:
                lo, hi, _, _ = P.by_seat(df, f"n_{g}", "turnout", BETA, RHO)
                iv = {"turnout": (lo, hi)}
                for p in common.PARTIES:              # vote shares, solved directly
                    slo, shi, _ = P.share_relaxed(df, f"n_{g}", p, BETA, RHO, seat="cluster")
                    iv[p] = (slo, shi)
                cells = ranges(r, iv)
                kind = "pid"
            out[(el, g)] = {est: (None if cells[est] is None else cells[est] + (kind,))
                            for est in EST}
    return out


_FITS = {}


def seat_cells(dun):
    """{(election, group): {est: (value, down, up, 'pid') or None}} for one dun."""
    out = {}
    for el in ELS:
        if el not in _FITS:
            df = jm.load_joint(el)
            _FITS[el] = (df, jm.fit_joint_tables(df, el)[0])
        df, per = _FITS[el]
        m = (df.dun == dun).to_numpy()
        d = df[m]
        CH = common.choices(el)
        stood = {p: d[p].sum() > 0 for p in common.PARTIES}
        T_all = per[m].sum(axis=0)                                   # 24 x C in this dun
        for dim, groups in common.DIMS.items():
            for g in groups:
                k = [i for i, (e, a) in enumerate(jm.JOINT) if (e if dim == "ethnicity" else a) == g]
                T = pd.Series(T_all[k].sum(axis=0), index=CH)
                N = T.sum()
                if N == 0:
                    out[(el, g)] = {est: None for est in EST}
                    continue
                r = points(T, N)
                iv = {"turnout": seat_interval(d, f"n_{g}", "turnout")}
                for p in common.PARTIES:              # vote shares, solved directly
                    if stood[p]:
                        slo, shi, _ = P.share_relaxed(d, f"n_{g}", p, BETA, RHO, seat="dun")
                        iv[p] = (slo, shi)
                cells = ranges(r, iv)
                out[(el, g)] = {est: (None if cells[est] is None else cells[est] + ("pid",))
                                for est in EST}
    return out


# ---------------------------------------------------------------- rendering
def fmt(c):
    """Two table columns: value (right-aligned) and its uncertainty (left-aligned)."""
    if c is None:   # not on the ballot: a dash centred on the value/range boundary, inside
        # the fixed-width columns so an all-dash column keeps its width
        return "\\makebox[0pt]{\\textcolor{black!45}{--}} & "
    v, down, up, kind = c
    unc = "\\footnotesize\\textcolor{black!60}"
    if kind in ("oos", "oos90"):
        mark = "$^\\dagger$" if kind == "oos90" else ""
        return f"{v:.1f} & {{{unc}{{$\\pm${down:.1f}{mark}}}}}"
    return f"{v:.1f} & {{{unc}{{$^{{{v + up:.1f}}}_{{{v - down:.1f}}}$}}}}"


def table(cells, groups_by_dim, notes):
    W = 2 * len(ELS)
    head = " & ".join(f"\\multicolumn{{{W}}}{{c}}{{\\textbf{{{LABEL[e]}}}}}" for e in EST)
    cm = "".join(f"\\cmidrule(lr){{{2 + W * i}-{1 + W * (i + 1)}}}" for i in range(len(EST)))
    sub = " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[el]}}}" for _ in EST for el in ELS)
    blocks = []
    for dim, groups in groups_by_dim:
        lines = [f"\\multicolumn{{{1 + W * len(EST)}}}{{l}}{{\\textit{{By {dim}}}}} \\\\[3pt]"]
        for g in groups:
            row = [fmt(cells[(el, g)][est]) for est in EST for el in ELS]
            lines.append(f"\\quad {g.replace('-', '--')} & " + " & ".join(row) + " \\\\")
        blocks.append("\n".join(lines))
    body = "\n\\addlinespace[14pt]\n".join(blocks)
    # fixed widths, so every election's header sits over its numbers the same way whatever
    # the uncertainty text (a +-0.4 is far narrower than a stacked 100.0/0.0 range)
    pair = (">{\\raggedleft\\arraybackslash}p{2.2em}@{\\,}"
            ">{\\raggedright\\arraybackslash}p{2.3em}")
    colspec = "l" + "".join("@{\\hspace{10pt}}" + " ".join([pair] * len(ELS)) for _ in EST)
    items = "\n".join(f"\\item[] {n}" for n in notes)
    return f"""\\begin{{table}}[H]
\\centering\\small
\\begin{{threeparttable}}
\\setlength{{\\tabcolsep}}{{3pt}}\\renewcommand{{\\arraystretch}}{{1.9}}
\\begin{{tabular}}{{{colspec}}}
\\toprule
 & {head} \\\\
{cm}
 & {sub} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}

\\vspace{{16pt}}
\\begin{{tablenotes}}[flushleft]
\\footnotesize\\setlength{{\\itemsep}}{{3pt}}\\setlength{{\\labelsep}}{{0pt}}
{items}
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""


SCALES = ["Turnout is \\% of the group's registered voters.",
          "BN, PH and PN are \\% of the group's valid votes."]
PID_NOTE = ("Small superscript and subscript figures are the upper and lower ends of a "
            "partial-identification range.")
MID_NOTE = "The point estimate is not necessarily the band's midpoint."


def page(title, subtitle, body):
    return f"""\\begin{{center}}
{{\\large\\bfseries {title}}}\\\\[4pt]
{{\\normalsize {subtitle}}}
\\end{{center}}
{body}
\\clearpage
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="only the sample seat (fast preview)")
    ap.add_argument("--cached", action="store_true",
                    help="reuse output/results_cells.pkl: re-render text/layout without recomputing")
    a = ap.parse_args()
    dims = [("ethnicity", common.ETH), ("age", common.AGE)]
    subtitle = " \\quad ".join(common.cfg(el)["label"] for el in ELS)

    import pickle
    cache = f"{OUT}/results_cells.pkl"
    seats = SAMPLE_SEATS if a.sample else sorted(common.load(ELS[-1]).dun.unique())
    if a.cached and os.path.exists(cache):
        with open(cache, "rb") as f:
            C = pickle.load(f)
    else:
        C = {"state": statewide_cells(), "seats": {}}
        for dun in seats:
            C["seats"][dun] = seat_cells(dun)
            print(f"  {dun}: done")
        with open(cache, "wb") as f:
            pickle.dump(C, f)
    state_cells = C["state"]
    soft = any(c and c[3] == "oos90" for cell in state_cells.values() for c in cell.values())
    state = table(state_cells, dims, [
        SCALES[0],
        "BN, PH and PN are \\% of the group's valid votes statewide (inc. zeroes in seats it "
        "did not contest).",
        "$\\pm$ is the out-of-sample validation error, measured against near-pure salurans "
        "($\\geq$95\\% one group); it is not a confidence interval.",
    ] + ([
        "$^\\dagger$ Measured at $\\geq$90\\% instead, because the group's 95\\%-pure salurans "
        "span fewer than 10 polling districts."] if soft else []) + [
        PID_NOTE + " This method of quantifying uncertainty is used where no out-of-sample test "
        "is possible at 90\\% concentration. " + MID_NOTE])
    pages = [page(f"{common.STATE}: turnout and coalition support by ethnicity and age",
                  subtitle, state)]

    for dun in seats:
        pages.append(page(f"{dun}: turnout and coalition support by ethnicity and age", subtitle,
                          table(C["seats"][dun], dims, [
                              SCALES[0],
                              "BN, PH and PN are \\% of the group's valid votes; -- indicates "
                              "the coalition did not contest this seat.",
                              PID_NOTE + " Every cell carries one, because no out-of-sample "
                              "test is possible at seat level. " + MID_NOTE])))
        print(f"  {dun}: done")

    doc = f"""\\documentclass[11pt]{{article}}
\\usepackage[a4paper,landscape,hmargin=0.9in,vmargin=0.45in]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern,microtype,booktabs,float,threeparttable,array}}
\\setlength{{\\intextsep}}{{4pt}}
\\usepackage{{xcolor}}
\\pagestyle{{empty}}
\\begin{{document}}
{"".join(pages)}
\\end{{document}}
"""
    path = f"{TEX}/results.tex"
    with open(path, "w") as f:
        f.write(doc)
    print(f"wrote {path}")
    if shutil.which("latexmk"):
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-outdir=out", "results.tex"], cwd=TEX, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            raise SystemExit("latexmk failed")
        shutil.copy(f"{TEX}/out/results.pdf", f"{TEX}/results.pdf")
        print(f"wrote {TEX}/results.pdf")


if __name__ == "__main__":
    main()
