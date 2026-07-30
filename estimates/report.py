"""
The report, rebuilt around the joint crosstab. Usage: python3 report.py

Supersedes 07's report.html as the headline deliverable: the ethnicity x age crosstab (from
the validated joint model, 10-12) leads, the four-cell marginals are derived from it, and the
saluran scatter (13) and marginal trajectories complete the picture. Reads the CSVs the
earlier scripts wrote; recomputes nothing. Overwrites output/report.html by design; 07's
REPORT.md and timeseries CSVs are left intact.

Run order: joint_estimates.py and 13_figures.py must have run first.
"""

import base64
import os
from html import escape

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES, ELECTIONS, ORDER

OUT = common.OUT
FIG = os.path.join(common.REPO, "tex", "dataviz")
ETH, AGE = common.ETH, common.AGE

RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def ramp(v, lo, hi, dark):
    f = 0.5 if hi <= lo else min(max((v - lo) / (hi - lo), 0.0), 1.0)
    steps = RAMP[::-1] if dark else RAMP
    i = int(round(f * (len(steps) - 1)))
    return steps[i], ("#ffffff" if (i >= 7) != dark else "#0b0b0b")


def b64(name):
    with open(f"{FIG}/{name}.png", "rb") as f:
        return base64.b64encode(f.read()).decode()


def crosstab_block(C, el):
    """Four heatmaps (one per estimand), rows=ethnicity, cols=age, coloured within estimand."""
    sub = C[C.election == el]
    blocks = []
    for e in ESTIMAND_NAMES:
        s = sub[sub.estimand == e]
        lo, hi = s.rate.min(), s.rate.max()
        rows = []
        for et in ETH:
            tds = []
            for ag in AGE:
                r = s[(s.eth == et) & (s.age == ag)].iloc[0]
                bl, tl = ramp(r.rate, lo, hi, False)
                bd, td = ramp(r.rate, lo, hi, True)
                star = ('<span class="star">*</span>') if not r.cite else ""
                tip = (f"{escape(et)} · {escape(ag)} · {e}\n"
                       f"estimate {100*r.rate:.1f}%  (95% error ±{100*r.error:.1f}pp)\n"
                       f"interval {100*r.lo:.1f}–{100*r.hi:.1f}%\n"
                       f"registered {int(r.registered):,} ({100*r.share:.1f}% of electorate)\n"
                       f"bounds {100*r.bound_lo:.1f}–{100*r.bound_hi:.1f}%")
                cls = "cell" + ("" if r.cite else " nocite")
                tds.append(f'<td class="{cls}" style="--cl:{bl};--cd:{bd};--tl:{tl};--td:{td}"'
                           f' title="{tip}"><span class="v">{100*r.rate:.0f}{star}</span>'
                           f'<span class="e">±{100*r.error:.0f}</span></td>')
            rows.append(f'<tr><th class="rh">{escape(et)}</th>{"".join(tds)}</tr>')
        head = "".join(f"<th>{a}</th>" for a in AGE)
        blocks.append(f"""<div class="ct">
          <h4>{e}</h4>
          <table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>
        </div>""")
    return f'<div class="ctgrid">{"".join(blocks)}</div>'


def marginal_table(M, dim):
    gs = DIMS[dim]
    sub = M[M.dim == dim]
    rows = []
    for g in gs:
        for k, el in enumerate(ORDER):
            cells = []
            for e in ESTIMAND_NAMES:
                r = sub[(sub.group == g) & (sub.election == el) & (sub.estimand == e)]
                if r.empty:
                    cells.append("<td>–</td>"); continue
                r = r.iloc[0]
                star = "" if r.error <= 0.05 else '<span class="star">*</span>'
                cells.append(f'<td class="mc">{100*r.rate:.1f}<span class="me"> ±{100*r.error:.0f}'
                             f'{star}</span></td>')
            name = f'<th rowspan="3" class="grp">{escape(g)}</th>' if k == 0 else ""
            rows.append(f'<tr>{name}<th class="el">{ELECTIONS[el]["short"]}</th>'
                        + "".join(cells) + "</tr>")
        if g != gs[-1]:
            rows.append('<tr class="sp"><td colspan="6"></td></tr>')
    head = "".join(f"<th>{e}</th>" for e in ESTIMAND_NAMES)
    return (f'<table class="mtab"><thead><tr><th></th><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def main():
    C = pd.read_csv(f"{OUT}/joint_final.csv")
    M = pd.read_csv(f"{OUT}/joint_marginals.csv")
    hold = pd.read_csv(f"{OUT}/joint_holdout.csv")
    hm = (hold[hold.model.isin(["joint", "separate"])]
          .pivot_table(index="dim", columns="model", values="miss", aggfunc="mean") * 100)

    figs = {k: b64(k) for k in ["scatter_malay", "scatter_chinese", "turnout", "collinearity"]}

    def g(dim, grp, e, el):
        r = C[(C.eth == grp) & (C.estimand == e) & (C.election == el)] if dim == "x" else None
        return r

    # a few numbers for the narrative, pulled straight from the crosstab
    se16 = C[C.election == "jhr_se16"]
    def cell(et, ag, e):
        return 100 * se16[(se16.eth == et) & (se16.age == ag) & (se16.estimand == e)].iloc[0].rate
    malay_bn = [cell("Malay", a, "BN") for a in AGE]
    chin_bn = [cell("Chinese", a, "BN") for a in AGE]

    doc = f"""<title>Johor: ethnicity × age crosstab of turnout and party support</title>
<style>
:root {{ color-scheme: light dark; }}
.viz {{
  --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --sec:#52514e; --mut:#84837d; --rule:#e4e3df;
  background:var(--bg); color:var(--ink); min-height:100vh;
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  padding:32px 20px 72px;
}}
@media (prefers-color-scheme:dark){{ :root:where(:not([data-theme=light])) .viz{{
  --bg:#141413; --card:#1c1c1b; --ink:#f2f2ef; --sec:#c3c2b7; --mut:#8f8e85; --rule:#33332f; }} }}
:root[data-theme=dark] .viz{{
  --bg:#141413; --card:#1c1c1b; --ink:#f2f2ef; --sec:#c3c2b7; --mut:#8f8e85; --rule:#33332f; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
h1 {{ font-size:1.6rem; margin:0 0 6px; letter-spacing:-.01em; }}
h2 {{ font-size:1.2rem; margin:46px 0 6px; }}
h3 {{ font-size:1rem; margin:26px 0 4px; }}
p.sub {{ color:var(--sec); margin:0 0 6px; max-width:80ch; }}
p.note {{ color:var(--mut); font-size:.84rem; margin:6px 0 14px; max-width:82ch; }}
.ctgrid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:18px; }}
@media (max-width:720px){{ .ctgrid{{ grid-template-columns:1fr; }} }}
.ct h4 {{ font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; color:var(--sec);
  margin:0 0 5px; }}
.ct table {{ border-collapse:separate; border-spacing:2px; width:100%;
  font-variant-numeric:tabular-nums; }}
.ct thead th {{ font-size:.72rem; color:var(--mut); font-weight:600; padding:0 0 2px; }}
th.rh {{ text-align:right; font-size:.8rem; color:var(--ink); padding-right:8px; width:1%;
  white-space:nowrap; }}
td.cell {{ background:var(--cl); color:var(--tl); border-radius:5px; padding:8px 4px;
  text-align:center; cursor:default; }}
td.cell .v {{ display:block; font-size:1rem; font-weight:650; }}
td.cell .e {{ display:block; font-size:.66rem; opacity:.72; }}
td.nocite {{ opacity:.5; }}
.star {{ font-weight:800; }}
.mtab {{ border-collapse:separate; border-spacing:2px; font-variant-numeric:tabular-nums;
  margin-top:4px; }}
.mtab thead th {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.05em;
  color:var(--mut); padding:0 8px 4px; }}
.mtab td {{ background:var(--card); border:1px solid var(--rule); border-radius:5px;
  padding:6px 10px; text-align:center; min-width:74px; }}
td.mc .me {{ color:var(--mut); font-size:.72rem; }}
th.grp {{ text-align:left; font-weight:650; padding-right:10px; }}
th.el {{ text-align:right; font-size:.76rem; color:var(--sec); font-weight:450; padding-right:6px; }}
tr.sp td {{ height:8px; padding:0; border:none; background:none; }}
:root[data-theme=dark] td.cell {{ background:var(--cd); color:var(--td); }}
@media (prefers-color-scheme:dark){{ :root:where(:not([data-theme=light])) td.cell {{
  background:var(--cd); color:var(--td); }} }}
figure {{ margin:14px 0 0; }}
figure img {{ width:100%; height:auto; border-radius:8px; background:#fcfcfb;
  box-shadow:0 1px 3px rgba(0,0,0,.09); }}
.callout {{ border-left:3px solid #2a78d6; padding:8px 0 8px 14px; margin:16px 0;
  color:var(--sec); }}
ul {{ max-width:82ch; color:var(--sec); }} li {{ margin:5px 0; }}
strong {{ color:var(--ink); }}
.big {{ display:flex; gap:26px; flex-wrap:wrap; margin:14px 0 4px; }}
.big div {{ }} .big .n {{ font-size:1.7rem; font-weight:700; letter-spacing:-.02em; }}
.big .l {{ font-size:.8rem; color:var(--mut); }}
</style>
<div class="viz"><div class="wrap">

<h1>Johor: turnout and party support by ethnicity × age</h1>
<p class="sub">Ecological estimates from 14,222 salurans across SE-15, GE-15 and SE-16, on a
joint 24-cell (ethnicity × age) model validated against real held-out voters. All figures are
<strong>% of that cell's registered voters</strong> — the denominator is registration, so not
voting is one of the measured choices.</p>
<p class="note">The joint model replaces the earlier separate-dimension model, which the
ground-truth test showed was wrong for age (it could not separate ethnicity within an age
band). Figures are the ordinary polling-day electorate; the early+postal bloc is excluded.
<span class="star">*</span> = do not cite (cell holds &lt;1.5% of the electorate, or 95% total
error &gt;5pp). Hover any cell for its full error decomposition.</p>

<div class="big">
  <div><div class="n">4.7 → 1.1pp</div><div class="l">age error vs real ground truth,<br>separate → joint model</div></div>
  <div><div class="n">16 → 2.5pp</div><div class="l">worst cell (70+ BN),<br>separate → joint</div></div>
  <div><div class="n">~1pp</div><div class="l">estimator choice buys this;<br>the row definition bought 4pp</div></div>
</div>

<h2>The crosstab — SE-16 (Jul 2026)</h2>
<p class="note">Read across a row for the age gradient within an ethnicity; down a column for
the ethnic gap within an age band. Colour is scaled within each panel.</p>
{crosstab_block(C, "jhr_se16")}

<div class="callout">
<strong>The age gradient in BN support is composition, not behaviour.</strong> Malay BN support
is flat across age ({malay_bn[0]:.0f}/{malay_bn[1]:.0f}/{malay_bn[2]:.0f}/{malay_bn[3]:.0f}%),
and Chinese BN support is low at every age
({chin_bn[0]:.0f}/{chin_bn[1]:.0f}/{chin_bn[2]:.0f}/{chin_bn[3]:.0f}%). The apparent
“BN falls with age” in the marginal is because the oldest voters are far more Chinese — a
composition effect the separate model mistook for an age effect. Turnout, by contrast, <em>is</em>
genuinely age-graded within each ethnicity (peaking at 50–69), and that finding survives.
</div>

<h2>Seeing the swing: the saluran scatter</h2>
<p class="note">Each point is one saluran, area ∝ electorate; the line is the electorate-weighted
fit through each election's cloud. The two most recent elections are overlaid —
<strong>GE-15 (Nov 2022) in black, SE-16 (Jul 2026) in red</strong> — so the swing reads off
the vertical gap between the two lines. For BN the red line sits well above the black at every
Malay share while for PN it sits well below: <strong>BN's gain is PN's loss, across the whole
Malay range</strong>. Among Chinese the PH lines barely move while turnout falls — support
demobilised rather than switched. The lines are a visual summary only; the cited figures are
the joint-model estimates in the crosstab above, not a read-off of these lines.</p>
<figure><img src="data:image/png;base64,{figs['scatter_malay']}" alt="vote share vs Malay share, GE-15 vs SE-16"/></figure>
<figure><img src="data:image/png;base64,{figs['scatter_chinese']}" alt="vote share vs Chinese share, GE-15 vs SE-16"/></figure>
<p class="note">The Malay and Chinese panels are near-mirror images because the two shares
trade off almost one-for-one across salurans — which is exactly why they carry the
identification, and why the other two groups cannot:</p>
<figure><img src="data:image/png;base64,{figs['collinearity']}" alt="why only Malay and Chinese identify"/></figure>
<figure style="max-width:640px"><img src="data:image/png;base64,{figs['turnout']}" alt="turnout vs composition"/></figure>

<h2>Marginals (derived from the joint model)</h2>
<p class="note">Summing the crosstab over the other dimension recovers the one-way marginals.
They differ from the separately-estimated marginals by &lt;0.4pp for age and &lt;1.2pp for
Malay/Chinese, so the previously reported marginals stand — now sourced from a validated model.
<span class="star">*</span> = error &gt;5pp.</p>
<h3>By ethnicity</h3>
{marginal_table(M, "ethnicity")}
<h3>By age</h3>
{marginal_table(M, "age")}

<h2>How the estimates were validated</h2>
<p class="note">SPR sorts voters into salurans by age, so ~38% of salurans are almost purely
one age band — a stratum where a group's turnout and vote choice are directly observed, not
inferred. Fitting on the mixed salurans and predicting these pure ones tests the model against
real behaviour. Mean miss, percentage points:</p>
<ul>
<li><strong>Age</strong>: separate model {hm.loc['age','separate']:.1f}pp → joint model
    <strong>{hm.loc['age','joint']:.1f}pp</strong></li>
<li><strong>Ethnicity</strong>: separate {hm.loc['ethnicity','separate']:.1f}pp → joint
    <strong>{hm.loc['ethnicity','joint']:.1f}pp</strong></li>
<li>Ethnic voting rates transport across a large age skew (Chinese-pure salurans are +35pp
    older, yet miss ~0pp); age rates do <em>not</em> transport across an ethnic skew unless
    ethnicity is modelled jointly. <strong>Ethnicity is the stable descriptor of Malaysian
    voting; age largely reports ethnic composition.</strong></li>
</ul>
<p class="note">The error bars combine a seat-cluster bootstrap (sampling) with a simulated
aggregation-bias term. That simulated term (~0.5pp on identified cells) is <em>smaller</em>
than the real held-out miss (~1.1–1.4pp), so the bars are best read as a lower bound; the
holdout is the conservative reality check. The deterministic bounds, shown on hover, are the
only assumption-free statement.</p>

</div></div>"""
    with open(f"{OUT}/report.html", "w") as f:
        f.write(doc)
    print(f"wrote {OUT}/report.html")


if __name__ == "__main__":
    main()
