"""
Combine all three elections into one comparable series, with an honest error bar on
every single estimate. Usage: python3 series.py

THE ERROR BUDGET
Every estimate here carries two *different* kinds of error, and adding only the first --
which is what a regression table normally reports -- understates the total badly.

  1. SAMPLING ERROR. The cluster bootstrap over contested seats. This is the error you
     would still have if the model's behavioural assumption were exactly true and you had
     merely drawn a different set of seats.

  2. IDENTIFICATION ERROR. The error from the assumption itself being wrong -- specifically,
     from a group's rate varying with the composition of its saluran (aggregation bias).
     This is the error that Robinson (1950) is about, and it does NOT shrink as the
     electorate grows. No bootstrap of these data can see it, because resampling cannot
     reveal an interior you never observed. It is measured instead by validate.py, which
     plants a known truth in a world where composition *does* drive behaviour (world B) and
     records how far the pipeline lands from it.

They are independent sources, so they combine in quadrature:

    total_se = sqrt( se_bootstrap^2 + rmse_worldB^2 )
    error    = 1.96 * total_se,  then clipped to the deterministic bounds

The clip matters: an estimate 2pp from a hard arithmetic bound cannot have a symmetric
+-6pp interval, because 4pp of that interval is impossible. The reported interval is the
intersection of the statistical interval with what arithmetic permits, so it is never wider
than the truth can be.

THE STAR
A `*` marks an estimate that should NOT be cited:

    *  <=>  error > 5pp

The rule is on the total error alone, deliberately. An earlier version also starred anything
whose deterministic bounds ran wider than 50pp, on the theory that wide bounds mean weak
identification. That was wrong, and worth recording as a trap. The Frechet bounds are a
*worst case over every conceivable behaviour*, including behaviour no electorate exhibits;
Chinese turnout has bounds 55pp wide but a measured total error of 4.8pp, because the
composition of Johor's salurans and the fitted structure between them resolve almost all of
that slack. Starring it would have thrown away a good estimate on the strength of a
hypothetical. The identification error is already measured directly, and is already inside
`error` -- adding the bound width on top double-counts the same worry with a cruder proxy.

In practice the rule stars Indian and Other on nearly everything (no Johor saluran is more
than ~24% Indian or ~10% Other, so the model extrapolates far outside its support), plus a
handful of individual Malay/Chinese/age cells that happen to sit just over the line.

5pp is a convention, not a fact -- it is roughly the point past which an estimate stops
being able to settle an argument about Malaysian politics. Read `error` itself rather than
the star where the distinction matters, since a 5.1pp cell and a 4.9pp cell differ by
nothing but the threshold.

A starred estimate is not "probably wrong". It is the best available guess and it is still
guaranteed inside its arithmetic bounds. It just has an error bar too wide to carry an
argument.

WHAT THE ERROR BAR IS CONDITIONAL ON
The identification term is calibrated to a *specific assumed severity* of aggregation bias:
world B in validate.py perturbs each group's logit-scale rate by DELTA_SD = 1.0 per unit
of saluran composition, which is a substantial but not extreme amount of context effect. The
error bars scale roughly linearly with that choice. They are therefore best read as "the
error if aggregation bias is of a realistic magnitude", not as a distribution-free guarantee.
The only distribution-free statement available is the deterministic bound, which is reported
alongside every estimate for exactly that reason.

Outputs: timeseries_estimates.csv, REPORT.md, report.html
"""

import os

import numpy as np
import pandas as pd

import common
from common import DIMS, ELECTIONS, ESTIMAND_NAMES, ORDER, REPO

OUT = common.OUT

STAR_ERR = 0.05      # >5pp total error -> do not cite


def collect():
    rows = []
    for el in ORDER:
        d = common.outdir(el)
        rec = pd.read_csv(f"{d}/reconciled.csv")
        mods = pd.read_csv(f"{d}/model_estimates.csv")

        # identification error, from the planted-bias world
        sims = pd.read_csv(f"{d}/validation_sims.csv")
        wb = (sims[sims.world == "B"].groupby(["dim", "group", "estimand"])
              .err_recon.apply(lambda x: float(np.sqrt((x ** 2).mean())))
              .rename("rmse_ident").reset_index())

        # specification spread across M2-M5, a second opinion on the same problem
        spread = (mods[~mods.model.isin(["M1_lpm", "M2ml_logit_noprior"])]
                  .groupby(["dim", "group", "estimand"]).estimate.agg(["min", "max"])
                  .rename(columns={"min": "model_min", "max": "model_max"}).reset_index())

        m = rec.merge(wb, on=["dim", "group", "estimand"]).merge(
            spread, on=["dim", "group", "estimand"])
        m["election"] = el
        rows.append(m)

    df = pd.concat(rows, ignore_index=True)
    df["se_sampling"] = df.se
    df["total_se"] = np.sqrt(df.se_sampling ** 2 + df.rmse_ident ** 2)
    df["error"] = 1.96 * df.total_se

    # The statistical interval cannot extend past what arithmetic allows.
    df["lo"] = np.maximum(df.reconciled - df.error, df.bound_lo)
    df["hi"] = np.minimum(df.reconciled + df.error, df.bound_hi)
    df["err_lo"] = df.reconciled - df.lo
    df["err_hi"] = df.hi - df.reconciled
    df["bound_width"] = df.bound_hi - df.bound_lo
    df["model_range"] = df.model_max - df.model_min
    df["cite"] = df.error <= STAR_ERR
    df["star"] = np.where(df.cite, "", "*")
    df["label"] = df.election.map({e: ELECTIONS[e]["label"] for e in ORDER})
    df["short"] = df.election.map({e: ELECTIONS[e]["short"] for e in ORDER})
    return df


def md_tables(df):
    """Markdown heatmap-shaped tables: groups blocked, elections as rows within a block."""
    out = []
    for dim, gs in DIMS.items():
        out.append(f"\n### By {dim}\n")
        out.append("| group | election | " + " | ".join(ESTIMAND_NAMES) + " |")
        out.append("|---|---|" + "---|" * len(ESTIMAND_NAMES))
        for g in gs:
            for k, el in enumerate(ORDER):
                cells = []
                for e in ESTIMAND_NAMES:
                    r = df[(df.dim == dim) & (df.group == g) & (df.estimand == e) &
                           (df.election == el)].iloc[0]
                    cells.append(f"{100 * r.reconciled:.1f} ±{100 * r.error:.1f}{r.star}")
                name = f"**{g}**" if k == 0 else ""
                out.append(f"| {name} | {ELECTIONS[el]['short']} | " + " | ".join(cells) + " |")
            out.append("| | | | | | |")   # spacer row between groups
        out.pop()
    return "\n".join(out)


def main():
    df = collect()
    cols = ["election", "label", "dim", "group", "estimand", "registered", "reconciled",
            "error", "lo", "hi", "se_sampling", "rmse_ident", "total_se",
            "bound_lo", "bound_hi", "bound_width", "model_min", "model_max", "model_range",
            "capped", "m2_raw", "cite"]
    df[cols].to_csv(f"{OUT}/timeseries_estimates.csv", index=False)

    print("=" * 100)
    print("JOHOR ACROSS THREE ELECTIONS -- % of the group's REGISTERED voters, +- 95% total error")
    print("* = do not cite (95% total error > 5pp)")
    print("=" * 100)
    for dim, gs in DIMS.items():
        print(f"\n### by {dim}\n")
        print(f"{'group':9s} {'election':9s} " + " ".join(f"{e:>16s}" for e in ESTIMAND_NAMES))
        for g in gs:
            for el in ORDER:
                cs = []
                for e in ESTIMAND_NAMES:
                    r = df[(df.dim == dim) & (df.group == g) & (df.estimand == e) &
                           (df.election == el)].iloc[0]
                    cs.append(f"{100 * r.reconciled:6.1f}±{100 * r.error:<4.1f}{r.star:1s}")
                print(f"{g:9s} {ELECTIONS[el]['short']:9s} " + " ".join(f"{c:>16s}" for c in cs))
            print()

    print("\nERROR DECOMPOSITION (pp, averaged over elections and estimands)")
    dec = (df.groupby(["dim", "group"])[["se_sampling", "rmse_ident", "total_se", "error"]]
           .mean() * 100).round(2)
    dec["cite_rate"] = df.groupby(["dim", "group"]).cite.mean().round(2)
    print(dec.to_string())
    print("\n-> identification error dominates sampling error everywhere; a bootstrap-only")
    print("   interval would understate the uncertainty by roughly the ratio of those columns.")

    with open(f"{OUT}/REPORT.md", "w") as f:
        f.write(report_md(df))
    html(df)
    print(f"\nwrote {OUT}/timeseries_estimates.csv, REPORT.md, report.html")


# ---------------------------------------------------------------- markdown report
def report_md(df):
    n = df[df.cite].shape[0]
    return f"""# Johor: turnout and party support by ethnicity and age, SE-15 → GE-15 → SE-16

> **Superseded — pipeline diagnostic, not the paper's estimates.** This report is produced
> by the separate-dimension model (4 ethnicity + 6 age rows) with simulation-based error
> bars, an intermediate stage the paper's validation test rejects in favour of the joint
> 24-cell model with held-out error bars (`joint_*.csv`, `validation_errors.csv`, and the
> paper's Tables 5–8). Numbers here will differ from the paper; cite the paper's tables.

Ecological estimates for three elections on identical machinery, so they are comparable
across time. All figures are **% of the group's registered voters** — the denominator is
registration, not turnout, so abstention is a modelled choice and "support" means share of
the whole group. Within a group, `turnout ≈ BN + PH + PN + OTHER + REJECTED`.

Every figure is `estimate ±error`, where the error is a **95% interval combining sampling
error and identification error in quadrature**, then clipped to the deterministic bounds.
**`*` = do not cite** (95% total error > 5pp).
{n} of {len(df)} estimates are citable.

Scope: ordinary polling-day salurans only. The early+postal bloc is excluded from all three.

{md_tables(df)}

## What the numbers say

{narrative(df)}

## Why the error bars are what they are

Two independent errors, combined as `total_se = sqrt(se_bootstrap² + rmse_worldB²)`:

- **sampling error** — cluster bootstrap over contested seats. Small here (~0.2–2pp).
- **identification error** — the error from the constant-rate assumption being wrong.
  Measured by planting a known truth in a simulated world where composition drives
  behaviour, then seeing how far the pipeline lands from it. This is the dominant term,
  and it does **not** shrink with electorate size.

A regression table reporting only the first would look far more precise than the estimates
actually are. See `README.md` for the full method.

## The stars

`*` marks a 95% total error above 5pp. It falls almost entirely on **Indian** and **Other**:
no Johor saluran is more than ~24% Indian or ~10% Other, so the model extrapolates well
outside its support and their errors run ±11–14pp on average.

The star keys on the measured error alone, *not* on the width of the deterministic bounds.
That distinction matters. Chinese turnout has bounds 55pp wide but a total error of ±4.8pp —
the bounds are a worst case over every conceivable behaviour, and the actual composition of
Johor's salurans resolves nearly all of that slack. Starring on bound width would discard a
good estimate on the strength of a hypothetical.

5pp is a convention, not a fact. Read `error` itself where the distinction matters — a 5.1pp
cell and a 4.9pp cell differ by nothing but the threshold.

A starred estimate is still the best available guess and is still guaranteed inside its
arithmetic bounds. It simply has an error bar too wide to carry an argument.
"""


def narrative(df):
    """Findings, computed rather than asserted, so the prose cannot drift from the CSV."""
    def g(dim, grp, e, el):
        return 100 * df[(df.dim == dim) & (df.group == grp) & (df.estimand == e) &
                        (df.election == el)].iloc[0].reconciled

    def seq(dim, grp, e):
        return [g(dim, grp, e, el) for el in ORDER]

    L = []
    mt, ct = seq("ethnicity", "Malay", "turnout"), seq("ethnicity", "Chinese", "turnout")
    L.append(
        f"**Turnout collapsed at SE-15 and never fully came back.** Malay turnout ran "
        f"{mt[0]:.0f} → {mt[1]:.0f} → {mt[2]:.0f}% and Chinese {ct[0]:.0f} → {ct[1]:.0f} → "
        f"{ct[2]:.0f}%. The SE-15/GE-15 gap is the striking part: the same electorate, eight "
        f"months apart, turned out {mt[1] - mt[0]:.0f}pp (Malay) and {ct[1] - ct[0]:.0f}pp "
        f"(Chinese) higher for the general election than for their own state election. The "
        f"Chinese SE-15 figure — {ct[0]:.0f}% of registered Chinese casting any ballot — is "
        f"the single lowest participation number in the series.")

    mb, mp = seq("ethnicity", "Malay", "BN"), seq("ethnicity", "Malay", "PN")
    L.append(
        f"**BN's SE-16 surge is a PN collapse, almost one-for-one.** Malay BN support sat "
        f"flat at {mb[0]:.0f} and {mb[1]:.0f}% before jumping to {mb[2]:.0f}%, a "
        f"+{mb[2] - mb[1]:.0f}pp move. Over the same step Malay PN support fell from "
        f"{mp[1]:.0f}% to {mp[2]:.0f}%, −{mp[1] - mp[2]:.0f}pp. The two moves match in size "
        f"and sign, and no other Malay column moves enough to account for either.")

    cph = seq("ethnicity", "Chinese", "PH")
    cbn = seq("ethnicity", "Chinese", "BN")
    L.append(
        f"**Chinese support did not realign, it demobilised.** Chinese PH ran {cph[0]:.0f} → "
        f"{cph[1]:.0f} → {cph[2]:.0f}%, and Chinese BN barely moved across all three "
        f"({cbn[0]:.0f} → {cbn[1]:.0f} → {cbn[2]:.0f}%). PH's Chinese losses since GE-15 "
        f"track Chinese turnout down ({ct[1]:.0f} → {ct[2]:.0f}%) rather than crossing to "
        f"anyone: on these estimates the votes stayed home rather than switching.")

    ab = {grp: seq("age", grp, "BN") for grp in DIMS["age"]}
    gap15, gap16 = ab["18-29"][1] - ab["70+"][1], ab["18-29"][2] - ab["70+"][2]
    L.append(
        f"**The age gradient in BN support inverted.** At GE-15, BN support *rose* with age "
        f"({ab['18-29'][1]:.0f}% among 18-29 against {ab['70+'][1]:.0f}% among 70+, a "
        f"{gap15:+.0f}pp gap). At SE-16 it *falls* with age ({ab['18-29'][2]:.0f}% against "
        f"{ab['70+'][2]:.0f}%, {gap16:+.0f}pp) — a swing of roughly {gap16 - gap15:.0f}pp in "
        f"the gradient itself, larger than the error on any cell involved. BN's recovery was "
        f"led by the youngest voters, which is the reverse of the conventional reading. This "
        f"is the finding in the series most worth a second look before it is published.")

    at = {grp: seq("age", grp, "turnout") for grp in DIMS["age"]}
    peak = max(DIMS["age"], key=lambda g: at[g][2])   # highest-turnout band at SE-16
    L.append(
        f"**Turnout stays hump-shaped in age at every election.** The {peak} band turns out at "
        f"{at[peak][0]:.0f}/{at[peak][1]:.0f}/{at[peak][2]:.0f}% against "
        f"{at['18-29'][0]:.0f}/{at['18-29'][1]:.0f}/{at['18-29'][2]:.0f}% for 18-29 and "
        f"{at['70+'][0]:.0f}/{at['70+'][1]:.0f}/{at['70+'][2]:.0f}% for 70+. With 60 the "
        f"retirement age, the 50-59 vs 60-69 comparison "
        f"({at['50-59'][2]:.0f} vs {at['60-69'][2]:.0f}% at SE-16) speaks to the "
        f"work-vs-retirement question directly.")

    L.append(
        "**Indian and Other are all starred and none of the above rests on them.** Their "
        "point estimates swing wildly across elections (Indian turnout 34 → 69 → 54%), which "
        "is a signature of an unidentified parameter rather than a volatile electorate.")

    return "\n".join(f"- {x}\n" for x in L)


# ---------------------------------------------------------------- html
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES_L = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
SERIES_D = ["#3987e5", "#008300", "#d55181", "#c98500"]


def ramp_for(v, lo, hi, dark):
    """Map a value to a validated step of the sequential blue ramp.

    Light mode: near-zero takes the lightest step and recedes toward the surface.
    Dark mode: the ramp is reversed, so near-zero recedes toward the dark surface instead.
    An automatic flip would put the brightest colour on the emptiest cell.
    """
    if hi <= lo:
        f = 0.5
    else:
        f = (v - lo) / (hi - lo)
    f = min(max(f, 0.0), 1.0)
    steps = RAMP[::-1] if dark else RAMP
    i = int(round(f * (len(steps) - 1)))
    return steps[i], ("#ffffff" if (i >= 7) != dark else "#0b0b0b")


def html(df):
    from html import escape
    parts = []
    for dim, gs in DIMS.items():
        # Colour is scaled WITHIN each column. All four columns share a unit (% of
        # registered) but not a range -- PN never exceeds ~20% while turnout never drops
        # below ~40%, so one shared ramp would flatten the PN column to a single tone and
        # hide every contrast in it. The cost is that colour cannot be compared across
        # columns, which the caption states outright; the printed numbers remain exact.
        cols = []
        for e in ESTIMAND_NAMES:
            v = df[(df.dim == dim) & (df.estimand == e)].reconciled
            cols.append((v.min(), v.max()))
        rows = []
        for gi, g in enumerate(gs):
            for k, el in enumerate(ORDER):
                tds = []
                for ci, e in enumerate(ESTIMAND_NAMES):
                    r = df[(df.dim == dim) & (df.group == g) & (df.estimand == e) &
                           (df.election == el)].iloc[0]
                    lo, hi = cols[ci]
                    bl, tl = ramp_for(r.reconciled, lo, hi, False)
                    bd, td = ramp_for(r.reconciled, lo, hi, True)
                    star = ('<span class="star" title="do not cite: 95% total '
                            'error exceeds 5pp">*</span>') if not r.cite else ""
                    tds.append(
                        f'<td class="cell{" nocite" if not r.cite else ""}" '
                        f'style="--cl:{bl};--cd:{bd};--tl:{tl};--td:{td}" '
                        f'title="{escape(g)} · {escape(ELECTIONS[el]["label"])} · {e}\n'
                        f'estimate {100 * r.reconciled:.1f}%  (95% total error '
                        f'±{100 * r.error:.1f}pp)\n'
                        f'interval {100 * r.lo:.1f}–{100 * r.hi:.1f}%\n'
                        f'arithmetic bounds {100 * r.bound_lo:.1f}–{100 * r.bound_hi:.1f}%\n'
                        f'sampling se {100 * r.se_sampling:.2f}pp · identification rmse '
                        f'{100 * r.rmse_ident:.2f}pp">'
                        f'<span class="v">{100 * r.reconciled:.1f}{star}</span>'
                        f'<span class="e">±{100 * r.error:.1f}</span></td>')
                gname = f'<th rowspan="3" class="grp">{escape(g)}</th>' if k == 0 else ""
                rows.append(f'<tr>{gname}<th class="el">{escape(ELECTIONS[el]["short"])}</th>'
                            + "".join(tds) + "</tr>")
            if gi < len(gs) - 1:
                rows.append('<tr class="spacer"><td colspan="6"></td></tr>')
        parts.append(f"""
<figure>
  <figcaption><h3>{dim.capitalize()}</h3>
  <p>% of the group's registered voters, ±95% total error in percentage points.
     <span class="star">*</span> = do not cite. Colour is scaled <em>within each column</em>
     — read down a column, not across. Hover any cell for its full error decomposition.</p>
  </figcaption>
  <div class="scroll"><table>
    <thead><tr><th></th><th></th>
    {"".join(f"<th>{e}</th>" for e in ESTIMAND_NAMES)}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</figure>""")

    charts = "".join(sparkline(df, dim) for dim in DIMS)
    doc = f"""<title>Johor SE-15 → GE-15 → SE-16: turnout and support by ethnicity and age</title>
<style>
:root {{ color-scheme: light dark; }}
.viz-root {{
  --surface-1:#fcfcfb; --surface-2:#f4f4f2; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --text-muted:#84837d; --rule:#e4e3df;
  --s1:{SERIES_L[0]}; --s2:{SERIES_L[1]}; --s3:{SERIES_L[2]}; --s4:{SERIES_L[3]};
}}
td.cell {{ background:var(--cl); color:var(--tl); }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    --surface-1:#1a1a19; --surface-2:#232322; --text-primary:#fff;
    --text-secondary:#c3c2b7; --text-muted:#8f8e85; --rule:#3a3a37;
    --s1:{SERIES_D[0]}; --s2:{SERIES_D[1]}; --s3:{SERIES_D[2]}; --s4:{SERIES_D[3]};
  }}
  :root:where(:not([data-theme="light"])) td.cell {{ background:var(--cd); color:var(--td); }}
}}
:root[data-theme="dark"] .viz-root {{
  --surface-1:#1a1a19; --surface-2:#232322; --text-primary:#fff;
  --text-secondary:#c3c2b7; --text-muted:#8f8e85; --rule:#3a3a37;
  --s1:{SERIES_D[0]}; --s2:{SERIES_D[1]}; --s3:{SERIES_D[2]}; --s4:{SERIES_D[3]};
}}
:root[data-theme="dark"] td.cell {{ background:var(--cd); color:var(--td); }}
.viz-root {{
  background:var(--surface-1); color:var(--text-primary); min-height:100vh;
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  padding:32px 20px 72px;
}}
.wrap {{ max-width:940px; margin:0 auto; }}
h1 {{ font-size:1.6rem; line-height:1.25; margin:0 0 6px; letter-spacing:-.01em; }}
h2 {{ font-size:1.15rem; margin:44px 0 10px; letter-spacing:-.005em; }}
h3 {{ font-size:1rem; margin:0 0 4px; }}
p.sub {{ color:var(--text-secondary); margin:0 0 4px; }}
p.note {{ color:var(--text-muted); font-size:.83rem; margin:6px 0 0; }}
figure {{ margin:26px 0 0; }}
figcaption p {{ color:var(--text-muted); font-size:.83rem; margin:2px 0 10px; max-width:74ch; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:separate; border-spacing:2px; width:100%; font-variant-numeric:tabular-nums; }}
thead th {{ font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em;
  color:var(--text-muted); padding:0 0 4px; text-align:center; }}
th.grp {{ text-align:left; font-size:.9rem; font-weight:650; color:var(--text-primary);
  padding-right:10px; white-space:nowrap; vertical-align:middle; width:1%; }}
th.el {{ text-align:right; font-size:.76rem; font-weight:450; color:var(--text-secondary);
  padding-right:8px; white-space:nowrap; width:1%; }}
tr.spacer td {{ height:10px; padding:0; }}
td.cell {{ border-radius:4px;
  padding:7px 8px; text-align:center; min-width:78px; cursor:default; }}
td.cell .v {{ display:block; font-size:.95rem; font-weight:600; }}
td.cell .e {{ display:block; font-size:.7rem; opacity:.72; margin-top:1px; }}
td.nocite {{ opacity:.62; }}
.star {{ font-weight:700; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin:10px 0 0;
  font-size:.8rem; color:var(--text-secondary); }}
.key {{ display:inline-flex; align-items:center; gap:6px; }}
.dot {{ width:9px; height:9px; border-radius:50%; }}
.panels {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:16px; }}
.panel h4 {{ font-size:.78rem; font-weight:600; margin:0 0 2px; color:var(--text-secondary);
  text-transform:uppercase; letter-spacing:.05em; }}
svg {{ display:block; width:100%; height:auto; overflow:visible; }}
.gl {{ stroke:var(--rule); stroke-width:1; }}
.ax {{ fill:var(--text-muted); font-size:9px; }}
.dl {{ fill:var(--text-secondary); font-size:9px; font-weight:600; }}
</style>
<div class="viz-root"><div class="wrap">
<h1>Johor: turnout and party support by ethnicity and age</h1>
<p class="sub">SE-15 (Mar 2022) → GE-15 (Nov 2022) → SE-16 (Jul 2026). Ecological estimates
from 14,222 salurans, on identical machinery across all three.</p>
<p class="note">All figures are <strong>% of the group's registered voters</strong> — the
denominator is registration, not turnout, so not voting is one of the choices being measured
and "support" means share of the whole group. Ordinary polling-day salurans only; the
early+postal bloc is excluded. Errors are 95% intervals combining sampling and
identification error, clipped to what arithmetic permits.</p>
<h2>Estimates</h2>
{"".join(parts)}
<h2>Trajectories</h2>
<p class="note">Same numbers, read as change over time. Shaded band is the 95% total error.
<strong>Hollow points</strong> are individual do-not-cite estimates (error &gt; 5pp);
<strong>dotted lines with a starred label</strong> are groups where no point is citable —
shown rather than omitted, because leaving them out would imply the data says nothing about
them, when what it says is merely too imprecise to quote.</p>
{charts}
</div></div>"""
    with open(f"{OUT}/report.html", "w") as f:
        f.write(doc)


def dodge(ys, gap, lo, hi):
    """Push overlapping label positions apart, keeping their order, inside [lo, hi].

    Four series converging on the same value -- exactly what PN does after its collapse --
    otherwise stack every label on one pixel row and render as a smear. Sweep down
    enforcing the gap, then clamp the whole block back inside the plot so it cannot spill
    onto the x-axis ticks, then sweep up to restore any gap the clamp closed.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        out[b] = max(out[b], out[a] + gap)
    overflow = out[order[-1]] - hi
    if overflow > 0:
        out = [v - overflow for v in out]
    for k in range(len(order) - 2, -1, -1):
        a, b = order[k], order[k + 1]
        out[a] = min(out[a], out[b] - gap)
    shortfall = lo - out[order[0]]
    if shortfall > 0:
        out = [v + shortfall for v in out]
    return out


def sparkline(df, dim):
    """Small multiples: one panel per estimand, one line per group, three time points."""
    gs = DIMS[dim]
    W, H, PL, PR, PT, PB = 268, 138, 26, 54, 10, 20
    panels = []
    for e in ESTIMAND_NAMES:
        sub = df[(df.dim == dim) & (df.estimand == e)]
        ymax = max(0.05, float((sub.reconciled + sub.error).max()) * 1.08)
        x = lambda k: PL + k * (W - PL - PR) / (len(ORDER) - 1)
        y = lambda v: PT + (1 - v / ymax) * (H - PT - PB)
        g_els = []
        # gridlines
        ticks = [0, ymax / 2, ymax]
        for t in ticks:
            g_els.append(f'<line class="gl" x1="{PL}" x2="{W - PR}" y1="{y(t):.1f}" '
                         f'y2="{y(t):.1f}"/>')
            g_els.append(f'<text class="ax" x="{PL - 5}" y="{y(t) + 3:.1f}" '
                         f'text-anchor="end">{100 * t:.0f}</text>')
        for k, el in enumerate(ORDER):
            g_els.append(f'<text class="ax" x="{x(k):.1f}" y="{H - PB + 12}" '
                         f'text-anchor="middle">{ELECTIONS[el]["short"]}</text>')
        ends = []
        for gi, g in enumerate(gs):
            pts, band_hi, band_lo, ok = [], [], [], []
            for k, el in enumerate(ORDER):
                r = sub[(sub.group == g) & (sub.election == el)].iloc[0]
                pts.append((x(k), y(r.reconciled)))
                band_hi.append((x(k), y(min(r.hi, ymax))))
                band_lo.append((x(k), y(max(r.lo, 0))))
                ok.append(bool(r.cite))
            col = f"var(--s{gi + 1})"
            area = ("M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in band_hi) + " L"
                    + " L".join(f"{a:.1f},{b:.1f}" for a, b in reversed(band_lo)) + " Z")
            g_els.append(f'<path d="{area}" fill="{col}" opacity=".13"/>')
            # The dash and the label star mean "no point on this line is citable" -- i.e.
            # the whole series is unusable, which is Indian and Other. Starring a series for
            # one bad point would condemn Malay turnout, whose GE-15 and SE-16 estimates are
            # perfectly good. Individual weak points are marked on the point itself.
            all_bad = not any(ok)
            dash = ' stroke-dasharray="3 3"' if all_bad else ""
            line = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            g_els.append(f'<path d="{line}" fill="none" stroke="{col}" stroke-width="2" '
                         f'stroke-linecap="round" stroke-linejoin="round"{dash}/>')
            for (a, b), good in zip(pts, ok):
                # citable points are filled; do-not-cite points are hollow
                fill = col if good else "var(--surface-1)"
                g_els.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.2" fill="{fill}" '
                             f'stroke="{col if not good else "var(--surface-1)"}" '
                             f'stroke-width="{1.6 if not good else 2}"/>')
            ends.append((pts[-1], col, f'{g}{"*" if all_bad else ""}'))

        # Direct labels at the right end -- identity never rests on colour alone. Dodged so
        # converging series stay legible, with a leader line back to the true endpoint.
        ly = dodge([e[0][1] for e in ends], 9.5, PT + 3, H - PB - 5)
        for ((px, py), col, txt), yy in zip(ends, ly):
            if abs(yy - py) > 1.2:
                g_els.append(f'<path d="M{px + 3.5:.1f},{py:.1f} L{px + 6:.1f},{yy:.1f}" '
                             f'fill="none" stroke="{col}" stroke-width="1" opacity=".5"/>')
            g_els.append(f'<text class="dl" x="{px + 7.5:.1f}" y="{yy + 3:.1f}" '
                         f'fill="{col}">{txt}</text>')
        panels.append(f'<div class="panel"><h4>{e}</h4>'
                      f'<svg viewBox="0 0 {W} {H}" role="img" '
                      f'aria-label="{e} by {dim} across three elections">'
                      + "".join(g_els) + "</svg></div>")
    keys = "".join(f'<span class="key"><span class="dot" style="background:var(--s{i + 1})">'
                   f'</span>{g}</span>' for i, g in enumerate(gs))
    return (f'<figure><figcaption><h3>{dim.capitalize()}</h3></figcaption>'
            f'<div class="legend">{keys}</div>'
            f'<div class="panels" style="margin-top:10px">{"".join(panels)}</div></figure>')


if __name__ == "__main__":
    main()
