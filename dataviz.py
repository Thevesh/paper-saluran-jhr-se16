"""
Generates all data visualisations used in the paper.
Dataviz generated:
     1. Coalition vote share and turnout across the three elections.
     2. PN's contest choice: the 33 seats it fought vs the 23 it stood down in.
     3. The Election Commission's age-sorting rule for salurans.
     4. Ethnic composition of each age band.
     5. Why only Malay and Chinese rates are identified.
     6. Turnout vs ethnic composition, GE-15 vs SE-16.
     7. Vote share vs Malay composition, GE-15 vs SE-16.
     8. Vote share vs Chinese composition, GE-15 vs SE-16.
     9. Estimated turnout by ethnicity, all three elections.
    10. Estimated turnout by age band, all three elections.
    11. Vote share vs the saluran's median age, GE-15 vs SE-16.

Estimation lives in estimates/; this script only reads what that pipeline writes to out/.
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estimates")
sys.path.insert(0, EST)
import common
import joint


FIG = "tex/dataviz"

# Colour encodes the ELECTION where two elections are overlaid (black = earlier, red = later),
# and the COALITION or the demographic group where several appear in one panel.
ELECS = [("ge15", "black", "GE-15"), ("jhr_se16", "red", "SE-16")]
PARTY_COLOUR = {"BN": "#031a93", "PH": "#e41a1c", "PN": "#4daf4a"}
# Age runs red for the three youngest bands and blue for the three oldest, darkest at each
# end, so the retirement-age split the paper turns on (Section 5.1) is visible at a glance.
AGE_COLOUR = {"18-29": "#99000d", "30-39": "#cb181d", "40-49": "#fb6a4a",
              "50-59": "#9ecae1", "60-69": "#4292c6", "70+": "#08306b"}
ETH_COLOUR = {"Malay": "#08306b", "Chinese": "#cb181d", "Indian": "orange", "Other": "#969696"}
# Indian and Other are never validated (Section 4.6), so wherever they appear as estimates they
# are hatched rather than drawn solid.
UNVALIDATED = ("Indian", "Other")

RC = {
    "font.size": 12,
    "font.family": "sans-serif",
    "grid.linestyle": "dotted",
    "figure.autolayout": True,
}


def save(name):
    """PNG for review, PDF for the manuscript. Not EPS: the saluran scatters rely on alpha to
    show 4,889 overlapping points, and PostScript renders transparent artists opaque."""
    os.makedirs(FIG, exist_ok=True)
    plt.savefig(f"{FIG}/{name}.png", dpi=400, bbox_inches="tight")
    plt.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    plt.close()


def tidy(ax):
    """Common axis treatment: drop the top and right spines, grey the remaining two so the
    frame recedes and the eye goes to the data, dotted grid behind everything."""
    for b in ["top", "right"]:
        ax.spines[b].set_visible(False)
    for b in ["left", "bottom"]:
        ax.spines[b].set_color("#cccccc")
    ax.tick_params(colors="#666666")
    ax.grid(True, color="lightgrey")
    ax.set_axisbelow(True)


def label(ax, x, y, text, colour, dy=9, size=10, dx=0, ha="center"):
    """A value label on a translucent plate, so overlapping series stay legible."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                fontsize=size, color=colour, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))


def declutter(values, gap):
    """Shift labels apart vertically so none overlaps, preserving their order.

    Three coalitions sit within 3 points of each other at GE-15 in the left panel of Figure 2,
    so labels placed at their true heights collide however they are nudged sideways. This
    returns a label height for each value: the order is kept, colour still identifies the
    series, and the leader dot marks the true position.
    """
    order = np.argsort(values)
    out = np.array(values, float)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        out[hi] = max(out[hi], out[lo] + gap)
    return out


def endpoint_labels(ax, x, series, gap, size=10):
    """Label two-point lines outside their endpoints, decluttered within each end.

    `series` = list of (values, colour) where values has one entry per x position.
    """
    for j, (xi, dx, ha) in enumerate([(x[0], -10, "right"), (x[-1], 10, "left")]):
        ys = [v[j] for v, _ in series]
        for (v, colour), yl in zip(series, declutter(ys, gap)):
            label(ax, xi, yl, f"{v[j]:.1f}", colour, dy=-4, dx=dx, ha=ha, size=size)


def panel_grid(n_top, n_bottom, figsize):
    """Two rows of panels, the shorter row centred: (2,1) or (1,2).

    Used where a third panel would otherwise be squeezed into a strip. Every panel keeps the
    proportions of a two-panel figure instead of shrinking to a third of the text width.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=1.05)
    spans = {1: [(1, 3)], 2: [(0, 2), (2, 4)]}
    axes = [fig.add_subplot(gs[0, a:b]) for a, b in spans[n_top]]
    axes += [fig.add_subplot(gs[1, a:b]) for a, b in spans[n_bottom]]
    return fig, axes


def median_age(df):
    """Median age of a saluran's registered voters, interpolated within the band that holds it.

    The roll gives counts per band, not ages, so the median is the standard grouped-data
    estimate. The open top band is closed at 85, which moves the median only in the handful of
    salurans whose median falls inside it.
    """
    edges = [(18, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 85)]
    n = np.column_stack([df[f"n_{b}"].to_numpy(float) for b in common.AGE])
    cum = np.cumsum(n, axis=1)
    target = n.sum(axis=1) / 2
    j = np.clip((cum < target[:, None]).sum(axis=1), 0, len(edges) - 1)
    r = np.arange(len(j))
    below = np.where(j > 0, cum[r, np.maximum(j - 1, 0)], 0.0)
    inband = n[r, j]
    frac = np.where(inband > 0, (target - below) / np.maximum(inband, 1e-9), 0.5)
    lo = np.array([edges[k][0] for k in j], float)
    hi = np.array([edges[k][1] for k in j], float)
    return lo + np.clip(frac, 0, 1) * (hi - lo)


def contested(df, el, party):
    """Salurans in seats where `party` actually stood.

    PN contested 33 of 56 seats at SE-16, so in the other 23 its vote is exactly zero for
    every saluran. Those points draw a false horizontal line along y = 0 that reads as
    behaviour rather than as a contest structure. They are dropped from the scatter and kept
    in the fit, so the line is unchanged.
    """
    seat = common.cfg(el)["seat_level"]
    total = df.groupby(seat)[party].transform("sum")
    return (total > 0).to_numpy()


def valid_votes(df):
    cols = ["BN", "PH", "PN"] + [c for c in ["BERSAMA", "OTHER"] if c in df.columns]
    return df[cols].sum(axis=1)


def pn_seat_split(df):
    """Split SE-16's 56 seats into those PN contested and those it stood down in.

    PN's absence from 23 seats was a coalition decision, not a failure to nominate, so the two
    groups are treated as distinct throughout the paper.
    """
    pn = df.groupby("dun").PN.sum()
    return sorted(pn[pn > 0].index), sorted(pn[pn == 0].index)


def on_same_territory(ge15, se16):
    """Attach SE-16 dun labels to GE-15 salurans via the polling district.

    Johor's 949 polling districts are identical across the two elections, so this holds
    territory exactly fixed and makes the two results comparable seat group by seat group.
    """
    dm_to_dun = se16.drop_duplicates("dm").set_index("dm").dun
    assert ge15.dm.isin(dm_to_dun.index).all(), "GE-15 polling district absent at SE-16"
    return ge15.assign(dun=ge15.dm.map(dm_to_dun))


# --------------------------------------------------------------------------------- 1


def election_trajectory():
    """
    Coalition vote share and turnout across the three elections.
    """
    rows = []
    for el in common.ORDER:
        df = common.load(el)
        valid = valid_votes(df).sum()
        rows.append({
            "election": common.cfg(el)["short"],
            **{p: 100 * df[p].sum() / valid for p in PARTY_COLOUR},
            "turnout": 100 * (1 - df.NONVOTE.sum() / df.V.sum()),
        })
    df = pd.DataFrame(rows)

    plt.rcParams.update({**RC, "figure.figsize": [9, 4.5]})
    _, axes = plt.subplots(1, 2)
    x = np.arange(len(df))

    for p, colour in PARTY_COLOUR.items():
        axes[0].plot(x, df[p], color=colour, lw=2, marker="o", ms=7, label=p)
    axes[1].plot(x, df.turnout, color="black", lw=2, marker="o", ms=7)

    # BN and PN cross between GE-15 and SE-16 and sit within a point of each other at both
    # 2022 elections, so labels are decluttered column by column. Decluttering only pushes
    # labels up, which parks the lowest label of a tight column right on the marker above it;
    # that label goes under its own dot instead.
    for xi in x:
        vals = [df[p].iloc[xi] for p in PARTY_COLOUR]
        lo = int(np.argmin(vals))
        tight = sorted(vals)[1] - vals[lo] < 5.0
        for j, (p, yl, v) in enumerate(zip(PARTY_COLOUR, declutter(vals, 5.0), vals)):
            if j == lo and tight:
                label(axes[0], xi, v, f"{v:.1f}", PARTY_COLOUR[p], dy=-16)
            else:
                label(axes[0], xi, yl, f"{v:.1f}", PARTY_COLOUR[p], dy=7)
    for xi, yi in zip(x, df.turnout):
        label(axes[1], xi, yi, f"{yi:.1f}", "black")

    axes[0].set_ylabel("% of valid votes\n", linespacing=0.67)
    axes[0].set_ylim(0, 76)
    axes[1].set_ylabel("turnout (% of registered)\n", linespacing=0.67)
    axes[1].set_ylim(40, 85)

    for ax in axes:
        tidy(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(df.election)
        ax.set_xlim(-0.4, len(df) - 0.6)

    axes[0].legend(loc="upper center", ncols=3, frameon=True, framealpha=1,
                   bbox_to_anchor=(0.5, 1.0))
    save("election_trajectory")


# --------------------------------------------------------------------------------- 2


def pn_standdown():
    """
    PN's contest choice: the 33 seats it fought vs the 23 it stood down in.
    """
    se16 = common.load("jhr_se16")
    ge15 = on_same_territory(common.load("ge15"), se16)
    fought, stood_down = pn_seat_split(se16)

    groups = [("PN fought (33 seats)", fought), ("PN stood down (23 seats)", stood_down)]
    rows = []
    for lab, seats in groups:
        for el, df in [("GE-15", ge15), ("SE-16", se16)]:
            d = df[df.dun.isin(seats)]
            valid = valid_votes(d).sum()
            rows.append({"group": lab, "election": el,
                         **{p: 100 * d[p].sum() / valid for p in PARTY_COLOUR}})
    df = pd.DataFrame(rows)

    plt.rcParams.update({**RC, "figure.figsize": [9, 4.6]})
    _, axes = plt.subplots(1, 2, sharey=True)

    # The comparison is a swing on fixed territory, so both panels are read the same way as
    # Figure 1: the slope of each line is the movement, and the two panels differ only in
    # whether PN was on the ballot at the right-hand end.
    x = np.arange(2)
    for ax, (lab, _) in zip(axes, groups):
        d = df[df.group == lab].set_index("election").loc[["GE-15", "SE-16"]]
        for p, colour in PARTY_COLOUR.items():
            ax.plot(x, d[p], color=colour, lw=2, marker="o", ms=7, label=p)
        endpoint_labels(ax, x, [(d[p].to_numpy(), c) for p, c in PARTY_COLOUR.items()], gap=4.5)
        tidy(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(["GE-15", "SE-16"])
        ax.set_xlim(-0.62, 1.62)
        ax.set_ylim(0, 78)
        ax.set_title(lab, fontsize=12, pad=8)

    axes[0].set_ylabel("% of valid votes\n", linespacing=0.67)
    axes[0].legend(loc="upper center", ncols=3, frameon=True, framealpha=1)
    save("pn_standdown")


# --------------------------------------------------------------------------------- 3


def saluran_sorting():
    """
    The Election Commission's age-sorting rule for salurans.
    """
    df = common.load("jhr_se16")
    # Polling districts differ in how many streams they have, so the last stream of a
    # three-stream district and of an eight-stream district are not comparable by number.
    # Relative position within the district is the meaningful axis.
    nmax = df.groupby("dm").saluran.transform("max")
    df = df[nmax >= 3].copy()
    df["pos"] = (df.saluran - 1) / (nmax[nmax >= 3] - 1)
    df["bin"] = pd.cut(df.pos, np.linspace(0, 1, 9), include_lowest=True, labels=False)

    comp = df.groupby("bin").apply(
        lambda d: pd.Series({b: 100 * d[f"n_{b}"].sum() / d.V.sum() for b in common.AGE}),
        include_groups=False,
    )

    plt.rcParams.update({**RC, "figure.figsize": [9.5, 5.0]})
    _, ax = plt.subplots()

    x = np.arange(len(comp))
    bottom = np.zeros(len(comp))
    for b in common.AGE:
        ax.bar(x, comp[b], 0.86, bottom=bottom, color=AGE_COLOUR[b], label=b,
               edgecolor="white", linewidth=0.5)
        bottom += comp[b].to_numpy()

    tidy(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x])
    ax.set_xlabel("\nposition of the saluran within its polling district "
                  "(1 = first stream, 8 = last)", linespacing=1.4)
    ax.set_ylabel("% of the saluran's registered voters\n", linespacing=0.67)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", ncols=6, frameon=True, framealpha=1,
              bbox_to_anchor=(0.5, 1.16), columnspacing=1.1, handlelength=1.4)
    save("saluran_sorting")


# --------------------------------------------------------------------------------- 4


def age_composition():
    """
    Ethnic composition of each age band.
    """
    # The roll carries ethnicity and birth year for the same individual, so the joint
    # margin is counted, not modelled. Here it is collapsed to ethnicity within age band.
    df = joint.load_joint("jhr_se16")
    n = {(e, a): df[f"{e}|{a}"].sum() for e in common.ETH for a in common.AGE}

    plt.rcParams.update({**RC, "figure.figsize": [9, 4.8]})
    _, ax = plt.subplots()

    x = np.arange(len(common.AGE))
    bottom = np.zeros(len(common.AGE))
    for e in common.ETH:
        vals = np.array([100 * n[(e, a)] / sum(n[(g, a)] for g in common.ETH)
                         for a in common.AGE])
        ax.bar(x, vals, 0.7, bottom=bottom, color=ETH_COLOUR[e], label=e,
               edgecolor="white", linewidth=0.5)
        for xi, yi, bi in zip(x, vals, bottom):
            if yi > 5:
                ax.annotate(f"{yi:.0f}", (xi, bi + yi / 2), ha="center", va="center",
                            fontsize=10.5, color="white" if e != "Other" else "black")
        bottom += vals

    tidy(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(common.AGE)
    ax.set_xlabel("\nage band", linespacing=0.67)
    ax.set_ylabel("% of the age band's registered voters\n", linespacing=0.67)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", ncols=4, frameon=True, framealpha=1, bbox_to_anchor=(0.5, 1.14))
    save("age_composition")


# --------------------------------------------------------------------------------- 5


def psize(w):
    """Point areas scaled by electorate, with a floor so a small saluran still reads as a dot
    and a cap so a large one does not blot the panel."""
    return np.clip(5.0 * w / w.mean(), 3.0, 40.0)


def support(x, w, q=0.99):
    """The range of x holding the middle `q` of the ELECTORATE, not of the salurans.

    Fitted lines are drawn across this range only. Weighting by voters rather than by units
    is the same rule the collinearity panel uses: identification is carried by where the
    electorate actually is, and a 30-voter saluran at an extreme is not data reaching that far.
    """
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    lo = x[o][np.searchsorted(cw, (1 - q) / 2)]
    hi = x[o][min(np.searchsorted(cw, 1 - (1 - q) / 2), len(x) - 1)]
    return lo, hi


def wls_line(x, y, w):
    """Weighted least squares y ~ a + b x."""
    W = np.sqrt(w)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    return coef


def scatter_panel(ax, series, alpha=0.18, smul=1.0):
    """Overlay one panel with several elections.

    `series` = list of (x, y, w, colour) or (x, y, w, colour, show), where `show` is a boolean
    mask of the points to DRAW. The fit always uses every point, so hiding the uncontested
    seats changes the cloud and not the line.

    `alpha` and `smul` exist for the single-seat panels, where a cloud is tens of points rather
    than thousands and the settings tuned for overplotting leave it almost invisible.

    Each election gets its cloud and a weighted fit line drawn across the range the electorate
    covers. The line stops where the voters stop and is never labelled with an endpoint value:
    reading a fitted line off at x = 100% IS the bivariate ecological estimate, not the
    joint-model estimate the paper reports. Its job here is to summarise where each election's cloud sits, so that the swing
    between the two reads as the vertical gap between the lines.
    """
    for t in series:
        x, y, w, colour = t[:4]
        m = t[4] if len(t) > 4 else np.ones(len(x), bool)
        ax.scatter(100 * x[m], 100 * y[m], s=smul * psize(w)[m], c=colour, alpha=alpha,
                   edgecolors="none", rasterized=True)
    for t in series:
        x, y, w, colour = t[:4]
        a, b = wls_line(x, y, w)
        lo, hi = support(x, w)
        xs = np.linspace(lo, hi, 50)
        ax.plot(100 * xs, 100 * (a + b * xs), color=colour, lw=2, zorder=4)

    ax.set_xlim(-3, 103)
    ax.set_ylim(-4, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    tidy(ax)


def election_legend(ax, loc="upper center", ncols=2):
    handles = [Line2D([0], [0], color=c, lw=2, marker="o", ms=6, label=lab)
               for _, c, lab in ELECS]
    ax.legend(handles=handles, loc=loc, ncols=ncols, frameon=True, framealpha=1)


def collinearity():
    """
    Why only Malay and Chinese rates are identified.
    """
    df = common.load("jhr_se16")
    w = df.V.to_numpy(float)

    plt.rcParams.update({**RC, "figure.figsize": [9.5, 9.0], "figure.autolayout": False})
    fig, axes = panel_grid(1, 2, [9.5, 9.0])

    # Malay against Chinese sits alone on the top row because it is the contrast that carries
    # identification; Indian and Other are below, on a shorter y-axis, because that is the
    # whole point -- the electorate never reaches far along either.
    pairs = [("Malay", "Chinese"), ("Malay", "Indian"), ("Malay", "Other")]
    for ax, (p, q) in zip(axes, pairs):
        xa = 100 * (df[f"n_{p}"] / df.V)
        xb = 100 * (df[f"n_{q}"] / df.V)
        ax.scatter(xa, xb, s=psize(w), c="black", alpha=0.18, edgecolors="none", rasterized=True)
        ax.set_xlabel(f"\n{p} % of electorate", linespacing=0.67)
        ax.set_ylabel(f"{q} % of electorate\n", linespacing=0.67)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103 if q == "Chinese" else 40)
        tidy(ax)

        # A voter-weighted 99th percentile, not the raw maximum: identification is carried by
        # where the electorate actually is, and one 30-voter saluran that happens to be 89%
        # Indian is not data reaching that far.
        order = np.argsort(xb.to_numpy())
        cw = np.cumsum(w[order]) / w.sum()
        p99 = xb.to_numpy()[order][np.searchsorted(cw, 0.99)]
        ax.axhline(p99, color="red", lw=1, ls="--")
        ax.annotate(f"99th percentile by voter: {p99:.0f}%", (103, p99),
                    textcoords="offset points", xytext=(0, 5), ha="right",
                    fontsize=9.5, color="red")

    save("collinearity")


# --------------------------------------------------------------------------------- 6


def turnout():
    """
    Turnout vs ethnic composition, GE-15 vs SE-16.
    """
    plt.rcParams.update({**RC, "figure.figsize": [9, 4.6]})
    _, axes = plt.subplots(1, 2)

    for ax, g in zip(axes, ["Malay", "Chinese"]):
        series = []
        for el, colour, _ in ELECS:
            df = common.load(el)
            series.append(((df[f"n_{g}"] / df.V).to_numpy(float),
                           (1 - df.NONVOTE / df.V).to_numpy(float),
                           df.V.to_numpy(float), colour))
        scatter_panel(ax, series)
        ax.set_xlabel(f"\n{g} % of electorate", linespacing=0.67)

    axes[0].set_ylabel("turnout (% of registered)\n", linespacing=0.67)
    election_legend(axes[0])
    save("turnout")


# --------------------------------------------------------------------------------- 7, 8


def party_scatter(group, order):
    """
    Vote share vs the given group's share of the electorate, GE-15 vs SE-16.

    `order` puts the two coalitions that move most for this group on the top row, with the
    third centred beneath, so no panel is squeezed to a third of the text width.
    """
    data = {el: common.load(el) for el, _, _ in ELECS}
    valid = {el: valid_votes(df) for el, df in data.items()}

    plt.rcParams.update({**RC, "figure.figsize": [9.5, 9.0], "figure.autolayout": False})
    fig, axes = panel_grid(2, 1, [9.5, 9.0])

    for ax, p in zip(axes, order):
        series = []
        for el, colour, _ in ELECS:
            df, v = data[el], valid[el]
            ok = v.to_numpy() > 0
            series.append(((df[f"n_{group}"] / df.V).to_numpy(float)[ok],
                           (df[p] / v.where(v > 0)).to_numpy(float)[ok],
                           df.V.to_numpy(float)[ok], colour,
                           contested(df, el, p)[ok]))
        scatter_panel(ax, series)
        ax.set_xlabel(f"\n{group} % of electorate", linespacing=0.67)
        ax.set_ylabel(f"{p} % of valid votes\n", linespacing=0.67)

    election_legend(axes[0])
    save(f"scatter_{group.lower()}")


# --------------------------------------------------------------------------------- 8b


def seat_scatter(seats, group="Chinese", party="PH"):
    """
    Vote share vs the given group's share of the electorate, one panel per seat, GE-15 vs SE-16.

    The state-wide scatter (Figures 7, 8) pools every saluran, so a seat-level story can only be
    read off it as a residual. Here the same two axes are drawn inside four seats, each of which
    spans nearly the full Chinese range on its own, so the gap between the two fitted lines is a
    swing within fixed territory rather than across a changing seat mix.

    GE-15 was fought on parliamentary boundaries, so its salurans carry no dun label. They are
    grouped by polling district, which is identical across the two elections, so each panel holds
    exactly the same ground at both.
    """
    data = {el: common.load(el) for el, _, _ in ELECS}
    data["ge15"] = on_same_territory(data["ge15"], data["jhr_se16"])
    valid = {el: valid_votes(df) for el, df in data.items()}

    plt.rcParams.update({**RC, "figure.figsize": [9.5, 9.0], "figure.autolayout": False})
    fig, axes = plt.subplots(2, 2, figsize=[9.5, 9.4])
    fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.28)

    for ax, seat in zip(axes.ravel(), seats):
        series = []
        for el, colour, _ in ELECS:
            df, v = data[el], valid[el]
            ok = ((df.dun == seat) & (v > 0)).to_numpy()
            assert ok.any(), f"{seat} absent from {el}"
            series.append(((df[f"n_{group}"] / df.V).to_numpy(float)[ok],
                           (df[party] / v.where(v > 0)).to_numpy(float)[ok],
                           df.V.to_numpy(float)[ok], colour))
        # Tens of points per panel, not thousands: the cloud needs opacity and size to read.
        scatter_panel(ax, series, alpha=0.55, smul=2.0)
        ax.set_title(seat, fontsize=12, pad=10)
        ax.set_xlabel(f"\n{group} % of electorate", linespacing=0.67)
        ax.set_ylabel(f"{party} % of valid votes\n", linespacing=0.67)

    # One legend for the figure, above the panels: with four seat titles in play there is no
    # panel corner free enough to hold it without sitting on the cloud or on a title.
    handles = [Line2D([0], [0], color=c, lw=2, marker="o", ms=6, label=lab)
               for _, c, lab in ELECS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncols=2,
               frameon=True, framealpha=1)
    save(f"scatter_seats_{group.lower()}")


# --------------------------------------------------------------------------------- 9, 10


def estimated_turnout():
    """Load estimated turnout by group and election from the joint model's marginals."""
    m = pd.read_csv(f"{common.OUT}/joint_marginals.csv")
    e = pd.read_csv(f"{common.OUT}/validation_errors.csv")
    e = e[e.estimand == "turnout"].set_index(["election", "group"]).err
    m = m[m.estimand == "turnout"].set_index(["election", "group"]).rate
    return m, e


def turnout_ethnicity():
    """
    Estimated turnout by ethnicity, all three elections.
    """
    rate, err = estimated_turnout()

    plt.rcParams.update({**RC, "figure.figsize": [9, 4.8]})
    _, ax = plt.subplots()

    width, centres = 0.2, np.arange(len(common.ORDER))
    for k, g in enumerate(common.ETH):
        vals = np.array([100 * rate.get((el, g), np.nan) for el in common.ORDER])
        bars = np.array([100 * err.get((el, g), np.nan) for el in common.ORDER])
        pos = centres + (k - (len(common.ETH) - 1) / 2) * width
        # Indian and Other could not be validated (Section 4.6), so they are hatched and
        # carry no error bar. The hatching is the same signal as the greying in the tables.
        ax.bar(pos, vals, width, color=ETH_COLOUR[g], label=g, edgecolor="white", linewidth=0.6,
               hatch="//" if g in UNVALIDATED else None, alpha=0.55 if g in UNVALIDATED else 1.0)
        ax.errorbar(pos, vals, yerr=bars, fmt="none", ecolor="black", elinewidth=1, capsize=3)
        for xi, yi in zip(pos, vals):
            label(ax, xi, yi, f"{yi:.0f}", ETH_COLOUR[g], dy=11, size=9)

    tidy(ax)
    ax.set_xticks(centres)
    ax.set_xticklabels([common.cfg(el)["short"] for el in common.ORDER])
    ax.set_ylabel("turnout (% of registered)\n", linespacing=0.67)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", ncols=4, frameon=True, framealpha=1)
    save("turnout_ethnicity")


def turnout_age():
    """
    Estimated turnout by age band, all three elections.
    """
    rate, err = estimated_turnout()

    plt.rcParams.update({**RC, "figure.figsize": [9, 5.0]})
    _, ax = plt.subplots()

    x = np.arange(len(common.AGE))
    styles = {"jhr_se15": ("#999999", "-"), "ge15": ("black", "-"), "jhr_se16": ("red", "-")}
    for el in common.ORDER:
        colour, ls = styles[el]
        vals = np.array([100 * rate.get((el, b), np.nan) for b in common.AGE])
        bars = np.array([100 * err.get((el, b), np.nan) for b in common.AGE])
        ax.plot(x, vals, color=colour, ls=ls, lw=2, marker="o", ms=7,
                label=common.cfg(el)["short"])
        ax.fill_between(x, vals - bars, vals + bars, color=colour, alpha=0.12, lw=0)

    # The retirement-age split is the point of the six bands, so mark it.
    ax.axvline(3.5, color="grey", lw=1, ls=":")
    ax.annotate("retirement age", (3.5, 92), ha="center", fontsize=9.5, color="grey")

    tidy(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("-", "\u2013") for b in common.AGE])
    ax.set_xlabel("\nage band", linespacing=0.67)
    ax.set_ylabel("turnout (% of registered)\n", linespacing=0.67)
    ax.set_ylim(35, 95)
    ax.legend(loc="lower center", ncols=3, frameon=True, framealpha=1)
    save("turnout_age")


# --------------------------------------------------------------------------------- 11


def age_scatter():
    """
    Vote share vs the saluran's median age, GE-15 vs SE-16.
    """
    data = {el: common.load(el) for el, _, _ in ELECS}
    valid = {el: valid_votes(df) for el, df in data.items()}

    plt.rcParams.update({**RC, "figure.figsize": [9.5, 9.0], "figure.autolayout": False})
    fig, axes = panel_grid(2, 1, [9.5, 9.0])

    for ax, p in zip(axes, ["BN", "PN", "PH"]):
        for el, colour, _ in ELECS:
            df, v = data[el], valid[el]
            ok = v.to_numpy() > 0
            x = median_age(df)[ok]
            y = 100 * (df[p] / v.where(v > 0)).to_numpy(float)[ok]
            w = df.V.to_numpy(float)[ok]
            m = contested(df, el, p)[ok]
            ax.scatter(x[m], y[m], s=psize(w)[m], c=colour, alpha=0.18, edgecolors="none",
                       rasterized=True)
            a, b = wls_line(x, y, w)
            lo, hi = support(x, w)
            xs = np.linspace(lo, hi, 50)
            ax.plot(xs, a + b * xs, color=colour, lw=2, zorder=4)
        tidy(ax)
        # The EC sorts voters into streams by age, so saluran median age runs from the low
        # twenties to the high seventies -- far more spread than a geographic unit would give.
        ax.set_xlim(20, 80)
        ax.set_ylim(-4, 104)
        ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
        ax.set_xlabel("\nmedian age of the saluran's electorate", linespacing=0.67)
        ax.set_ylabel(f"{p} % of valid votes\n", linespacing=0.67)

    election_legend(axes[0])
    save("age_scatter")


if __name__ == "__main__":
    print("\nPlotting coalition vote share and turnout across the three elections...")
    election_trajectory()
    print("\nPlotting PN's contest choice, fought vs stood down...")
    pn_standdown()
    print("\nPlotting the Election Commission's age-sorting rule...")
    saluran_sorting()
    print("\nPlotting ethnic composition of each age band...")
    age_composition()
    print("\nPlotting why only Malay and Chinese rates are identified...")
    collinearity()
    print("\nPlotting turnout vs ethnic composition...")
    turnout()
    print("\nPlotting vote share vs Malay composition...")
    party_scatter("Malay", ["BN", "PN", "PH"])
    print("\nPlotting vote share vs Chinese composition...")
    party_scatter("Chinese", ["BN", "PH", "PN"])
    print("\nPlotting PH vote share vs ethnic composition in four seats...")
    for g in ["Chinese", "Malay"]:
        seat_scatter(["N.19 Yong Peng", "N.30 Paloh", "N.46 Perling", "N.51 Bukit Batu"], g)
    print("\nPlotting estimated turnout by ethnicity...")
    turnout_ethnicity()
    print("\nPlotting estimated turnout by age band...")
    turnout_age()
    print("\nPlotting vote share vs the saluran's median age...")
    age_scatter()
    print("\n✨✨✨ DONE ✨✨✨\n")
