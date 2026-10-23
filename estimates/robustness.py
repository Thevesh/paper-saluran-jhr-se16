"""Referee-facing robustness checks on the joint model and its error bars.
Usage: python3 robustness.py [election ...]

Three questions, each raised by careful review of the validation design, each answered
against the data rather than by assertion.

1. PRIOR SENSITIVITY. The N(0, 2.5^2) prior on each logit coefficient (Gelman et al. 2008)
   is negligible wherever the likelihood is informative, but on unidentified cells it shrinks
   the rate towards 0.5. If a printed number is the prior talking, refitting under a
   different prior moves it; if it is the data talking, it does not. We refit the entire
   joint model at prior sd in {1.0, 2.5, 5.0} plus unpenalised ML, and report how far every
   marginal moves. Expected: validated groups frozen, Indian/Other slide.

2. TRANSPORTABILITY. The error bars are measured in near-pure streams and carried to mixed
   ones, which assumes a group behaves the same across the composition range. Two probes:

   a. EXTRAPOLATION HOLDOUT. The standard design fits on everything below 95% purity,
      which includes streams at 90% -- close cousins of the target. Refit using only
      streams below 80% purity (genuinely mixed) and predict the >=95% pure ones. If the
      constancy assumption fails along the composition gradient, this harder extrapolation
      must degrade; if the miss barely moves, behaviour is stable across the range the
      transport crosses.

   b. SPLIT-SAMPLE STABILITY. Within the mixed streams (<95% pure), split at the
      electorate-weighted median of the dominant group's share and fit the joint model
      separately on each half. Both halves see NO near-pure stream. Cell rates from the
      two halves are aggregated to marginals with the SAME statewide weights, so any gap
      is behavioural drift with composition, not composition itself.

3. THE CENSORED-MISS BRACKET. A prediction inside the permitted interval scores zero, so
   the reported error is a lower envelope of the true holdout distance. But the same
   interval supplies a certified UPPER envelope: the truth is inside [L, U], so the distance
   from prediction to truth is at most max(|p-L|, |p-U|). We run the identical district
   bootstrap on that quantity and report both ends. The gap between the envelopes is the
   width of the target, which is small exactly because the strata are near-pure.

4. LOCALITY (design C). Reweight the training streams to the held-out strata's district-size
   distribution, predict, and report signed district-level residuals against district size.
   Design B only filters to districts holding both kinds of stream; C forces a common
   covariate distribution.

5. SPLIT AT THE JOINT-CELL LEVEL, under two partitions (compositional gradient, district
   size), because offsetting cell instability can vanish inside a marginal.

6. PURITY-THRESHOLD FRONTIER: width, districts, coverage, training size and both envelopes
   on one grid -- the threshold is a design choice that moves all of them at once.

Outputs
    output/robustness_prior.csv       marginals under each prior
    output/robustness_locality.csv, robustness_district_residuals.csv, robustness_residual_trend.csv
    output/robustness_split_joint.csv
    output/robustness_threshold_frontier.csv
    output/robustness_transport.csv   extrapolation holdout misses
    output/robustness_split.csv       split-sample marginal gaps
    output/robustness_envelope.csv    lower/upper error envelopes per estimand
"""

import numpy as np
import pandas as pd

import common
from common import DIMS, ESTIMAND_NAMES

import models
import groundtruth as gt
import joint as jm
import holdout as hold
import errors as err

OUT = common.OUT
JOINT = jm.JOINT
NBOOT = 2000
SEED = 20260711
PRIORS = [1.0, 2.5, 5.0, None]      # None = unpenalised ML
FIT_THRESHOLDS = [0.80, 0.95]       # 0.95 is the paper's design; 0.80 is the hard version


def patched_fit_link(sd):
    orig = models.fit_link

    def f(X, Y, V, link, start=None, prior_sd="unused", offset=0.0):
        return orig(X, Y, V, link, start=start, prior_sd=sd, offset=offset)

    return orig, f


# ---------------------------------------------------------------- 1. prior sensitivity
def prior_sensitivity(els):
    rows = []
    for el in els:
        df = jm.load_joint(el)
        N_tot = jm.jcounts(df).sum(axis=0)
        for sd in PRIORS:
            orig, f = patched_fit_link(sd)
            models.fit_link = f
            try:
                total, _ = jm.fit_joint(df, el)
            finally:
                models.fit_link = orig
            for dim, gs in DIMS.items():
                for e in ESTIMAND_NAMES:
                    mm = jm.marginal(total, N_tot, el, e, dim)
                    for j, g in enumerate(gs):
                        rows.append(dict(election=el, prior_sd=np.inf if sd is None else sd,
                                         ml=sd is None, dim=dim, group=g, estimand=e,
                                         rate=float(mm[j])))
        print(f"  prior sweep {el}: done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/robustness_prior.csv", index=False)

    base = R[R.prior_sd == 2.5].set_index(["election", "dim", "group", "estimand"]).rate
    R["shift"] = 100 * (R.rate - base.reindex(
        R.set_index(["election", "dim", "group", "estimand"]).index).to_numpy())
    validated = common.ETH[:2] + common.AGE
    R["validated"] = R.group.isin(validated)
    print("\nMax |shift| from the reported prior (sd 2.5), pp, by prior and group class\n")
    p = R.pivot_table(index="prior_sd", columns="validated", values="shift",
                      aggfunc=lambda s: s.abs().max())
    p.columns = ["Indian/Other", "validated groups"]
    print(p.round(2).to_string())
    print("\nIndian & Other turnout at SE-16 under each prior (the cell the text flags):")
    s = R[(R.election == els[-1]) & (R.group.isin(["Indian", "Other"])) &
          (R.estimand == "turnout")]
    print(s.pivot_table(index="group", columns="prior_sd", values="rate")
           .mul(100).round(1).to_string())
    return R


# ---------------------------------------------------------------- 2a. extrapolation holdout
def stream_table_at(df, el, dim, keep_groups, fit_thr):
    """errors.stream_table, but the fit set is capped at fit_thr purity.

    The pure (target) stratum stays at the paper's 0.95 threshold throughout; only the
    training set shrinks. fit_thr=0.95 reproduces the paper's design A exactly.
    """
    pure, arg = gt.split(df, dim, hold.THR)
    Xd = common.shares(df, dim)
    fit_mask = Xd.max(axis=1) < fit_thr - 1e-9
    fit_df = df[fit_mask].reset_index(drop=True)

    total, _ = jm.fit_joint(fit_df, el)
    Nf = jm.jcounts(fit_df).sum(axis=0)
    b = {e: jm.jrates(total, Nf, el, e) for e in ESTIMAND_NAMES}

    Njoint, V, N4 = jm.jcounts(df), df.V.to_numpy(float), common.counts(df, dim)
    out = {}
    for j, g in enumerate(DIMS[dim]):
        if g not in keep_groups:
            continue
        m = pure & (arg == j)
        if m.sum() == 0:
            continue
        idx = [k for k, (e, a) in enumerate(JOINT) if (e if dim == "ethnicity" else a) == g]
        w = Njoint[m][:, idx]
        t = dict(dm=df.dm.to_numpy()[m], V=V[m], N=N4[m, j],
                 wt=np.maximum(w.sum(axis=1), 1e-9), group_electorate=N4[:, j].sum())
        for e in ESTIMAND_NAMES:
            t[f"Y_{e}"] = common.outcome(df, el, e)[m]
            t[f"p_{e}"] = (w * b[e][idx]).sum(axis=1) / t["wt"]
        out[g] = pd.DataFrame(t)
    return out, int(fit_mask.sum())


def transport(els):
    rows = []
    for el in els:
        df = jm.load_joint(el)
        for dim in DIMS:
            keep = sorted({r["group"] for r in hold.holdout(df, el, dim, "A")})
            for thr in FIT_THRESHOLDS:
                tabs, n_fit = stream_table_at(df, el, dim, keep, thr)
                for g, t in tabs.items():
                    for e, v in err.aggregate_miss(t).items():
                        rows.append(dict(election=el, dim=dim, group=g, estimand=e,
                                         fit_thr=thr, n_fit=n_fit, miss=v))
        print(f"  transport {el}: done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/robustness_transport.csv", index=False)

    p = R.pivot_table(index=["dim", "group", "estimand"], columns="fit_thr",
                      values="miss") * 100
    print("\nAggregate holdout miss (pp): fit on <95% pure (paper) vs fit on <80% pure only\n")
    print(f"mean  0.80: {p[0.80].mean():.2f}   0.95: {p[0.95].mean():.2f}")
    print(f"max   0.80: {p[0.80].max():.2f}   0.95: {p[0.95].max():.2f}")
    core = p[~p.index.get_level_values("estimand").str.endswith("_v")]
    print(f"core estimands only -- mean 0.80: {core[0.80].mean():.2f}  0.95: "
          f"{core[0.95].mean():.2f};  max 0.80: {core[0.80].max():.2f}  0.95: "
          f"{core[0.95].max():.2f}")
    return R


# ---------------------------------------------------------------- 2b. split-sample stability
def split_stability(els):
    rows = []
    for el in els:
        df = jm.load_joint(el)
        N_state = jm.jcounts(df).sum(axis=0)
        for dim, gs in DIMS.items():
            Xd = common.shares(df, dim)
            mixed = Xd.max(axis=1) < hold.THR - 1e-9
            # Context variable: Malay share for ethnicity (the compositional axis of the
            # state); stream purity for age, where the sorting rule makes the dominant
            # band's share the natural gradient from thoroughly-mixed to nearly-pure.
            lead = Xd[:, 0] if dim == "ethnicity" else Xd.max(axis=1)
            V = df.V.to_numpy(float)
            order = np.argsort(lead[mixed])
            cum = np.cumsum(V[mixed][order]) / V[mixed].sum()
            med = float(np.sort(lead[mixed])[np.searchsorted(cum, 0.5)])
            halves = {"low": mixed & (lead < med), "high": mixed & (lead >= med)}
            cell_rates = {}
            for name, m in halves.items():
                sub = df[m].reset_index(drop=True)
                total, _ = jm.fit_joint(sub, el)
                Nh = jm.jcounts(sub).sum(axis=0)
                cell_rates[name] = {e: jm.jrates(total, Nh, el, e) for e in ESTIMAND_NAMES}
            for e in ESTIMAND_NAMES:
                for j, g in enumerate(gs):
                    idx = [k for k, (et, a) in enumerate(JOINT)
                           if (et if dim == "ethnicity" else a) == g]
                    w = N_state[idx]
                    r = {name: float((w * cell_rates[name][e][idx]).sum() / w.sum())
                         for name in halves}
                    rows.append(dict(election=el, dim=dim, group=g, estimand=e,
                                     split_at=float(med), low=r["low"], high=r["high"],
                                     gap=100 * (r["high"] - r["low"]),
                                     n_low=int(halves["low"].sum()),
                                     n_high=int(halves["high"].sum())))
        print(f"  split {el}: done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/robustness_split.csv", index=False)

    validated = common.ETH[:2] + common.AGE
    v = R[R.group.isin(validated)]
    print("\nSplit-sample marginal gap |high - low| (pp), validated groups, statewide weights\n")
    print(f"mean {v.gap.abs().mean():.2f}   median {v.gap.abs().median():.2f}   "
          f"p90 {v.gap.abs().quantile(0.9):.2f}   max {v.gap.abs().max():.2f}")
    print("\nworst five:")
    print(v.reindex(v.gap.abs().sort_values(ascending=False).index)
           .head(5)[["election", "dim", "group", "estimand", "low", "high", "gap"]]
           .round(3).to_string(index=False))
    return R


# ---------------------------------------------------------------- 3. the censored bracket
def envelope_miss(t, rows=None):
    """(lower, upper, width) per estimand: truncated miss, certified max distance, U-L."""
    s = t if rows is None else t.iloc[rows]
    N, V, wt = s.N.to_numpy(), s.V.to_numpy(), s.wt.to_numpy()
    Ns, ws = N.sum(), wt.sum()
    out = {}
    for e in ESTIMAND_NAMES:
        Y = s[f"Y_{e}"].to_numpy()
        lo = np.maximum(0.0, Y - (V - N)).sum() / Ns
        hi = np.minimum(Y, N).sum() / Ns
        p = (wt * s[f"p_{e}"].to_numpy()).sum() / ws
        out[e] = (max(lo - p, p - hi, 0.0), max(abs(p - lo), abs(p - hi)), hi - lo)
    return out


def envelope(els):
    rng = np.random.default_rng(SEED)
    rows = []
    for el in els:
        df = jm.load_joint(el)
        for dim in DIMS:
            keep = sorted({r["group"] for r in hold.holdout(df, el, dim, "A")})
            for g, t in err.stream_table(df, el, dim, keep).items():
                blocks = t.groupby("dm").indices
                keys = list(blocks)
                draws = {e: [] for e in ESTIMAND_NAMES}
                for _ in range(NBOOT):
                    pick = rng.integers(0, len(keys), len(keys))
                    rr = np.concatenate([blocks[keys[k]] for k in pick])
                    m = envelope_miss(t, rr)
                    for e in draws:
                        draws[e].append(m[e])
                point = envelope_miss(t)
                for e, v in draws.items():
                    a = np.array(v)
                    rows.append(dict(
                        election=el, dim=dim, group=g, estimand=e,
                        err_lower=float(np.sqrt((a[:, 0] ** 2).mean())),
                        err_upper=float(np.sqrt((a[:, 1] ** 2).mean())),
                        width=float(a[:, 2].mean()),
                        point_lower=point[e][0], point_upper=point[e][1]))
        print(f"  envelope {el}: done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/robustness_envelope.csv", index=False)

    print("\nError envelopes (pp): reported bar (lower) vs certified ceiling (upper)\n")
    p = R.copy()
    for c in ["err_lower", "err_upper", "width"]:
        p[c] = 100 * p[c]
    print(p.groupby("dim")[["err_lower", "err_upper", "width"]].mean().round(2).to_string())
    print("\nmean gap upper - lower: "
          f"{(p.err_upper - p.err_lower).mean():.2f}pp; max {(p.err_upper - p.err_lower).max():.2f}pp")
    by_g = p.groupby(["dim", "group"])[["err_lower", "err_upper", "width"]].mean().round(2)
    print("\nby group:\n" + by_g.to_string())
    return R


# ---------------------------------------------------------------- 4. locality: design C
def _district_size_bins(df, n=5):
    size = df.groupby("dm").V.transform("sum").to_numpy(float)
    return size, np.quantile(np.log(size), np.linspace(0, 1, n + 1)[1:-1])


def locality(els):
    """Design C: reweight the training streams to the pure strata's district-size
    distribution, then predict; and report SIGNED district-level residuals against
    district size. Design B only filters to districts holding both kinds of stream; C
    additionally forces the training and validation sets onto a common district-size
    distribution (the covariate that most separates pure from mixed streams), which is what
    'holding locality fixed' should mean when district fixed effects are infeasible for a
    24-row model over 949 districts."""
    rows, resid = [], []
    for el in els:
        df = jm.load_joint(el)
        size, cuts = _district_size_bins(df)
        sbin = np.digitize(np.log(size), cuts)
        V = df.V.to_numpy(float)
        CH = common.choices(el)
        for dim in DIMS:
            keep = sorted({r["group"] for r in hold.holdout(df, el, dim, "A")})
            pure, arg = gt.split(df, dim, hold.THR)
            # importance weights: pure-stratum share of each size bin / training share
            w_bin = {}
            for b in range(len(cuts) + 1):
                ps = V[pure & (sbin == b)].sum() / V[pure].sum()
                ms = V[~pure & (sbin == b)].sum() / V[~pure].sum()
                w_bin[b] = ps / ms if ms > 0 else 0.0
            wts = np.array([w_bin[b] for b in sbin])
            fit_df = df[~pure].reset_index(drop=True).copy()
            wf = wts[~pure]
            for c in CH + jm.JCOL + ["V"]:
                fit_df[c] = fit_df[c].to_numpy(float) * wf
            total, _ = jm.fit_joint(fit_df, el)
            Nf = jm.jcounts(fit_df).sum(axis=0)
            b_c = {e: jm.jrates(total, Nf, el, e) for e in ESTIMAND_NAMES}
            # design A predictions for comparison
            tabs_a = err.stream_table(df, el, dim, keep)
            Njoint, N4 = jm.jcounts(df), common.counts(df, dim)
            for j, g in enumerate(DIMS[dim]):
                if g not in keep:
                    continue
                m = pure & (arg == j)
                idx = [k for k, (e_, a) in enumerate(JOINT)
                       if (e_ if dim == "ethnicity" else a) == g]
                w = Njoint[m][:, idx]
                t = tabs_a[g].copy()
                for e in ESTIMAND_NAMES:
                    t[f"p_{e}"] = (w * b_c[e][idx]).sum(axis=1) / t["wt"]
                mc, ma = err.aggregate_miss(t), err.aggregate_miss(tabs_a[g])
                for e in ESTIMAND_NAMES:
                    rows.append(dict(election=el, dim=dim, group=g, estimand=e,
                                     miss_A=ma[e], miss_C=mc[e]))
                # signed district residuals (design A), against log district size
                ta = tabs_a[g]
                for dm, ix in ta.groupby("dm").indices.items():
                    s = ta.iloc[ix]
                    Ns = s.N.sum()
                    for e in ESTIMAND_NAMES:
                        Y = s[f"Y_{e}"].to_numpy()
                        L = np.maximum(0, Y - (s.V.to_numpy() - s.N.to_numpy())).sum() / Ns
                        U = np.minimum(Y, s.N.to_numpy()).sum() / Ns
                        p = (s.wt * s[f"p_{e}"]).sum() / s.wt.sum()
                        resid.append(dict(election=el, dim=dim, group=g, estimand=e, dm=dm,
                                          signed=p - (L + U) / 2,
                                          outside=max(L - p, p - U, 0) * np.sign(p - (L + U) / 2),
                                          log_size=float(np.log(size[df.dm.to_numpy() == dm][0])),
                                          n_group=Ns))
        print(f"  locality {el}: done")
    R, D = pd.DataFrame(rows), pd.DataFrame(resid)
    R.to_csv(f"{OUT}/robustness_locality.csv", index=False)
    D.to_csv(f"{OUT}/robustness_district_residuals.csv", index=False)
    core = R[~R.estimand.str.endswith("_v")]
    print("\nDesign A vs design C (district-size reweighted training), aggregate miss pp\n")
    print((100 * core.groupby("dim")[["miss_A", "miss_C"]].mean()).round(2).to_string())
    print(f"overall mean A {100*core.miss_A.mean():.2f}  C {100*core.miss_C.mean():.2f}; "
          f"max A {100*core.miss_A.max():.2f}  C {100*core.miss_C.max():.2f}")
    # does the signed residual trend with district size?
    out = []
    for (el, g, e), s in D.groupby(["election", "group", "estimand"]):
        if len(s) < 10:
            continue
        w = s.n_group.to_numpy()
        x, y = s.log_size.to_numpy(), s.signed.to_numpy()
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        slope = (w * (x - xm) * (y - ym)).sum() / (w * (x - xm) ** 2).sum()
        r = (w * (x - xm) * (y - ym)).sum() / np.sqrt((w * (x - xm) ** 2).sum() * (w * (y - ym) ** 2).sum())
        out.append(dict(election=el, group=g, estimand=e, slope_pp_per_log=100 * slope, r=r,
                        mean_signed=100 * ym, n_dm=len(s)))
    T = pd.DataFrame(out)
    T.to_csv(f"{OUT}/robustness_residual_trend.csv", index=False)
    print("\nSigned district residual vs log district size (electorate-weighted):")
    print(f"  |r| mean {T.r.abs().mean():.2f}, median {T.r.abs().median():.2f}, max {T.r.abs().max():.2f}")
    print(f"  slope pp per log-unit: mean |slope| {T.slope_pp_per_log.abs().mean():.2f}, max {T.slope_pp_per_log.abs().max():.2f}")
    print(f"  mean signed residual (pp): mean {T.mean_signed.mean():.2f}, |max| {T.mean_signed.abs().max():.2f}")
    return R, D


# ---------------------------------------------------------------- 5. split at the joint-cell level
def split_joint(els, min_share=0.015):
    """Split-sample stability for the 24 JOINT cells, under two partitions of the mixed
    streams: the compositional gradient (Malay share) and district size. Offsetting
    instability across cells can vanish in a marginal; this reports the cells themselves."""
    rows = []
    for el in els:
        df = jm.load_joint(el)
        X = jm.jshares(df)
        N_state = jm.jcounts(df).sum(axis=0)
        share = N_state / N_state.sum()
        mixed = X.max(axis=1) < hold.THR - 1e-9
        V = df.V.to_numpy(float)
        size, _ = _district_size_bins(df)
        parts = {"malay_share": common.shares(df, "ethnicity")[:, 0], "district_size": np.log(size)}
        for pname, lead in parts.items():
            order = np.argsort(lead[mixed])
            cum = np.cumsum(V[mixed][order]) / V[mixed].sum()
            med = float(np.sort(lead[mixed])[np.searchsorted(cum, 0.5)])
            rates = {}
            for name, m in {"low": mixed & (lead < med), "high": mixed & (lead >= med)}.items():
                sub = df[m].reset_index(drop=True)
                total, _ = jm.fit_joint(sub, el)
                Nh = jm.jcounts(sub).sum(axis=0)
                rates[name] = {e: jm.jrates(total, Nh, el, e) for e in ESTIMAND_NAMES}
            for k, (eth, age) in enumerate(JOINT):
                for e in ESTIMAND_NAMES:
                    rows.append(dict(election=el, partition=pname, eth=eth, age=age, estimand=e,
                                     share=share[k], low=rates["low"][e][k],
                                     high=rates["high"][e][k],
                                     gap=100 * (rates["high"][e][k] - rates["low"][e][k])))
        print(f"  split_joint {el}: done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/robustness_split_joint.csv", index=False)
    v = R[(R.share >= min_share) & R.eth.isin(["Malay", "Chinese"])]
    print(f"\nJoint-cell split gaps |high - low| (pp), Malay/Chinese cells with >= {100*min_share:.1f}% share\n")
    print(v.groupby("partition").gap.agg(mean=lambda s: s.abs().mean(), median=lambda s: s.abs().median(),
                                         p90=lambda s: s.abs().quantile(.9), max=lambda s: s.abs().max()).round(2).to_string())
    print("\nworst five:")
    print(v.reindex(v.gap.abs().sort_values(ascending=False).index).head(5)
           [["election", "partition", "eth", "age", "estimand", "low", "high", "gap"]].round(3).to_string(index=False))
    return R


# ---------------------------------------------------------------- 6. threshold frontier
def threshold_frontier(els, thresholds=(0.90, 0.95, 0.98)):
    """Everything that changes with the purity threshold, on one frontier: target width,
    districts, coverage, training size, and the miss. The threshold is a DESIGN choice that
    moves all of these at once, not a knob that varies only the target's sharpness."""
    rows = []
    for el in els:
        df = jm.load_joint(el)
        for dim in DIMS:
            keep = sorted({r["group"] for r in hold.holdout(df, el, dim, "A")})
            for thr in thresholds:
                orig = hold.design_masks
                hold.design_masks = lambda d, dm_, design, thr=None, t=thr: orig(d, dm_, design, thr=t)
                try:
                    tabs = err.stream_table(df, el, dim, keep)
                finally:
                    hold.design_masks = orig
                Xd = common.shares(df, dim)
                n_fit = int((Xd.max(axis=1) < thr - 1e-9).sum())
                for g, t in tabs.items():
                    m = envelope_miss(t)
                    for e in ESTIMAND_NAMES:
                        rows.append(dict(election=el, dim=dim, group=g, estimand=e, thr=thr,
                                         miss=m[e][0], upper=m[e][1], width=m[e][2],
                                         n_pure=len(t), n_dm=int(t.dm.nunique()), n_fit=n_fit,
                                         coverage=100 * t.N.sum() / t.group_electorate.iloc[0]))
        print(f"  frontier {el}: done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/robustness_threshold_frontier.csv", index=False)
    s = R.groupby("thr").agg(miss=("miss", "mean"), upper=("upper", "mean"), width=("width", "mean"),
                             n_dm=("n_dm", "mean"), coverage=("coverage", "mean"), n_fit=("n_fit", "mean"))
    for c in ["miss", "upper", "width"]:
        s[c] *= 100
    print("\nThreshold frontier (means over group x election x estimand)\n")
    print(s.round(2).to_string())
    return R


if __name__ == "__main__":
    els = common.arg_elections()
    print("=" * 78 + "\n1. PRIOR SENSITIVITY\n" + "=" * 78)
    prior_sensitivity(els)
    print("\n" + "=" * 78 + "\n2a. EXTRAPOLATION HOLDOUT (fit <80% pure, predict >=95%)\n" + "=" * 78)
    transport(els)
    print("\n" + "=" * 78 + "\n2b. SPLIT-SAMPLE STABILITY WITHIN MIXED STREAMS\n" + "=" * 78)
    split_stability(els)
    print("\n" + "=" * 78 + "\n3. CENSORED-MISS BRACKET\n" + "=" * 78)
    envelope(els)
    print("\n" + "=" * 78 + "\n4. LOCALITY: DESIGN C AND SIGNED DISTRICT RESIDUALS\n" + "=" * 78)
    locality(els)
    print("\n" + "=" * 78 + "\n5. SPLIT-SAMPLE STABILITY AT THE JOINT-CELL LEVEL\n" + "=" * 78)
    split_joint(els)
    print("\n" + "=" * 78 + "\n6. PURITY-THRESHOLD FRONTIER\n" + "=" * 78)
    threshold_frontier(els)


