"""
Benchmark our pipeline against eiPack::ei.MD.bayes on the real ground-truth holdout.
Usage: python3 benchmark.py [election]        (default jhr_se16)

Standalone: writes only its own files. Nothing in 00-10 is touched.

WHY THIS EXISTS
Rosen, Jiang, King & Tanner (2001) is the standard R x C ecological inference model, shipping
as eiPack::ei.MD.bayes, and empirical comparisons put it at or near the top. It reaches the
same guarantee we reach by raking -- bounds respected by construction -- but does so inside a
single coherent hierarchical Bayesian model rather than a two-step seed-then-reconcile
procedure. A paper proposing anything else has to say why not simply use it. This answers
that question with a number instead of an argument.

THE COMPARISON
Both methods are handed exactly the same problem: fit on the MIXED salurans, predict the
group rates, and be scored against the real observed behaviour of the held-out age-pure
salurans (groundtruth.py). Identical data, identical holdout, identical scoring.

Collapsed to 5 choice columns (BN / PH / PN / OTHER / NONVOTE) because MCMC over the full
8-column table on ~4,900 units is not worth the runtime for a comparison whose whole point is
the first three columns. OTHER absorbs BERSAMA, rejected and not-returned. Our own estimates
are recomputed on the same 5-column collapse so the two are strictly like-for-like rather
than differing by the column definition.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

import common
from common import ESTIMAND_NAMES

import models
import reconcile as rec

OUT = common.OUT
TMP = os.environ.get("BENCH_TMP", "/tmp")
THR = 0.95
# A 6x5 table needs a longer chain than the 4x5 it was first tuned on: at 20k/5k the
# two oldest rows collapsed onto the OTHER column (turnout 100%, every party 0%), a
# degenerate corner rather than a fit. Overridable for convergence checks.
ITERS = int(os.environ.get("BENCH_ITERS", 100000))
BURN = int(os.environ.get("BENCH_BURN", 30000))
THIN = int(os.environ.get("BENCH_THIN", 20))

# 5-column collapse. Both methods see exactly this.
def collapse(df, el):
    CH = common.choices(el)
    other = [c for c in CH if c not in ("BN", "PH", "PN", "NONVOTE")]
    out = pd.DataFrame({
        "c1": df["BN"], "c2": df["PH"], "c3": df["PN"],
        "c4": df[other].sum(axis=1), "c5": df["NONVOTE"],
    })
    return out, ["BN", "PH", "PN", "OTHER", "NONVOTE"]


def our_rates(fit_df, el, dim, Ycols):
    """Our pipeline (seed + rake) on the 5-column collapse."""
    X, V = common.shares(fit_df, dim), fit_df.V.to_numpy(float)
    seed = np.zeros((X.shape[1], Ycols.shape[1]))
    for k in range(Ycols.shape[1]):
        seed[:, k], _ = models.fit_link(X, Ycols[:, k].astype(float), V, "logit")
    N = common.counts(fit_df, dim)
    per = rec.ipf(seed, N, Ycols.astype(float))
    total = per.sum(axis=0)
    return total / total.sum(axis=1, keepdims=True)      # rows x 5, rows sum to 1


def main():
    el = sys.argv[1] if len(sys.argv) > 1 else "jhr_se16"
    dim = "age"                     # the dimension where the holdout has real power
    df = common.load(el)
    groups = common.DIMS[dim]

    Xd = common.shares(df, dim)
    pure = Xd.max(axis=1) >= THR - 1e-9
    arg = Xd.argmax(axis=1)
    fit_df = df[~pure].reset_index(drop=True)
    Yc, cnames = collapse(fit_df, el)
    N = common.counts(fit_df, dim)

    # ---- eiPack input: row margins, column margins, reconciling total
    ein = pd.DataFrame({f"r{j+1}": N[:, j].astype(int) for j in range(N.shape[1])})
    for k in range(Yc.shape[1]):
        ein[f"c{k+1}"] = Yc.iloc[:, k].astype(int).to_numpy()
    ein["total"] = fit_df.V.astype(int).to_numpy()
    assert (ein[[f"r{j+1}" for j in range(N.shape[1])]].sum(axis=1) == ein.total).all()
    assert (ein[[f"c{k+1}" for k in range(Yc.shape[1])]].sum(axis=1) == ein.total).all()
    os.makedirs(TMP, exist_ok=True)
    fin, fout = f"{TMP}/ei_in.csv", f"{TMP}/ei_out.csv"
    ein.to_csv(fin, index=False)
    print(f"prepared {len(ein):,} mixed salurans, {N.shape[1]}x{Yc.shape[1]} table")

    # ---- run eiPack
    r = subprocess.run(["Rscript", os.path.join(common.ROOT, "benchmark.R"), fin, fout,
                        str(N.shape[1]), str(Yc.shape[1]), str(ITERS), str(BURN), str(THIN)],
                       capture_output=True, text=True)
    # Print BOTH streams: R writes its progress to stdout and its errors to stderr, so
    # "stdout or stderr" silently hides the failure whenever the fit got as far as printing.
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print("R stderr:", r.stderr.strip()[-1500:])
    if not os.path.exists(fout):
        print("eiPack failed; aborting benchmark")
        return

    ep = pd.read_csv(fout)
    # lambda params come back named like "lambda.r1.c1"
    E = np.zeros((N.shape[1], Yc.shape[1]))
    for _, row in ep.iterrows():
        parts = str(row.param).split(".")
        ri = int([p for p in parts if p.startswith("r")][0][1:]) - 1
        ci = int([p for p in parts if p.startswith("c")][0][1:]) - 1
        E[ri, ci] = row["mean"]

    O = our_rates(fit_df, el, dim, Yc.to_numpy())

    # ---- score both against the real held-out truth
    V = df.V.to_numpy(float)
    N4 = common.counts(df, dim)
    est_map = {"BN": ["BN"], "PH": ["PH"], "PN": ["PN"],
               "turnout": [c for c in common.choices(el) if c != "NONVOTE"]}
    rows = []
    for j, g in enumerate(groups):
        m = pure & (arg == j)
        if m.sum() == 0 or N4[m, j].sum() / N4[:, j].sum() < 0.02:
            continue
        for e in ESTIMAND_NAMES:
            Y = df[est_map[e]].sum(axis=1).to_numpy(float)[m]
            lo = np.maximum(0.0, Y - (V[m] - N4[m, j])).sum() / N4[m, j].sum()
            hi = np.minimum(Y, N4[m, j]).sum() / N4[m, j].sum()
            ci = {"BN": 0, "PH": 1, "PN": 2}.get(e)
            pe = E[j, ci] if ci is not None else 1 - E[j, 4]
            po = O[j, ci] if ci is not None else 1 - O[j, 4]
            for model, p in [("eiPack_MD", pe), ("ours_reconciled", po)]:
                rows.append(dict(election=el, dim=dim, group=g, estimand=e, model=model,
                                 pred=p, truth_lo=lo, truth_hi=hi,
                                 miss=max(lo - p, p - hi, 0.0),
                                 inside=bool(lo - 1e-9 <= p <= hi + 1e-9)))
    B = pd.DataFrame(rows)
    B.to_csv(f"{OUT}/benchmark_eipack.csv", index=False)

    print("\n" + "=" * 80)
    print(f"BENCHMARK vs REAL GROUND TRUTH -- {common.cfg(el)['label']}, {dim}")
    print("both fitted on the same mixed salurans, scored on the same pure ones")
    print("=" * 80)
    p = B.pivot_table(index=["group", "estimand"], columns="model", values="miss") * 100
    p["ours_better_by"] = p.eiPack_MD - p.ours_reconciled
    print(p.round(2).to_string())
    print("\nmean |miss| pp:")
    print((B.groupby("model").miss.mean() * 100).round(2).to_string())
    print("\ninside the truth interval, %:")
    print((B.groupby("model").inside.mean() * 100).round(1).to_string())
    print(f"\nwrote {OUT}/benchmark_eipack.csv")


if __name__ == "__main__":
    main()
