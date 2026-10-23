"""
Seat-level near-pure check: does the model get each seat's columns right where the data can
see? Usage: python3 seatcheck.py [election ...]

The statewide holdouts (groundtruth.py, errors.py) score the headline estimands statewide.
They cannot see a column that exists in one seat only -- Berjasa in Nilai -- and they average
away seat-to-seat differences in who stood. This check works seat by seat:

  1. fit the joint model on MIXED salurans only (every stream under 85% one ethnicity);
  2. in each seat, for each column on its ballot and each group with near-pure streams there
     (>= 85% of the electorate, >= MIN_STREAMS streams), compare
        predicted = the model's group share of that column among the group's valid votes,
                    over the seat's mixed salurans
        observed  = the column's share of valid votes in the seat's near-pure streams.

Prediction and observation come from different streams of the same seat, so a gap mixes model
error with genuine within-seat context; that floor is the same for any seed, so comparing seeds
on it isolates the seed. Run under each seed in reconcile.SEEDS: the original statewide seed, the contest-aware
seed, and the candidate-party seed.

Writes output/seatcheck.csv.
"""

import numpy as np
import pandas as pd

import common
import joint as jm
import reconcile as rec

THR, MIN_STREAMS = 0.85, 3
VALID = ["BN", "PH", "PN", "BERSAMA", "OTHER"]
GROUPS = ["Malay", "Chinese"]


def run(el, seed):
    df = jm.load_joint(el)
    E = common.shares(df, "ethnicity")
    pure = E.max(axis=1) >= THR
    fit = df[~pure].reset_index(drop=True)
    per, _ = jm.fit_joint_tables(fit, el, seed=seed)
    CH = common.choices(el)
    cols = [c for c in VALID if c in CH]
    rows = []
    for s in df.cluster.unique():
        mi = (fit.cluster == s).to_numpy()
        T = per[mi].sum(axis=0)                                 # 24 x C over mixed streams
        for gi, g in enumerate(common.ETH):
            if g not in GROUPS:
                continue
            ps = (df.cluster == s).to_numpy() & (E[:, gi] >= THR)
            if ps.sum() < MIN_STREAMS:
                continue
            obs_valid = df.loc[ps, cols].to_numpy(float).sum()
            k = [i for i, (e, _) in enumerate(jm.JOINT) if e == g]
            pv = T[k][:, [CH.index(c) for c in cols]].sum()
            for c in cols:
                if df.loc[df.cluster == s, c].sum() == 0:
                    continue                                    # not on this seat's ballot
                obs = 100 * df.loc[ps, c].sum() / obs_valid
                pred = 100 * T[k, CH.index(c)].sum() / pv
                rows.append(dict(election=el, seat=s, group=g, column=c, n_pure=int(ps.sum()),
                                 observed=obs, predicted=pred, miss=pred - obs))
    return pd.DataFrame(rows)


def main():
    out = []
    for el in common.arg_elections():
        for sd in rec.SEEDS:
            out.append(run(el, sd).assign(seed=sd))
        print(f"  {el}: done")
    R = pd.concat(out, ignore_index=True)
    R.to_csv(f"{common.OUT}/seatcheck.csv", index=False)
    R["abs"] = R.miss.abs()
    pd.set_option("display.width", 200)
    print("\nmean |miss| (pp) by column, group and seed  [n = seat x column cells]\n")
    print(R.pivot_table(index=["election", "group", "column"], columns="seed", values="abs",
                        aggfunc="mean").round(1).to_string())
    print("\nworst 15 cells under each seed:")
    for sd, sub in R.groupby("seed"):
        print(f"\n[{sd}]")
        print(sub.nlargest(15, "abs")[["election", "seat", "group", "column", "n_pure",
                                       "observed", "predicted", "miss"]].round(1).to_string(index=False))
    print(f"\nwrote {common.OUT}/seatcheck.csv")


if __name__ == "__main__":
    main()
