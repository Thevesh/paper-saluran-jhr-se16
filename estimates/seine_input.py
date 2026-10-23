"""Build the input panels for the covariate-adjusted semiparametric benchmark (benchmark_seine.R).
Usage: python3 seine_input.py

One CSV per election (GE-15, SE-16): predictors = the four ethnic shares of the electorate;
outcomes = every choice as a share of the electorate (they sum to one); total = electorate;
covariates = log polling-district electorate, log saluran electorate, the district's outcome
shares at the PREVIOUS election (SE-15 for GE-15, GE-15 for SE-16; districts are identical
across elections so the lag is exact), and the saluran's age composition counted off the roll.

Outputs: output/seine_input_ge15.csv, output/seine_input_jhr_se16.csv
"""
import numpy as np
import pandas as pd

import common

OUT = common.OUT
PREV = {"jhr_se16": "ge15", "ge15": "jhr_se15"}


def build(el):
    df = common.load(el).copy()
    CH = common.choices(el)
    V = df.V.to_numpy(float)
    for g in common.ETH:
        df[f"x_{g}"] = df[f"n_{g}"] / V
    for c in CH:
        df[f"y_{c}"] = df[c] / V
    df["y_turnout"] = 1 - df["NONVOTE"] / V
    df["log_dm_size"] = np.log(df.groupby("dm").V.transform("sum"))
    df["log_size"] = np.log(V)
    p = common.load(PREV[el])
    lag = p.groupby("dm").agg(lag_BN=("BN", "sum"), lag_PH=("PH", "sum"), lag_PN=("PN", "sum"),
                              lag_NONVOTE=("NONVOTE", "sum"), lag_V=("V", "sum"))
    for c in ["BN", "PH", "PN", "NONVOTE"]:
        lag[f"lag_{c}"] = lag[f"lag_{c}"] / lag.lag_V
    df = df.merge(lag.drop(columns="lag_V"), left_on="dm", right_index=True, how="left")
    assert df.lag_BN.notna().all(), f"{el}: a district has no lagged outcome"
    for a in common.AGE:
        df[f"age_{a}"] = df[f"n_{a}"] / V
    cols = (["dm", "dun", "parlimen", "V"] + [f"x_{g}" for g in common.ETH] + [f"y_{c}" for c in CH]
            + ["y_turnout", "log_dm_size", "log_size", "lag_BN", "lag_PH", "lag_PN", "lag_NONVOTE"]
            + [f"age_{a}" for a in common.AGE])
    return df[cols]


def main():
    for el in ["ge15", "jhr_se16"]:
        d = build(el)
        d.to_csv(f"{OUT}/seine_input_{el}.csv", index=False)
        print(f"  {el}: {len(d)} salurans -> output/seine_input_{el}.csv")


if __name__ == "__main__":
    main()
