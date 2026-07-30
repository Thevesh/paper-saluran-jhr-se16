"""Shared config, paths and loaders for the Johor ecological estimation.

Covers three elections on identical machinery, so the estimates are comparable across time:

    jhr_se15  Johor state election   12 Mar 2022   56 duns
    ge15      general election       19 Nov 2022   26 Johor parlimens
    jhr_se16  Johor state election   11 Jul 2026   56 duns

Every script takes an election key as its first argument and writes to out/<election>/.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit as _expit

ROOT = os.path.dirname(os.path.abspath(__file__))  # estimates/
REPO = os.path.dirname(ROOT)  # repo root
OUT = os.path.join(ROOT, "output")  # everything the pipeline writes
ROLL_DIR = (
    "/Users/theveshtheva/data/spr/voter_roll/public"  # public files downloaded from ElectionData.MY
)

# ---------------------------------------------------------------- elections
ELECTIONS = {
    "jhr_se15": dict(
        ballots=f"{REPO}/data/jhr_se15_ballots.parquet",
        stats=f"{REPO}/data/jhr_se15_stats.parquet",
        roll=f"{ROLL_DIR}/jhr_se15_2022.parquet",
        year=2022,
        bersama=False,
        seat_level="dun",
        label="SE-15 (Mar 2022)",
        short="SE-15",
    ),
    "ge15": dict(
        ballots=f"{REPO}/data/ge15_ballots.parquet",
        stats=f"{REPO}/data/ge15_stats.parquet",
        roll=f"{ROLL_DIR}/ge15_2022.parquet",
        year=2022,
        bersama=False,
        seat_level="parlimen",
        label="GE-15 (Nov 2022)",
        short="GE-15",
    ),
    "jhr_se16": dict(
        ballots=f"{REPO}/data/jhr_se16_ballots.parquet",
        stats=f"{REPO}/data/jhr_se16_stats.parquet",
        roll=f"{ROLL_DIR}/jhr_se16_2026.parquet",
        year=2026,
        bersama=True,
        seat_level="dun",
        label="SE-16 (Jul 2026)",
        short="SE-16",
    ),
}
ORDER = ["jhr_se15", "ge15", "jhr_se16"]  # chronological
STATE = "Johor"

# Demographic partitions. Each is exhaustive and mutually exclusive over the registered
# electorate of a saluran, so each defines a valid set of table rows.
ETH = ["Malay", "Chinese", "Indian", "Other"]
# Decade bands, with one institutional boundary: 60 is Malaysia's retirement age, so the
# 50-59 / 60-69 split makes the life-cycle turnout question (work vs retirement) addressable.
AGE = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
DIMS = {"ethnicity": ETH, "age": AGE}


def age_sql(band, year):
    """SQL condition selecting a band, derived from its label so bands never desync."""
    if band.endswith("+"):
        return f"{year} - birth_year >= {band[:-1]}"
    lo, hi = band.split("-")
    return f"{year} - birth_year BETWEEN {lo} AND {hi}"


PARTIES = ["BN", "PH", "PN"]


def cfg(el):
    if el not in ELECTIONS:
        raise SystemExit(f"unknown election {el!r}; choose from {list(ELECTIONS)}")
    return ELECTIONS[el]


def choices(el):
    """The choice set open to a registered voter. Exhaustive and mutually exclusive:
    these count columns sum to the saluran's registered electorate.

    NONVOTE is a choice like any other -- the demographic margins cover the whole
    electorate, so abstention is inside the table rather than outside it.

    BERSAMA is broken out for SE-16 only, where it contests alone in Johor. It did not
    exist at SE-15 or GE-15, so those elections fold it into OTHER. This only refines the
    seed; the BN/PH/PN/turnout estimands are defined identically across all three.
    """
    mid = ["BERSAMA"] if cfg(el)["bersama"] else []
    return PARTIES + mid + ["OTHER", "REJECTED", "NOTRETURNED", "NONVOTE"]


def turnout_choices(el):
    return [c for c in choices(el) if c != "NONVOTE"]


def estimands(el):
    """name -> the choice columns whose sum is the numerator."""
    return {"turnout": turnout_choices(el), "BN": ["BN"], "PH": ["PH"], "PN": ["PN"]}


ESTIMAND_NAMES = ["turnout", "BN", "PH", "PN"]


# ---------------------------------------------------------------- paths
def outdir(el):
    d = os.path.join(OUT, el)
    os.makedirs(d, exist_ok=True)
    return d


def dataset(el):
    return os.path.join(outdir(el), "saluran.parquet")


def load(el):
    p = dataset(el)
    if not os.path.exists(p):
        raise SystemExit(f"run data.py {el} first")
    return pd.read_parquet(p)


def arg_elections():
    """Election keys from argv, defaulting to all three in chronological order."""
    els = sys.argv[1:] or ORDER
    for e in els:
        cfg(e)
    return els


# ---------------------------------------------------------------- matrices
def groups(dim):
    return DIMS[dim]


def ncol(g):
    return f"n_{g}"


def shares(df, dim):
    """Saluran x group matrix of registered-electorate shares (rows sum to 1)."""
    X = df[[ncol(g) for g in groups(dim)]].to_numpy(float)
    return X / X.sum(axis=1, keepdims=True)


def counts(df, dim):
    """Saluran x group matrix of registered counts."""
    return df[[ncol(g) for g in groups(dim)]].to_numpy(float)


def outcome(df, el, estimand):
    """Saluran vector of the estimand's numerator count."""
    return df[estimands(el)[estimand]].sum(axis=1).to_numpy(float)


def logit(p):
    return np.log(p / (1 - p))


# scipy's is the numerically stable one; the naive 1/(1+exp(-x)) overflows once the
# optimiser starts pushing a group's rate towards a corner, which it does here.
expit = _expit
