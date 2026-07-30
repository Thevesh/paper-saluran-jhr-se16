"""
Build the saluran-level estimation dataset. Usage: python3 data.py [election ...]

One row per saluran. Each row is a 2-way contingency table with both margins known
and the interior unobserved:

    rows    = demographic groups (ethnicity, or age), from the public voter roll
    columns = the choice set BN / PH / PN [/ BERSAMA] / OTHER / REJECTED /
              NOTRETURNED / NONVOTE, from the ballots and stats tables
    both margins sum to `voters_total`, the registered electorate

Counts are taken as exact integers off the roll and the ballots parquet rather than
reconstructed from the rounded shares in the ANALYSIS sheet of the stakeholder report.

Early (Undi Awal, /00) and postal (Undi Pos, /UP) salurans are both excluded, for
different reasons.

Postal *cannot* be used: /UP rows carry ballots but no registered electorate on the
roll, so they have no demographic margin and there is no table to build.

Early *could* be used -- /00 rows do carry both a roll electorate and real ballots --
but is excluded by choice. Early and postal voters are one bloc (police, military,
essential services and spouses), not a random slice of the electorate, and they are
demographically and behaviourally unlike the ordinary saluran they are attached to.
Leaving them in would let an unrepresentative bloc contaminate the group rates being
estimated for the general electorate. The exclusion is symmetric across the whole bloc,
which is the point: keeping early merely because it happens to have a margin, while
dropping postal because it does not, would be an artifact of data availability rather
than a decision.

Consequence: every estimate here is for the *ordinary polling-day electorate only*, and
the combined early+postal bloc is outside the scope of all of it.
"""

import duckdb
import numpy as np
import pandas as pd

import common
from common import PARTIES, STATE

KEY = ["seat", "dm", "pm", "saluran"]

# The early+postal bloc, identified off the dm code: /00 = Undi Awal, /UP = Undi Pos.
EXCLUDE_PAT = r"/(?:00|UP) "
ROLL_EXCLUDE = "dm NOT LIKE '%/UP %' AND dm NOT LIKE '%/00 %'"


def vote_columns(ballots, el):
    """Saluran x party counts. OTHER is every valid vote that is not BN/PH/PN(/BERSAMA)."""
    b = ballots.copy()
    bersama = common.cfg(el)["bersama"]
    b["bucket"] = np.where(
        b.coalition.isin(PARTIES), b.coalition,
        np.where(bersama & (b.party == "BERSAMA"), "BERSAMA", "OTHER"),
    )
    v = b.groupby(KEY + ["bucket"]).votes.sum().unstack("bucket", fill_value=0).reset_index()
    vc = [c for c in common.choices(el) if c not in ("REJECTED", "NOTRETURNED", "NONVOTE")]
    for c in vc:
        if c not in v.columns:
            v[c] = 0
    return v[KEY + vc]


def demographics(el):
    """Saluran x demographic-group registered counts, straight off the public roll.

    Ethnicity "Other" folds in Bumi Sarawak / Bumi Sabah / Orang Asli / Other, matching
    the Malay / Chinese / Indian / Other framework. Age is election year minus birth
    year (the roll carries birth_year only).
    """
    c = common.cfg(el)
    yr = c["year"]
    return duckdb.query(f"""
        SELECT
            dm, pm, CAST(saluran AS INTEGER) AS saluran,
            COUNT(*)                                                          AS roll_n,
            COUNT(*) FILTER (WHERE ethnicity = 'Malay')                       AS "n_Malay",
            COUNT(*) FILTER (WHERE ethnicity = 'Chinese')                     AS "n_Chinese",
            COUNT(*) FILTER (WHERE ethnicity = 'Indian')                      AS "n_Indian",
            COUNT(*) FILTER (WHERE ethnicity NOT IN
                             ('Malay', 'Chinese', 'Indian'))                  AS "n_Other",
            {", ".join(f'COUNT(*) FILTER (WHERE {common.age_sql(b, yr)}) AS "n_{b}"'
                       for b in common.AGE)}
        FROM read_parquet('{c["roll"]}')
        WHERE state = '{STATE}' AND {ROLL_EXCLUDE}
        GROUP BY 1, 2, 3
    """).df()


def build(el):
    c = common.cfg(el)
    CHOICES = common.choices(el)
    print(f"\n{'=' * 70}\n{el}  --  {c['label']}\n{'=' * 70}")

    ballots = pd.read_parquet(c["ballots"]).query("state == @STATE")
    stats = pd.read_parquet(c["stats"]).query("state == @STATE")

    bloc = stats.dm.str.contains(EXCLUDE_PAT, regex=True)
    early = stats.dm.str.contains("/00 ", regex=False)
    postal = stats.dm.str.contains("/UP ", regex=False)
    print(f"excluding the early+postal bloc: {int(bloc.sum())} salurans, "
          f"{stats.loc[bloc, 'ballots_issued'].sum():,} ballots")
    print(f"  early  (/00): {int(early.sum()):3d} salurans, "
          f"{stats.loc[early, 'ballots_issued'].sum():7,} ballots, "
          f"{stats.loc[early, 'voters_total'].sum():7,} registered")
    print(f"  postal (/UP): {int(postal.sum()):3d} salurans, "
          f"{stats.loc[postal, 'ballots_issued'].sum():7,} ballots, "
          f"{stats.loc[postal, 'voters_total'].sum():7,} registered (no roll margin)")
    stats = stats[~bloc].reset_index(drop=True)

    df = stats.merge(vote_columns(ballots, el), on=KEY, how="left")
    df = df.merge(demographics(el), on=["dm", "pm", "saluran"], how="left")
    assert df.roll_n.notna().all(), "saluran with no roll match"

    # The roll is the sole demographic source; the stats table is the sole electorate
    # source. They must be the same electorate or the margins do not line up.
    bad = df[df.roll_n != df.voters_total]
    assert bad.empty, f"{el}: roll/stats electorate mismatch on {len(bad)} salurans:\n{bad.head()}"

    df["V"] = df.voters_total.astype(int)
    df["REJECTED"] = df.votes_rejected.astype(int)
    df["NOTRETURNED"] = df.ballots_not_returned.astype(int)
    df["NONVOTE"] = df.V - df.ballots_issued

    assert (df[CHOICES].sum(axis=1) == df.V).all(), "choice columns do not sum to electorate"
    assert (df[CHOICES] >= 0).all().all(), "negative choice count"

    # A dm belongs to exactly one dun, so whichever geography the election does not name
    # is recovered from the roll.
    geo = duckdb.query(f"""
        SELECT DISTINCT dm, dun, parlimen FROM read_parquet('{c["roll"]}')
        WHERE state = '{STATE}' AND {ROLL_EXCLUDE}
    """).df()
    if c["seat_level"] == "dun":
        df["dun"] = df.seat
        df["parlimen"] = df.dun.map(dict(zip(geo.dun, geo.parlimen)))
    else:
        df["parlimen"] = df.seat
        df["dun"] = df.dm.map(dict(zip(geo.dm, geo.dun)))
    assert df.parlimen.notna().all() and df.dun.notna().all()

    # Clustering is on the contested seat: that is the level at which a campaign is run
    # and therefore the level at which ecological correlation lives.
    df["cluster"] = df.seat

    ncols = [c2 for c2 in df.columns if c2.startswith("n_")]
    out = df[["parlimen", "dun", "dm", "pm", "saluran", "cluster", "V"] + ncols + CHOICES]
    out = out.sort_values(["parlimen", "dun", "dm", "saluran"]).reset_index(drop=True)
    out.to_parquet(common.dataset(el), index=False)

    print(f"\nsaved {common.dataset(el)}")
    print(f"  {len(out):,} salurans, {out.cluster.nunique()} clusters "
          f"({c['seat_level']}), {out.dun.nunique()} duns, {out.parlimen.nunique()} parlimens")
    print(f"  electorate {out.V.sum():,}; turnout "
          f"{100 * (1 - out.NONVOTE.sum() / out.V.sum()):.2f}%")
    print("  statewide choice split (% of registered): " + ", ".join(
        f"{c2} {100 * out[c2].sum() / out.V.sum():.2f}" for c2 in CHOICES))
    print("  registered by group (%): " + ", ".join(
        f"{c2[2:]} {100 * out[c2].sum() / out.V.sum():.1f}" for c2 in ncols))
    # Compositional variation is what identifies an ecological model; with no spread in
    # the shares there is nothing to regress on.
    print("  share spread (sd | frac>0.9): " + ", ".join(
        f"{c2[2:]} {(out[c2] / out.V).std():.2f}|{(out[c2] / out.V > .9).mean():.2f}"
        for c2 in ncols))


if __name__ == "__main__":
    for e in common.arg_elections():
        build(e)
