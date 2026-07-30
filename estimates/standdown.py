"""PN's stand-down at SE-16: the 33 duns it contested vs the 23 it did not.

Usage: python3 standdown.py

Backs Section 5.2 of the paper. PN stood down in 23 of 56 seats and endorsed BN there, which
raises the possibility that BN's gain was a transfer arranged between coalitions rather than a
movement of opinion. This script tests that, descriptively and without re-estimating anything.

Territory is held exactly fixed: Johor's 949 polling districts are identical at GE-15 and
SE-16, so GE-15 results are re-aggregated onto SE-16 dun boundaries with no approximation.

The decisive test is the last one -- within the 33 seats PN contested, restricted to polling
districts above 90% Malay, where the accounting bounds nearly pin the answer and PN's vote is
directly observed. There BN's gain and PN's loss match at a ratio of 1.01.
"""
import sys, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))          # estimates/
REPO = os.path.dirname(ROOT)                               # repo root
sys.path.insert(0, ROOT)

import common
OUT = common.OUT

pd.set_option("display.width", 200)

# ------------------------------------------------------------------ load
se16 = pd.read_parquet(f"{OUT}/jhr_se16/saluran.parquet")
ge15 = pd.read_parquet(f"{OUT}/ge15/saluran.parquet")
se15 = pd.read_parquet(f"{OUT}/jhr_se15/saluran.parquet")

bal16 = pd.read_parquet(f"{REPO}/data/jhr_se16_ballots.parquet")
bal16 = bal16[~bal16["dm"].str.contains("/00 |/UP ")]      # drop early + postal

# ------------------------------------------------------------------ 1. contest structure
seat_coal = bal16.groupby("seat")["coalition"].apply(lambda s: set(s))
pn_seats = sorted([s for s, c in seat_coal.items() if "PN" in c])
sd_seats = sorted([s for s, c in seat_coal.items() if "PN" not in c])
assert len(pn_seats) == 33 and len(sd_seats) == 23

# third-party presence in the stood-down seats
third = (bal16[bal16.coalition == "ALONE"]
         .drop_duplicates(["seat", "candidate_uid"])
         .groupby("seat")["party"].apply(lambda s: "+".join(sorted(s))))

se16["group"] = np.where(se16.dun.isin(pn_seats), "PN contested (33)", "PN stood down (23)")
gmap = se16.drop_duplicates("dun").set_index("dun")["group"]

PARTY16 = ["BN", "PH", "PN", "BERSAMA", "OTHER"]
PARTY15 = ["BN", "PH", "PN", "OTHER"]


def agg(df, parties, by):
    """Aggregate saluran rows to `by`, returning counts + shares of valid vote."""
    cols = ["V"] + parties + ["REJECTED", "NOTRETURNED", "NONVOTE"]
    g = df.groupby(by)[cols].sum()
    g["valid"] = g[parties].sum(axis=1)
    g["ballots"] = g["valid"] + g["REJECTED"] + g["NOTRETURNED"]
    g["turnout"] = 100 * g["ballots"] / g["V"]
    for p in parties:
        g[f"{p}_sh"] = 100 * g[p] / g["valid"]
    for e in ["Malay", "Chinese", "Indian", "Other"]:
        g[f"pc_{e}"] = 100 * df.groupby(by)[f"n_{e}"].sum() / g["V"]
    return g


# ------------------------------------------------------------------ 2. like-for-like, fixed territory
# Polling districts (`dm`) are identical sets across GE-15 and SE-16 (949 each), and each
# dm maps to the same dun in both files -> exact territory match, no approximation.
common_dm = set(ge15.dm) & set(se16.dm)
assert len(common_dm) == ge15.dm.nunique() == se16.dm.nunique()
assert (ge15.drop_duplicates("dm").set_index("dm").dun
        == se16.drop_duplicates("dm").set_index("dm").dun).all()

ge15["group"] = ge15.dun.map(gmap)
se15["group"] = se15.dun.map(gmap)

G16 = agg(se16, PARTY16, "group")
G15 = agg(ge15, PARTY15, "group")
S15 = agg(se15, PARTY15, "group")
A16 = agg(se16.assign(all_="Johor"), PARTY16, "all_")
A15 = agg(ge15.assign(all_="Johor"), PARTY15, "all_")

# ------------------------------------------------------------------ 6. saluran regression
d = se16.copy()
d["mal"] = d.n_Malay / d.V
d["valid"] = d[PARTY16].sum(axis=1)
d = d[(d.valid > 0) & (d.mal >= 0.60)]
d["bn"] = 100 * d.BN / d.valid
d["sd"] = (d.group == "PN stood down (23)").astype(float)
d["mal_c"] = 100 * (d.mal - d.mal.mean())
X = np.column_stack([np.ones(len(d)), d.mal_c, d.sd, d.mal_c * d.sd])
w = d.valid.to_numpy(float)
y = d.bn.to_numpy(float)
XtW = X.T * w
beta = np.linalg.solve(XtW @ X, XtW @ y)
resid = y - X @ beta
XtWXi = np.linalg.inv(XtW @ X)
# cluster-robust on dun
meat = np.zeros((4, 4))
for _, idx in d.groupby("dun").indices.items():
    u = (X[idx] * (w[idx] * resid[idx])[:, None]).sum(0)
    meat += np.outer(u, u)
V = XtWXi @ meat @ XtWXi
se = np.sqrt(np.diag(V))
r2 = 1 - (w * resid**2).sum() / (w * (y - np.average(y, weights=w))**2).sum()

out = {
    "pn_seats": pn_seats, "sd_seats": sd_seats, "third": third,
    "G16": G16, "G15": G15, "S15": S15, "A16": A16, "A15": A15,
    "reg": (beta, se, r2, len(d), d.dun.nunique(), d.mal.mean()),
}

if __name__ == "__main__":
    print("=== stood-down seats, third candidates ===")
    print(third.reindex(sd_seats).fillna("(straight BN v PH)").to_string())
    print("\n=== SE-16 by group ===\n", G16.round(1).to_string())
    print("\n=== GE-15 by group (same territory) ===\n", G15.round(1).to_string())
    print("\n=== SE-15 by group (same territory) ===\n", S15.round(1).to_string())
    print("\n=== statewide ===\n", pd.concat([A15.round(1), A16.round(1)]).to_string())
    print("\n=== regression (Malay>=60% salurans, BN share ~ malay_c + sd + inter) ===")
    for n, b, s in zip(["const", "malay_c(pp)", "stood_down", "interaction"], beta, se):
        print(f"  {n:14s} {b:8.3f}  (se {s:.3f})  t={b/s:6.2f}")
    print(f"  R2={r2:.3f}  n_saluran={len(d)}  n_dun={d.dun.nunique()}")


# ================================================================== extras
def per_reg(g, parties):
    """Party votes as % of the *registered* electorate (fixed denominator)."""
    return pd.DataFrame({f"{p}_reg": 100 * g[p] / g["V"] for p in parties})


def block(name):
    r16, r15, r15s = G16.loc[name], G15.loc[name], S15.loc[name]
    rows = {}
    for p in ["BN", "PH", "PN"]:
        rows[p] = dict(
            se15_sh=100 * r15s[p] / r15s["valid"], ge15_sh=100 * r15[p] / r15["valid"],
            se16_sh=100 * r16[p] / r16["valid"],
            d_sh=100 * r16[p] / r16["valid"] - 100 * r15[p] / r15["valid"],
            ge15_reg=100 * r15[p] / r15["V"], se16_reg=100 * r16[p] / r16["V"],
            d_reg=100 * r16[p] / r16["V"] - 100 * r15[p] / r15["V"],
            ge15_n=r15[p], se16_n=r16[p], d_n=r16[p] - r15[p])
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    for gname in G16.index:
        print(f"\n=== {gname}: GE-15 -> SE-16 (same territory) ===")
        print(block(gname).round(1).to_string())
        r16, r15 = G16.loc[gname], G15.loc[gname]
        bn_gain = r16.BN - r15.BN
        pn_pool = r15.PN - r16.PN
        print(f"  registered: GE-15 {r15.V:,.0f} -> SE-16 {r16.V:,.0f}")
        print(f"  valid:      GE-15 {r15.valid:,.0f} -> SE-16 {r16.valid:,.0f}")
        print(f"  BN gain (votes)          {bn_gain:>10,.0f}")
        print(f"  PN votes lost (votes)    {pn_pool:>10,.0f}")
        print(f"  PN pool / BN gain        {100*pn_pool/bn_gain:>10.1f}%")
        print(f"  PH loss (votes)          {r15.PH - r16.PH:>10,.0f}")
        print(f"  turnout {r15.turnout:.1f} -> {r16.turnout:.1f} ({r16.turnout-r15.turnout:+.1f}pp)")

    # ---- 5. Malay electorate split
    print("\n=== Malay electorate split (SE-16 roll) ===")
    m = se16.groupby("group")[["n_Malay", "n_Chinese", "V"]].sum()
    m["pc_of_Malay_state"] = 100 * m.n_Malay / m.n_Malay.sum()
    m["pc_of_Chinese_state"] = 100 * m.n_Chinese / m.n_Chinese.sum()
    m["pc_of_electorate"] = 100 * m.V / m.V.sum()
    print(m.round(1).to_string())
    mg = ge15.groupby("group")[["n_Malay", "V"]].sum()
    print("GE-15 roll, Malay share in 33 contested: %.1f%%" % (100 * mg.n_Malay.iloc[0] / mg.n_Malay.sum()))

    # ---- per-seat table
    S = agg(se16, PARTY16, "dun").join(agg(ge15, PARTY15, "dun"), rsuffix="_g15")
    S["group"] = S.index.map(gmap)
    S["third"] = third.reindex(S.index).fillna("-")
    S["dBN"] = S.BN_sh - S.BN_sh_g15
    S["dPH"] = S.PH_sh - S.PH_sh_g15
    S["dPN"] = S.PN_sh - S.PN_sh_g15
    S["dTO"] = S.turnout - S.turnout_g15
    cols = ["group", "third", "V", "pc_Malay", "pc_Chinese", "turnout", "BN_sh", "PH_sh",
            "PN_sh", "BERSAMA_sh", "OTHER_sh", "BN_sh_g15", "PH_sh_g15", "PN_sh_g15",
            "turnout_g15", "dBN", "dPH", "dPN", "dTO"]
    S[cols].round(1).to_csv(f"{OUT}/pn_standdown_by_seat.csv")
    print("\nwrote out/pn_standdown_by_seat.csv")
    print(S.groupby("group")[["dBN", "dPH", "dPN", "dTO"]].mean().round(1).to_string(), "  (unweighted seat means)")
    print("\nBN gain spread (pp), by group:")
    print(S.groupby("group")["dBN"].describe().round(1).to_string())


# ---- 6b. polling-district (dm) level: change regression on fixed territory
def wls(X, y, w, cl):
    XtW = X.T * w
    b = np.linalg.solve(XtW @ X, XtW @ y)
    r = y - X @ b
    A = np.linalg.inv(XtW @ X)
    M = np.zeros((X.shape[1],) * 2)
    for _, idx in pd.Series(range(len(y))).groupby(cl.to_numpy()).indices.items():
        idx = np.asarray(idx)
        u = (X[idx] * (w[idx] * r[idx])[:, None]).sum(0)
        M += np.outer(u, u)
    V = A @ M @ A
    r2 = 1 - (w * r**2).sum() / (w * (y - np.average(y, weights=w))**2).sum()
    return b, np.sqrt(np.diag(V)), r2


def dmagg(df, parties):
    g = df.groupby("dm")[["V"] + parties + ["REJECTED", "NOTRETURNED"]
                         + [f"n_{e}" for e in ["Malay"]]].sum()
    g["valid"] = g[parties].sum(axis=1)
    g["ballots"] = g["valid"] + g["REJECTED"] + g["NOTRETURNED"]
    return g


if __name__ == "__main__":
    D16, D15 = dmagg(se16, PARTY16), dmagg(ge15, PARTY15)
    D = pd.DataFrame(index=D16.index)
    D["dun"] = se16.drop_duplicates("dm").set_index("dm").dun
    D["group"] = D.dun.map(gmap)
    D["mal"] = 100 * D16.n_Malay / D16.V
    D["bn16"] = 100 * D16.BN / D16.valid
    D["bn15"] = 100 * D15.BN / D15.valid
    D["pn15"] = 100 * D15.PN / D15.valid
    D["dbn"] = D.bn16 - D.bn15
    D["w"] = (D16.valid + D15.valid) / 2
    D["sd"] = (D.group == "PN stood down (23)").astype(float)
    D = D[(D16.valid > 0) & (D15.valid > 0)]

    for lab, sub in [("all 949 dm", D), ("Malay>=60% dm", D[D.mal >= 60])]:
        s = sub.copy()
        s["mc"] = s.mal - np.average(s.mal, weights=s.w)
        X = np.column_stack([np.ones(len(s)), s.mc, s.sd, s.mc * s.sd])
        for yv, yl in [("dbn", "CHANGE in BN share GE15->SE16"), ("bn16", "SE-16 BN share (level)")]:
            b, se_, r2 = wls(X, s[yv].to_numpy(float), s.w.to_numpy(float), s.dun)
            print(f"\n[{lab}] dep={yl}  n_dm={len(s)}  R2={r2:.3f}")
            for n, bb, ss in zip(["const", "malay_c(pp)", "stood_down", "interaction"], b, se_):
                print(f"   {n:14s} {bb:8.3f} (se {ss:.3f})  t={bb/ss:6.2f}")


# ---- 4b. near-homogeneous Malay polling districts: near-assumption-free check
if __name__ == "__main__":
    D16, D15 = dmagg(se16, PARTY16), dmagg(ge15, PARTY15)
    idx = se16.drop_duplicates("dm").set_index("dm")
    for thr in [80, 90, 95]:
        m = 100 * D16.n_Malay / D16.V
        sel = m >= thr
        a = D16[sel].join(idx[["dun"]]).assign(group=lambda x: x.dun.map(gmap))
        b = D15[sel].join(idx[["dun"]]).assign(group=lambda x: x.dun.map(gmap))
        r = []
        for g in ["PN contested (33)", "PN stood down (23)"]:
            A, B = a[a.group == g].sum(numeric_only=True), b[b.group == g].sum(numeric_only=True)
            r.append(dict(thr=thr, group=g, n_dm=(a.group == g).sum(), V16=A.V,
                          BN15=100*B.BN/B.V, BN16=100*A.BN/A.V, dBN=100*A.BN/A.V-100*B.BN/B.V,
                          PN15=100*B.PN/B.V, PN16=100*A.PN/A.V, dPN=100*A.PN/A.V-100*B.PN/B.V,
                          PH15=100*B.PH/B.V, PH16=100*A.PH/A.V, dPH=100*A.PH/A.V-100*B.PH/B.V,
                          # turnout on the paper's definition: ballots issued / registered
                          TO15=100*B.ballots/B.V, TO16=100*A.ballots/A.V,
                          dTO=100*A.ballots/A.V-100*B.ballots/B.V))
        print(f"\n=== dm >= {thr}% Malay, votes as % of REGISTERED electorate ===")
        print(pd.DataFrame(r).round(1).to_string(index=False))
