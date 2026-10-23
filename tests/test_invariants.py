"""Invariants of the estimation pipeline. Run: uv run pytest -q

These cover the mechanical guarantees the manuscript relies on: raking matches margins and
respects structural zeros; the closed-form Duncan-Davis bound equals the transportation LP;
raking is invariant to how rows and columns are ordered; joint cells collapse to the
reported marginals; the holdout scorer is zero inside the permitted interval and the exact
distance outside; district-block resampling keeps districts whole; ratio propagation never
prints a bar below either of its inputs; and the manuscript tables carry a bar only where a
direct held-out score exists.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "estimates"))

import common            # noqa: E402
import reconcile as rec  # noqa: E402
import errors as err     # noqa: E402

OUT = common.OUT
rng = np.random.default_rng(7)


def random_tables(n=40, G=4, C=7):
    """n saluran tables with consistent margins, some rows/cols structurally zero."""
    T = rng.integers(0, 50, size=(n, G, C)).astype(float)
    T[:5, 1, :] = 0          # a group absent in some streams
    T[:, :, 2] *= (rng.random(n) > 0.3)[:, None]  # a choice absent in some streams
    return T.sum(axis=2), T.sum(axis=1)


def test_ipf_matches_margins_and_keeps_structural_zeros():
    N, Y = random_tables()
    seed = rng.random((N.shape[1], Y.shape[1])) + 0.1
    T = rec.ipf(seed, N, Y)
    assert (T >= 0).all()
    assert np.allclose(T.sum(axis=2), N, atol=1e-6)
    assert np.allclose(T.sum(axis=1), Y, atol=1e-6)
    assert (T[N == 0] == 0).all()
    assert (np.transpose(T, (0, 2, 1))[Y == 0] == 0).all()


def test_ipf_permutation_invariance():
    N, Y = random_tables()
    seed = rng.random((N.shape[1], Y.shape[1])) + 0.1
    T = rec.ipf(seed, N, Y)
    pg, pc = rng.permutation(N.shape[1]), rng.permutation(Y.shape[1])
    T2 = rec.ipf(seed[np.ix_(pg, pc)], N[:, pg], Y[:, pc])
    assert np.allclose(T2, T[:, pg][:, :, pc], atol=1e-6)


def test_frechet_bound_equals_transportation_lp():
    from scipy.optimize import linprog
    N, Y = random_tables(n=15)
    G, C = N.shape[1], Y.shape[1]
    A = np.zeros((G + C - 1, G * C))
    for g in range(G):
        A[g, g * C:(g + 1) * C] = 1
    for c in range(C - 1):
        A[G + c, c::C] = 1
    for i in range(len(N)):
        for g, c in [(0, 0), (2, 3), (1, 5)]:
            w = np.zeros((G, C)); w[g, c] = 1
            b = np.concatenate([N[i], Y[i, :-1]])
            lo = linprog(w.ravel(), A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun
            hi = -linprog(-w.ravel(), A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun
            V = N[i].sum()
            assert abs(lo - max(0, Y[i, c] - (V - N[i, g]))) < 1e-6
            assert abs(hi - min(Y[i, c], N[i, g])) < 1e-6


@pytest.mark.skipif(not os.path.exists(f"{OUT}/joint_estimates.csv"), reason="pipeline outputs absent")
def test_joint_cells_collapse_to_reported_marginals():
    J = pd.read_csv(f"{OUT}/joint_estimates.csv")
    M = pd.read_csv(f"{OUT}/joint_marginals.csv").set_index(["election", "group", "estimand"]).rate
    for (el, est), s in J.groupby(["election", "estimand"]):
        for dim, key in [("ethnicity", "eth"), ("age", "age")]:
            agg = s.groupby(key).apply(lambda d: np.average(d.rate, weights=d.registered))
            for g, r in agg.items():
                assert abs(r - M[(el, g, est)]) < 1e-9, (el, est, g)


def _stream_table():
    return pd.DataFrame(dict(dm=["a", "a", "b"], V=[100.0, 100, 100], N=[96.0, 95, 97],
                             wt=[96.0, 95, 97], group_electorate=[1000.0] * 3,
                             Y_turnout=[80.0, 70, 90], Y_BN=[40.0, 30, 50],
                             Y_PH=[20.0, 20, 20], Y_PN=[5.0, 5, 5],
                             p_turnout=[0.8] * 3, p_BN=[0.3] * 3, p_PH=[0.2] * 3, p_PN=[0.5] * 3))


def test_aggregate_miss_is_zero_inside_and_exact_outside():
    t = _stream_table()
    m = err.aggregate_miss(t)
    N, V = t.N.to_numpy(), t.V.to_numpy()
    for e, p in [("turnout", 0.8), ("BN", 0.3), ("PN", 0.5)]:
        Y = t[f"Y_{e}"].to_numpy()
        L = np.maximum(0, Y - (V - N)).sum() / N.sum()
        U = np.minimum(Y, N).sum() / N.sum()
        assert abs(m[e] - max(L - p, p - U, 0)) < 1e-12
    assert m["turnout"] == 0.0          # 0.8 sits inside its interval
    assert m["PN"] > 0.0                # 0.5 is far above the PN ceiling


def test_bootstrap_resamples_whole_districts(monkeypatch):
    monkeypatch.setattr(err, "NBOOT", 50)
    t = _stream_table()
    seen = []
    orig = err.aggregate_miss

    def spy(tab, rows=None):
        seen.append(rows)
        return orig(tab, rows)
    monkeypatch.setattr(err, "aggregate_miss", spy)
    err.bootstrap(t, np.random.default_rng(0))
    sizes = {"a": 2, "b": 1}
    for rows in seen:
        dms = t.dm.to_numpy()[rows]
        # each drawn district appears with all of its streams, the same number of times
        for d, k in sizes.items():
            assert (dms == d).sum() % k == 0


@pytest.mark.skipif(not os.path.exists(f"{OUT}/validation_errors.csv"), reason="pipeline outputs absent")
def test_ratio_propagation_never_below_inputs():
    S = pd.read_csv(f"{OUT}/validation_errors.csv")
    idx = S.set_index(["election", "group", "estimand"]).err
    marg = pd.read_csv(f"{OUT}/joint_marginals.csv").set_index(["election", "group", "estimand"]).rate
    sc = pd.read_csv(f"{OUT}/joint_scaled.csv").set_index(["election", "group", "party"]).share_of_voters
    for r in S[S.estimand.str.endswith("_v")].itertuples():
        p = r.estimand[:-2]
        t, s = marg[(r.election, r.group, "turnout")], sc[(r.election, r.group, p)]
        prop = (idx[(r.election, r.group, p)] + s * idx[(r.election, r.group, "turnout")]) / t
        assert r.err >= r.err_measured - 1e-12 and r.err >= prop - 1e-12


@pytest.mark.skipif(not os.path.exists(f"{OUT}/joint_validation_errors.csv"), reason="pipeline outputs absent")
def test_tables_carry_bars_only_where_directly_validated():
    tex = open(os.path.join(ROOT, "tex", "tab_crosstab.tex")).read()
    J = pd.read_csv(f"{OUT}/joint_validation_errors.csv")
    direct = {(e, a) for e, a in J[J.election == "jhr_se16"][["eth", "age"]].itertuples(index=False)}
    for line in tex.splitlines():
        m = re.match(r"\\quad (\w+) & (.*)\\\\$", line.strip())
        if not m:
            continue
        eth, cells = m.group(1), m.group(2).split(" & ")
        for age, c in zip(common.AGE, cells):
            assert ("textsf{d}" in c) == ((eth, age) in direct), (eth, age, c)
    res = open(os.path.join(ROOT, "tex", "tab_results.tex")).read()
    for line in res.splitlines():
        if "Indian" in line or "Other" in line:
            block_start = res.index(line)
    # Indian / Other rows carry no bar anywhere
    for line in res.splitlines():
        if line.startswith("\\multirow{3}{*}{\\textbf{Indian}}") or line.startswith("\\multirow{3}{*}{\\textbf{Other}}"):
            assert "textsf{d}" not in line


@pytest.mark.skipif(not os.path.exists(f"{OUT}/monotone_bounds.csv"), reason="pipeline outputs absent")
def test_monotone_sets_nest_inside_mob_and_match_frechet():
    M = pd.read_csv(f"{OUT}/monotone_bounds.csv")
    C = pd.read_csv(f"{OUT}/contrast_bounds.csv")
    C = C[(C.kind == "level") & (C.territory == "statewide")].set_index("claim")
    for (g, e), s in M.groupby(["group", "estimand"]):
        mob = s[s.assumption == "MOB"].iloc[0]
        f = C.loc[f"{g} {e}"]
        assert abs(mob.ge15_lo - f.ge15_lo) < 1e-9 and abs(mob.se16_hi - f.se16_hi) < 1e-9
        for r in s[~s.refuted].itertuples():
            assert r.ge15_lo >= mob.ge15_lo - 1e-9 and r.ge15_hi <= mob.ge15_hi + 1e-9
            assert r.se16_lo >= mob.se16_lo - 1e-9 and r.se16_hi <= mob.se16_hi + 1e-9


def test_pipeline_stage_order_respects_data_dependencies():
    import importlib.util
    spec = importlib.util.spec_from_file_location("est", os.path.join(ROOT, "estimates.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    names = [s for s, _, _ in mod.STAGES]
    assert names.index("robustness") < names.index("monotone")        # monotone reads robustness_split.csv
    assert names.index("contrasts") < names.index("monotone")         # monotone asserts against contrast_bounds.csv
    assert names.index("seine_input") < names.index("benchmark_seine")
    assert names.index("jointholdout") < names.index("tables")         # tables read joint_validation_errors.csv
    assert names.index("errors") < names.index("jointholdout")


def test_seine_inputs_are_generated_in_code():
    import seine_input
    d = seine_input.build("ge15")
    x = d[[f"x_{g}" for g in common.ETH]].sum(axis=1)
    y = d[[c for c in d.columns if c.startswith("y_") and c != "y_turnout"]].sum(axis=1)
    assert np.allclose(x, 1) and np.allclose(y, 1)
    assert d.lag_BN.between(0, 1).all() and d.lag_BN.notna().all()


def test_monotone_table2_against_hand_computed_fixture():
    import monotone as mo
    # three 'salurans', binary X share and outcome share, equal weights
    df = pd.DataFrame(dict(V=[100.0, 100, 100], n_g=[90.0, 50, 10], c=[80.0, 50, 20]))
    P = np.full(3, 1 / 3); X = df.n_g / 100; Y = df.c / 100
    EX = float(P @ X); EY = float(P @ Y)
    cov = float(P @ ((X - EX) * (Y - EY))); varXN = float(P @ (X - EX) ** 2)
    D_ER = cov / varXN
    m = dict(Y1_lo=float(P @ np.maximum(Y - (1 - X), 0)) / EX, Y1_hi=float(P @ np.minimum(Y, X)) / EX,
             Y1_ER=EY + D_ER * (1 - EX), Y1_NM=float(P @ (X * Y)) / EX, D_ER=D_ER)
    lo, hi, er, nm = m["Y1_lo"], m["Y1_hi"], m["Y1_ER"], m["Y1_NM"]
    assert mo.table2(m, "?", "?") == (lo, hi)
    assert mo.table2(m, "?", "+") == (max(nm, lo), hi)
    assert mo.table2(m, "+", "?") == (lo, min(er, hi))
    assert mo.table2(m, "-", "+") == (max(er, nm, lo), hi)
    assert mo.table2(m, "+", "-") == (lo, min(nm, er, hi))
    # ER is the two-group Goodman regression with intercept: slope * 1 + intercept
    A = np.column_stack([np.ones(3), X]); beta = np.linalg.lstsq(A * np.sqrt(P)[:, None], Y * np.sqrt(P), rcond=None)[0]
    assert abs((beta[0] + beta[1]) - er) < 1e-9


def test_sharp_contrast_lp_matches_brute_force_on_tiny_table():
    """BN - PN for group 0 on a 2x3 table: LP bound equals the extreme over the vertices of the
    transportation polytope, enumerated on an integer grid."""
    from scipy.optimize import linprog
    N = np.array([6.0, 4.0]); Y = np.array([5.0, 3.0, 2.0])   # rows: groups; cols: BN, PN, other
    G, C = 2, 3
    A = np.zeros((G + C - 1, G * C))
    for g in range(G): A[g, g * C:(g + 1) * C] = 1
    for c in range(C - 1): A[G + c, c::C] = 1
    W = np.zeros((G, C)); W[0, 0] = 1; W[0, 1] = -1
    b = np.concatenate([N, Y[:-1]])
    lo = linprog(W.ravel(), A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun
    hi = -linprog(-W.ravel(), A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun
    best_lo, best_hi = np.inf, -np.inf
    for t00 in range(7):
        for t01 in range(7 - t00):
            t02 = 6 - t00 - t01
            t10, t11, t12 = 5 - t00, 3 - t01, 2 - t02
            if min(t10, t11, t12) < 0: continue
            v = t00 - t01; best_lo, best_hi = min(best_lo, v), max(best_hi, v)
    assert abs(lo - best_lo) < 1e-9 and abs(hi - best_hi) < 1e-9
