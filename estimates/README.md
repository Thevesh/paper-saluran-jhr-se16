# Johor — ecological estimation of turnout and party support by ethnicity and age

Three elections on identical machinery, so the estimates are comparable across time:

| key | election | polled | seats | salurans | electorate |
|---|---|---|---|---|---|
| `jhr_se15` | SE-15 | 12 Mar 2022 | 56 duns | 4,638 | 2,574,835 |
| `ge15` | GE-15 | 19 Nov 2022 | 26 parlimens | 4,695 | 2,593,562 |
| `jhr_se16` | SE-16 | 11 Jul 2026 | 56 duns | 4,889 | 2,703,175 |

Estimates, for each of Malay / Chinese / Indian / Other and 18-29 / 30-39 / 40-49 / 50-59 /
60-69 / 70+:

- voter turnout
- support for BN, PH, PN

The preferred model estimates the **joint** ethnicity-by-age crosstab; the marginals above are
sums of its cells. The roll carries both attributes for the same individual, so the joint
margin is *counted*, not modelled — which turns out to matter far more than the choice of
estimator. Every estimate is constrained to be arithmetically possible given what each
saluran's composition and result already imply.

## Run

From the repository root, one command reproduces every number, table and figure:

```bash
python3 estimates.py                 # all but the two slow stages
python3 estimates.py --all           # including them
python3 estimates.py --figures       # also regenerate tex/dataviz/
python3 estimates.py --from joint    # resume partway
python3 estimates.py --list          # show the stages
```

Two stages are skipped unless `--all` is passed. `data.py` rebuilds the saluran panel from the
external voter roll (~10 min; set `ROLL_DIR` in `common.py`). `benchmark.py` fits the eiPack
hierarchical Bayesian comparison via `benchmark.R` (~1.5 h; needs R with eiPack installed).
Everything else runs in a few minutes.

Stages are plain modules and can be run alone. Each takes election keys and defaults to all
three, chronologically: `cd estimates && python3 joint.py jhr_se16`.

| module | what it does |
|---|---|
| `common.py` | shared config, paths, loaders. The group definitions live here |
| `data.py` | builds the saluran panel from the roll and the ballot files |
| `bounds.py` | deterministic Duncan–Davis bounds: what arithmetic alone permits |
| `models.py` | the candidate estimators — LPM, logit, probit, context, within-seat |
| `reconcile.py` | rakes the estimates onto each saluran's two known margins |
| `validate.py` | the simulation harness the estimators are scored against |
| `bivariate.py` | the two-group scatter estimate, and why it *is* Goodman |
| `series.py` | aligns estimates across the three elections |
| `scaled.py` | support as a share of voters rather than of registrants |
| `groundtruth.py` | the near-pure saluran holdout — the paper's central validation |
| `joint.py` | the joint ethnicity-by-age model |
| `joint_estimates.py` | its crosstab, marginals and scaled panel — the reported estimates |
| `standdown.py` | the three PN stand-down checks |
| `holdout.py` | unrestricted (A) vs locality-fixed (B) holdout designs |
| `kappa.py` | the bounded-heterogeneity frontier: what bounds-only reporting would cost |
| `errors.py` | the reported error bars, and the purity-threshold sensitivity |
| `benchmark.py` | the eiPack hierarchical Bayesian comparison (`benchmark.R`) |
| `tables.py` | writes `tex/tab_results`, `tab_crosstab`, `tab_errors` |
| `workbook.py` | the per-election estimate workbooks in `output/` |
| `report.py` | the HTML report in `output/` |

Everything the pipeline writes lands in `output/`: cross-election results at the top level,
per-election intermediates in `output/<election>/`, and the HTML and markdown reports. The path
is defined once, as `common.OUT`.

Figures live outside this directory: `dataviz.py` at the repository root reads what the
pipeline writes to `estimates/output/` and renders every figure into `tex/dataviz/`.

## The estimand

For group *g*:

```
b_g = (registered voters in g who did X) / (registered voters in g)
```

The denominator is **registration, not turnout**. The roll's ethnic and age margins describe
the whole electorate, so abstention is a choice inside the model rather than a filter applied
before it. "BN support among Chinese = 6.9%" therefore means *6.9% of all registered Chinese
cast a BN ballot* — not 6.9% of Chinese who voted. The two differ a lot when turnout is 61%.
Turnout and the three party numbers are all shares of the same denominator, so within a group
`turnout ≈ BN + PH + PN + BERSAMA + OTHER + REJECTED + NOTRETURNED`. `scaled.py` produces the
conventional share-of-voters panel alongside.

## Data

| | |
|---|---|
| unit | saluran — 14,222 across the three elections |
| demographics | public voter roll, exact counts (not the rounded ANALYSIS shares) |
| votes | `data/<election>_ballots.parquet` + `<election>_stats.parquet` |

Each saluran is a 2-way table with **both margins known and the interior unobserved**:

- rows: demographic groups, from the roll
- columns: `BN / PH / PN [/ BERSAMA] / OTHER / REJECTED / NOTRETURNED / NONVOTE`
- both margins sum to the saluran's registered electorate

BERSAMA is broken out for SE-16 only, where it contests alone in Johor; it did not exist at
SE-15 or GE-15, which fold it into OTHER. This only refines the seed — the BN/PH/PN/turnout
estimands are defined identically across all three.

**The early + postal bloc is excluded from every election:**

| election | early (`/00`) | postal (`/UP`) | ballots dropped |
|---|---|---|---|
| SE-15 | 82 salurans, 18,564 ballots | 56 salurans, 36,461 ballots | 55,025 |
| GE-15 | 86 salurans, 18,263 ballots | 26 salurans, 44,454 ballots | 62,717 |
| SE-16 | 91 salurans, 19,471 ballots | 56 salurans, 24,677 ballots | 44,148 |

Postal *cannot* be used: it has ballots but no registered electorate on the roll, so there is
no demographic margin and no table to build. Early *could* be used, but is excluded by choice —
early and postal voters are one bloc (police, military, essential services and spouses), not a
random slice, and they are demographically and behaviourally unlike the ordinary saluran they
are attached to. Keeping early merely because it happens to have a margin, while dropping
postal because it does not, would be an artifact of data availability rather than a decision.

**Every estimate here is therefore for the ordinary polling-day electorate only** — under 3.5%
of ballots excluded.

Verified before anything is estimated, for every election: `ballots_issued = valid + rejected +
not_returned`, `valid = Σ candidate votes`, `roll count = voters_total` for every saluran, and
the choice columns sum to the electorate.

## Method

### 1. Bounds — the ecology constraint (`bounds.py`)

For a saluran with electorate *V*, group size *N*, outcome total *Y*, the group's contribution
*y* is pinned by accounting alone:

```
max(0, Y − (V − N))  ≤  y  ≤  min(Y, N)
```

These are the Fréchet–Hoeffding bounds (Duncan & Davis 1953) and they are **sharp** — both ends
are attained by a feasible table, so nothing tightens them without an assumption. Collapsing the
other parties into one "not this outcome" column loses nothing: the sharp bound on a cell of the
full R×C table is the same expression.

Summing over salurans gives statewide bounds far tighter than treating the state as one
aggregate, because each saluran contributes its own accounting. They remain wide: Malay BN
support at SE-16 is bounded to [34.3, 72.0] before any modelling.

### 2. Models (`models.py`)

All share the identity that makes ecological inference possible — a saluran's rate is the
composition-weighted average of its groups' rates:

```
p_i = Σ_g x_ig · b_ig
```

| model | parameterisation | fit |
|---|---|---|
| M1 Goodman LPM | `b_ig = β_g` | WLS, weights *V_i* |
| M2 ecological logit | `b_ig = expit(β_g)` | binomial ML |
| M3 ecological probit | `b_ig = Φ(β_g)` | binomial ML |
| M4 logit + context | `b_ig = expit(β_g + γ_g z_i)` | binomial ML |
| M5 logit, within-seat | M2 inside each contested seat, pooled | binomial ML |

**The link is applied to each group's rate, not to the saluran's linear predictor.** `p_i` stays
linear in the group rates whichever link is used, because the mixture weights `x_ig` are data. So
M2 and M3 are reparameterisations of M1 that force rates into (0,1) — they can differ from M1
only where M1 wants to leave the unit interval, and from each other essentially not at all. This
is borne out: **logit and probit agree to 0.030pp**. Running all three is a check on whether the
constraint binds, not a menu of rival theories.

Fitting `expit(Σ_g x_ig β_g)` — the link on the linear predictor — would be a *different and
wrong* model here: its coefficients are not group rates and it decomposes nothing. It is not
fitted.

A weakly-informative `N(0, 2.5²)` prior (Gelman et al. 2008) is placed on each β. PN is
near-absent among non-Malays, so the unpenalised likelihood is maximised by driving those β to
−∞ — a corner solution reporting exactly 0.000% with an infinite coefficient and a meaningless
standard error. Against 2.7m individuals the prior is negligible wherever the data identify
anything; it only bites at corners.

### 3. Imposing the constraint (`reconcile.py`)

The obvious approach — fit, then clip the statewide estimate into `[lo, hi]` — is weak on three
counts. It is slack almost everywhere, because a statewide bound averages saluran bounds, so an
estimate can sit inside it while implying impossible counts in hundreds of individual salurans.
It breaks the accounting, because clipping one group rebalances nothing. And it only binds when
the model is already badly wrong.

The better route uses a fact about the bounds: **they are not an extra assumption, they are
exactly "cells are non-negative and both margins are respected."** The Fréchet bounds are a
*consequence* of the margins. So imposing the margins on the interior at saluran level imposes
the bounds automatically, everywhere, by construction.

Each saluran's group×choice table is raked (IPF) to its two known margins from a
model-supplied seed. IPF converges to the unique table matching both margins while preserving
the seed's odds ratios (Deming & Stephan 1940; Ireland & Kullback 1968). This is
structure-preserving estimation (SPREE; Purcell & Kish 1980), the standard small-area tool for
exactly this shape of problem: known margins, borrowed interior.

The division of labour is the attractive part, and it is automatic:

- In a **near-homogeneous** saluran the margins nearly pin the interior, the bounds are nearly
  a point, and the seed barely matters. **The data speak.**
- In a **balanced** saluran the margins say little and the seed carries the answer. **The model
  speaks.**

There is a second payoff. Multiply each group's estimate by its registered count and add up —
the reconciled estimates reproduce the actual statewide totals *exactly*, every ballot
accounted for, because the margins were imposed. Unconstrained M2 does not, implying over a
million phantom votes on the ethnic model. An estimate that does not reproduce the election it
is decomposing is not a decomposition.

**What raking buys is coherence, not accuracy.** Scored against real held-out voters
(`groundtruth.py`), the full apparatus lands almost exactly where Goodman's 1953 regression
lands. Non-negative cells, every saluran's margins reproduced, statewide totals exact — all
worth having, and the honest reason to prefer this to the LPM. But any claim that reconciliation
makes the estimates *closer to the truth* is not supported by the data here.

### 4. Validation against observed behaviour (`groundtruth.py`, `holdout.py`)

This is the paper's central move, and it is possible only because the Election Commission
sorts voters into streams by age. Many salurans are therefore near-pure in a single band, and
in a 95%-pure stream the margins nearly pin the dominant group's rate down. So: **fit on
demographically mixed salurans only, predict the near-pure ones, and compare the prediction
with what those streams actually did.**

The test changed the paper. Estimating age separately from ethnicity fails badly — a mean miss
of 4.93 points across the held-out age outcomes, and up to 19 points on BN support among the
over-70s — because Johor's age-pure streams are disproportionately Chinese. Using the exact
ethnicity-by-age margins instead cuts the mean age-cell miss to **0.98**. The estimator was
never the problem; the row definition was. A hierarchical Bayesian R×C model (eiPack) scores
5.49 against our 4.88 on the same specification: indistinguishable, and failing in the same
places.

`holdout.py` runs this two ways. **Design A** fits on every mixed saluran and predicts every
pure one. **Design B** restricts both sides to polling districts containing both kinds, holding
locality roughly fixed. B returns smaller misses, so A is the conservative choice and the one
reported.

### 5. The error bars (`errors.py`)

Every reported `±` is a **measured held-out validation error**, specific to one group, one
election *and* one estimand — Malay BN support is scored against held-out Malay BN behaviour,
turnout against held-out turnout. Three things make it what it is:

- **Measured at the scale of the estimand.** The miss against an individual stream averages
  5.80 points, against a statewide rate 1.37 — a factor of four. That gap is variation between
  streams, which a statewide rate averages over. Reporting the first as the second would be the
  mistake of quoting a regression's residual standard deviation as the standard error of its
  mean.
- **With the luck of the validation set priced in.** Polling districts are resampled with
  replacement within the held-out stratum, the aggregate miss recomputed in each of 2,000
  replicates, and the root mean square reported across them.
- **Derived quantities inherit, not re-score.** The share-of-voters panel is a ratio, whose
  admissible interval is about twice as wide as its components' and whose two component errors
  can cancel. Scoring it directly gave Malay BN a miss of *zero* at SE-16 while the model was
  demonstrably wrong on both turnout and BN votes. So it inherits `(E_party + s·E_turnout)/t`
  instead, and keeps the direct measurement only where that is larger.

These are typical errors, **not confidence intervals**, and carry no coverage claim. For
changes between elections the two endpoint errors are **added**, not combined in quadrature:
adding assumes the worst case, whereas quadrature assumes an independence we have no grounds to
assert across elections sharing the same geography, assignment rule and estimator.

The purity threshold is a design choice, so `errors.py` also reports the aggregate miss at
0.90, 0.95 and 0.98: **0.55, 1.22 and 1.67 points**. A stricter threshold narrows the interval
the margins permit, so the same prediction error is more often caught outside it — the measured
error rises because the test sharpens, not because the model degrades.

**Indian and Other voters have no near-pure saluran at any threshold**, so no validation is
possible. They carry no error bar, are greyed in every table, and should not be cited. This is
a statement about identification, not a precision cutoff: Johor's residential geography never
concentrates either group into a stream. Their bounds span 75 and 78 points, and their apparent
swings across elections (Indian turnout 35 → 69 → 55%) are the signature of an unidentified
parameter, not a volatile electorate.

### Why there are no significance tests

Nothing was sampled — the margins are counts of the whole electorate — so the uncertainty that
comes from observing only part of it is absent by construction. What constrains these estimates
is *identification*, not variance: being wrong about the behavioural assumption produces bias,
and bias does not shrink with 2.7 million voters. Indian PN support at SE-16 has a bootstrap
standard error under three points and deterministic bounds running from 0.3% to 96.5%. A
procedure that certifies that estimate as precise is not a filter but a rubber stamp.

`kappa.py` reports the assumption-free alternative: relax constancy to a band of radius κ on
the logit scale and compute what the margins then allow. Two results come out of it. Constancy
is not merely doubtful but **arithmetically impossible** — no single rate per group fits every
stream's margins at once. And at the narrowest feasible band the bounds are barely tighter than
assuming nothing at all, and cannot even sign the headline Malay swing.

## The scatterplot version (`bivariate.py`)

The natural quick-and-dirty approach: scatter PH's vote share against the Chinese share of the
electorate, fit a line, read it off at x = 100%. **That is not an approximation of M1 — it is
M1.**

Fit `y = a + b·x`. Read off at the ends: `x=1 → a+b`, `x=0 → a`. Now write the two-group
Goodman regression, no intercept, shares summing to 1:

```
y = B_chi·x + B_non·(1−x) = B_non + (B_chi − B_non)·x
```

Matching terms: `a = B_non`, `b = B_chi − B_non`, so **`a + b = B_chi`**. The extrapolation to
100% and the Goodman coefficient are the same number, algebraically, and this is verified
numerically to 4.7e-15. Reading a fitted line off at its endpoint *is* ecological regression;
Goodman just writes it in the form that generalises.

What the full model adds is exactly two things. *One — fit all groups at once, not "group vs
everyone else",* which otherwise forces groups that behave oppositely to share one coefficient.
*Two — the link stops the line running off the edge of the world.* A straight line is free to
predict a negative rate, and does. A rate below 0% is not a small error; it is an impossible
answer.

The pattern in the failures is the point. Where a group has salurans near its own extreme, the
quick version is fine — for the age bands and for Malay/Chinese it lands within a few points of
the full model. Where it does not, it fails catastrophically: no Johor saluran is more than
~35% Indian, so "read off at x = 100%" extrapolates far beyond any data that exists.

## Limitations — read before quoting any number

- **Indian and Other should not be quoted at all.** See above. Nothing in the findings rests on
  them.
- **The error bars measure transport into near-pure streams.** We observe how well a model
  fitted on mixed streams predicts near-pure ones, and report it as the error of an estimate
  covering an electorate mostly in streams that are neither. In a near-pure stream the margins
  nearly pin the rate down; in a mixed stream the model does more of the work. The place we can
  check is the place the method finds easiest, and this is the assumption most likely to flatter
  us.
- **The estimand is a group aggregate, not an individual transition.** A one-for-one net
  movement between two coalitions is consistent with offsetting churn underneath. These
  estimates do not license individual-level claims about any voter, and the ecological fallacy
  is not repealed by a narrow error bar.
- **Nothing here identifies *why* groups moved.** The readings in the paper are conjecture
  disciplined by the estimated magnitudes.
- **All results are for the ordinary polling-day electorate**, excluding early and postal.
- **Turnout is off the roll**, not off SPR's own turnout figures: `ballots_issued /
  voters_total` per saluran, which is what the margins require.
- **SE-15 and GE-15 use the same roll year (2022) but different age bases**, both computed as
  election year minus birth year. The eight-month gap is inside the rounding.
- **Johor's electoral geography is unusually favourable to EI**, with extreme Malay–Chinese
  compositional variation across ~4,900 small units. A more evenly mixed state would give wider
  bounds and weaker identification.

## References

- Duncan & Davis (1953), *An alternative to ecological correlation*, ASR 18(6).
- Robinson (1950), *Ecological correlations and the behavior of individuals*, ASR 15(3).
- Goodman (1953), *Ecological regressions and behavior of individuals*, ASR 18(6).
- Deming & Stephan (1940), *On a least squares adjustment of a sampled frequency table*, AMS 11(4).
- Ireland & Kullback (1968), *Contingency tables with given marginals*, Biometrika 55(1).
- Purcell & Kish (1980), *Postcensal estimates for local areas*, International Statistical Review 48.
- King (1997), *A Solution to the Ecological Inference Problem*, Princeton.
- Rosen, Jiang, King & Tanner (2001), *Bayesian and frequentist inference for ecological inference*, Statistica Neerlandica 55(2).
- Gelman, Jakulin, Pittau & Su (2008), *A weakly informative default prior*, AOAS 2(4).
- Manski (1995), *Identification Problems in the Social Sciences*, Harvard.
