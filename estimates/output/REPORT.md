# Johor: turnout and party support by ethnicity and age, SE-15 → GE-15 → SE-16

> **Superseded — pipeline diagnostic, not the paper's estimates.** This report is produced
> by the separate-dimension model (4 ethnicity + 6 age rows) with simulation-based error
> bars, an intermediate stage the paper's validation test rejects in favour of the joint
> 24-cell model with held-out error bars (`joint_*.csv`, `validation_errors.csv`, and the
> paper's Tables 5–8). Numbers here will differ from the paper; cite the paper's tables.

Ecological estimates for three elections on identical machinery, so they are comparable
across time. All figures are **% of the group's registered voters** — the denominator is
registration, not turnout, so abstention is a modelled choice and "support" means share of
the whole group. Within a group, `turnout ≈ BN + PH + PN + OTHER + REJECTED`.

Every figure is `estimate ±error`, where the error is a **95% interval combining sampling
error and identification error in quadrature**, then clipped to the deterministic bounds.
**`*` = do not cite** (95% total error > 5pp).
93 of 120 estimates are citable.

Scope: ordinary polling-day salurans only. The early+postal bloc is excluded from all three.


### By ethnicity

| group | election | turnout | BN | PH | PN |
|---|---|---|---|---|---|
| **Malay** | SE-15 | 62.7 ±6.3* | 35.9 ±5.9* | 2.5 ±1.3 | 21.8 ±4.4 |
|  | GE-15 | 77.8 ±4.2 | 36.4 ±4.9 | 6.3 ±2.9 | 34.1 ±3.7 |
|  | SE-16 | 75.1 ±4.7 | 63.7 ±5.1* | 3.6 ±1.9 | 6.8 ±2.4 |
| | | | | | |
| **Chinese** | SE-15 | 44.5 ±5.4* | 5.6 ±4.5 | 31.7 ±4.8 | 1.1 ±3.4 |
|  | GE-15 | 69.6 ±4.6 | 5.8 ±4.2 | 62.6 ±3.9 | 0.3 ±1.7 |
|  | SE-16 | 62.3 ±4.7 | 7.7 ±4.5 | 52.5 ±3.2 | 0.0 ±1.4 |
| | | | | | |
| **Indian** | SE-15 | 34.0 ±18.3* | 11.1 ±15.9* | 12.1 ±8.5* | 2.2 ±9.7* |
|  | GE-15 | 68.7 ±10.6* | 5.6 ±11.5* | 62.3 ±11.6* | 0.0 ±6.9* |
|  | SE-16 | 53.8 ±13.7* | 26.1 ±13.4* | 17.1 ±6.6* | 0.0 ±4.0 |
| | | | | | |
| **Other** | SE-15 | 37.6 ±21.8* | 15.3 ±18.1* | 2.9 ±3.5 | 15.1 ±16.5* |
|  | GE-15 | 64.5 ±15.7* | 9.6 ±16.1* | 17.1 ±11.2* | 37.4 ±29.1* |
|  | SE-16 | 60.0 ±14.6* | 46.5 ±15.0* | 5.8 ±4.6 | 0.0 ±3.7 |

### By age

| group | election | turnout | BN | PH | PN |
|---|---|---|---|---|---|
| **18-29** | SE-15 | 51.4 ±3.9 | 21.3 ±3.8 | 11.5 ±2.8 | 14.2 ±2.5 |
|  | GE-15 | 74.1 ±3.7 | 17.7 ±3.7 | 30.3 ±5.3* | 25.6 ±4.4 |
|  | SE-16 | 66.0 ±4.9 | 42.8 ±5.0* | 17.8 ±4.0 | 3.5 ±1.4 |
| | | | | | |
| **30-39** | SE-15 | 46.5 ±4.0 | 19.9 ±3.8 | 9.7 ±2.5 | 13.8 ±2.5 |
|  | GE-15 | 69.9 ±4.2 | 20.4 ±3.9 | 26.1 ±5.9* | 22.8 ±4.7 |
|  | SE-16 | 65.6 ±4.6 | 41.9 ±4.5 | 18.2 ±3.8 | 3.7 ±1.4 |
| | | | | | |
| **40-49** | SE-15 | 52.0 ±2.4 | 20.0 ±2.2 | 14.2 ±2.5 | 13.5 ±1.7 |
|  | GE-15 | 75.0 ±1.5 | 20.9 ±3.1 | 34.1 ±5.0 | 19.2 ±2.9 |
|  | SE-16 | 68.3 ±1.4 | 40.8 ±2.9 | 20.7 ±2.5 | 4.6 ±1.3 |
| | | | | | |
| **50-59** | SE-15 | 62.3 ±1.9 | 24.7 ±2.2 | 19.1 ±3.0 | 12.8 ±1.5 |
|  | GE-15 | 80.2 ±1.2 | 25.4 ±2.9 | 37.4 ±4.1 | 16.2 ±2.6 |
|  | SE-16 | 74.6 ±1.3 | 38.5 ±3.0 | 29.0 ±2.7 | 4.3 ±1.5 |
| | | | | | |
| **60-69** | SE-15 | 65.3 ±1.7 | 29.0 ±3.1 | 21.4 ±3.2 | 9.7 ±1.3 |
|  | GE-15 | 80.5 ±1.2 | 31.4 ±4.1 | 37.0 ±4.4 | 10.7 ±1.8 |
|  | SE-16 | 79.3 ±1.0 | 41.0 ±3.2 | 32.0 ±3.2 | 3.3 ±1.2 |
| | | | | | |
| **70+** | SE-15 | 46.5 ±2.7 | 24.2 ±3.3 | 13.7 ±2.3 | 4.7 ±0.9 |
|  | GE-15 | 58.0 ±2.7 | 26.8 ±4.0 | 24.8 ±3.7 | 4.4 ±1.6 |
|  | SE-16 | 59.7 ±2.4 | 30.4 ±3.7 | 24.8 ±2.9 | 1.8 ±0.7 |

## What the numbers say

- **Turnout collapsed at SE-15 and never fully came back.** Malay turnout ran 63 → 78 → 75% and Chinese 45 → 70 → 62%. The SE-15/GE-15 gap is the striking part: the same electorate, eight months apart, turned out 15pp (Malay) and 25pp (Chinese) higher for the general election than for their own state election. The Chinese SE-15 figure — 45% of registered Chinese casting any ballot — is the single lowest participation number in the series.

- **BN's SE-16 surge is a PN collapse, almost one-for-one.** Malay BN support sat flat at 36 and 36% before jumping to 64%, a +27pp move. Over the same step Malay PN support fell from 34% to 7%, −27pp. The two moves match in size and sign, and no other Malay column moves enough to account for either.

- **Chinese support did not realign, it demobilised.** Chinese PH ran 32 → 63 → 53%, and Chinese BN barely moved across all three (6 → 6 → 8%). PH's Chinese losses since GE-15 track Chinese turnout down (70 → 62%) rather than crossing to anyone: on these estimates the votes stayed home rather than switching.

- **The age gradient in BN support inverted.** At GE-15, BN support *rose* with age (18% among 18-29 against 27% among 70+, a -9pp gap). At SE-16 it *falls* with age (43% against 30%, +12pp) — a swing of roughly 22pp in the gradient itself, larger than the error on any cell involved. BN's recovery was led by the youngest voters, which is the reverse of the conventional reading. This is the finding in the series most worth a second look before it is published.

- **Turnout stays hump-shaped in age at every election.** The 60-69 band turns out at 65/81/79% against 51/74/66% for 18-29 and 46/58/60% for 70+. With 60 the retirement age, the 50-59 vs 60-69 comparison (75 vs 79% at SE-16) speaks to the work-vs-retirement question directly.

- **Indian and Other are all starred and none of the above rests on them.** Their point estimates swing wildly across elections (Indian turnout 34 → 69 → 54%), which is a signature of an unidentified parameter rather than a volatile electorate.


## Why the error bars are what they are

Two independent errors, combined as `total_se = sqrt(se_bootstrap² + rmse_worldB²)`:

- **sampling error** — cluster bootstrap over contested seats. Small here (~0.2–2pp).
- **identification error** — the error from the constant-rate assumption being wrong.
  Measured by planting a known truth in a simulated world where composition drives
  behaviour, then seeing how far the pipeline lands from it. This is the dominant term,
  and it does **not** shrink with electorate size.

A regression table reporting only the first would look far more precise than the estimates
actually are. See `README.md` for the full method.

## The stars

`*` marks a 95% total error above 5pp. It falls almost entirely on **Indian** and **Other**:
no Johor saluran is more than ~24% Indian or ~10% Other, so the model extrapolates well
outside its support and their errors run ±11–14pp on average.

The star keys on the measured error alone, *not* on the width of the deterministic bounds.
That distinction matters. Chinese turnout has bounds 55pp wide but a total error of ±4.8pp —
the bounds are a worst case over every conceivable behaviour, and the actual composition of
Johor's salurans resolves nearly all of that slack. Starring on bound width would discard a
good estimate on the strength of a hypothetical.

5pp is a convention, not a fact. Read `error` itself where the distinction matters — a 5.1pp
cell and a 4.9pp cell differ by nothing but the threshold.

A starred estimate is still the best available guess and is still guaranteed inside its
arithmetic bounds. It simply has an error bar too wide to carry an argument.
