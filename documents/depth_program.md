# Depth program — squeeze every component of every model to its limit

Opened 2026-07-31 on Samuel's directive: *improve every single aspect of each
model, step by step, market-blind, until the theoretical limit of prediction.*

## Why this exists

Past campaigns were **breadth**: bolt a NEW channel onto a frozen blend, screen
it, record a null (~60 nulls across three ledgers). What was never done is
**depth**: re-examining the components we already ship. Every league's model is
a linear logistic blend over features whose own hyperparameters were tuned once,
greedily, one at a time, and never revisited jointly. Interactions, nonlinear
forms, model class, calibration and shrinkage are all untouched.

Rules unchanged: market-blind (no odds in any feature), DEV screens only,
one TEST look per adopted change, ledger row either way.

## The axis grid (each cell = one screenable step)

| # | Axis | MLB | NFL | NHL |
|---|---|---|---|---|
| A | Core rating hyperparameters, **jointly** re-tuned (not greedily) | FIP/xFIP-Elo k, HFA, season regression | v7 β, τ, draw band, leverage weights | Elo k/ha/regress **with** xG-Elo K/HA/regress |
| B | Player-rating engine internals | per-PA TrueSkill β, τ, draw, priors | QB Kalman R, opp-quality k, σ-discount | (n/a — no shipped player feature) |
| C | Blend model class on the SAME features | logistic vs GBM vs stack | same | same |
| D | Feature functional form | splines/bins on continuous features | same | rest/B2B nonlinearity |
| E | Interactions | starter×bullpen, Elo×rest, park×power | QB×opponent-D, rest×travel | xG×Elo, rest×B2B |
| F | Regularization / shrinkage | blend C, per-feature priors | C, recency half-life | C, feature priors |
| G | Calibration | Platt vs isotonic, per-tier, drift | same | same |
| H | Training protocol | recency weighting, walk-forward cadence | half-life (currently 3) | warm-start window |
| I | Ensembling | tier-average, seed-average | same | same |

## Status

Round 1 launched 2026-07-31 (axes A, C, D, E, F per league; G-I in round 2).
Results land in `data/{mlb,nfl,nhl}_depth_dev.json` and are summarized here as
each round closes. A cell is CLOSED only when a DEV screen exists for it —
"we already tuned that once" is not a closure.

## Closure log

### Round 1 — 2026-07-31. Axes A-F × 3 leagues. **0 adoptions, 0 TEST looks spent.**

Every candidate was measured twice: the raw best-cell number, and a **nested**
estimate where the winning cell is re-chosen without seeing the fold it is
scored on. The gap between them is search optimism, and it was brutal.

| league | best candidate | raw | nested | optimism | pre-registered gate | verdict |
|---|---|---|---|---|---|---|
| MLB | TrueSkill read-time low-sample prior (n0≈200) | +0.00026 | **+0.00024** | 9% | +0.0020 | sub-gate, NOT adopted |
| MLB | A+B+F combined | +0.00065 | ≈+0.00051 | — | +0.0020 | sub-gate |
| NFL | v7 engine knobs, joint | +0.00497 | **+0.00092** | **82%** | 0.0020 | sub-gate, NOT adopted |
| NHL | 6-parameter joint core re-tune | +0.00020 | **−0.00007** | **130%** | +0.0005 | NULL |

**The premise of this round was half right.** For NFL, greedy one-at-a-time
re-tuning is worth *less than nothing* (nested −0.00062) while only the joint
search produces a positive honest estimate (+0.00092) — the mechanism is
visible: `lev_exp=0` is worse at the shipped anchor yet belongs to the joint
optimum, exactly the blind spot greedy tuning has. For NHL the premise is
simply **false**: greedy and joint differ by 0.6 SE, and 4,973 of 21,000 grid
points sit within 1 SE of the best. The shipped point was already on the
plateau.

**Three structural findings worth more than the numbers:**

1. **A rating's home-field parameter is a free parameter** when a logistic with
   an intercept sits downstream — the intercept absorbs a constant logit shift
   exactly. Same for any knob that merely scales the logit (`beta`, `k`). This
   is why decades of "Elo tuning" left nothing on the table here, and it means
   those knobs can only ever matter through their effect on the *online update*.
2. **MLB TrueSkill mu is read with no low-sample shrinkage** — a debut batter
   enters the lineup average at full weight. Adding `mu ← mu0 + (mu−mu0)·n/(n+n0)`
   at read time is SIG at every n0 from 25 to 400 with a smooth plateau. Real,
   mechanistically sound, and still 8× below the gate. Logged as a latent design
   gap, not adopted.
3. **The joint optima SIMPLIFY the engines.** NFL's best config turns *off* both
   leverage weighting and opponent-quality scaling. Complexity added in earlier
   campaigns is not paying rent.

**Closed as null across all three leagues:** model class (gradient boosting is
2.5-3 SE worse everywhere, train-CV gaps up to 0.016; stacks assign the GBM
negative weight), functional form (no curvature anywhere; MLB spline gains were
96% optimism), interactions (all null or SIG worse), regularization (blend C is
flat over 7 orders of magnitude — there is no variance to regularize).

**Protocol note.** The +0.0020 gate was pre-registered for NEW FEATURES, which
must earn their complexity. A zero-complexity correction like finding 2 arguably
deserves a different bar — but choosing that bar *after* seeing the result is
precisely the sin the protocol exists to prevent. If a lower bar for
zero-complexity corrections is ever wanted, it must be pre-registered in its own
document first, before any candidate is screened against it.

### Round 2 — 2026-07-31. Axes G-I + error map + floor. **0 adoptions, 0 TEST looks.**

| axis | MLB | NFL | NHL |
|---|---|---|---|
| G calibration | nested −0.000021 | null | null |
| H training cadence | nested −0.000013 | null | null |
| I ensembling | nested −0.000089 | null | null |

Largest raw gain anywhere: **+0.000008**, 250x below gate, and it inverts when
re-selected honestly. Three findings explain *why* these had to be null:

1. **Platt-on-a-rating is not calibration, it is a reparametrization** the blend
   already performs — verified algebraically (post-hoc Platt on the blend's own
   training predictions returns exactly (b0,b1)=(0,1) to 8 decimals). The round-1
   "free parameter" lesson, applied to the calibration axis.
2. **There is no calibration drift.** Per-season Platt slopes: Q=12.65 on 13 df
   (Q/df=0.97); intercepts Q/df=1.07. The season-to-season wobble is *exactly*
   sampling noise. The apparent cost of a "stale" curve decomposes into the
   oracle's own in-sample degrees of freedom (+0.000137 predicted vs +0.000234
   measured). No disease -> recency weighting (axis H) had nothing to cure, which
   is why it is monotonically harmful: half-life 0.5 seasons costs +0.000605,
   1-season windows +0.001252. **Expanding-window training is optimal.**
3. **Per-tier calibration IS load-bearing** — the one calibration choice that is
   not free. The Elo slope falls 1.31 -> 0.70 as features are added; pinning one
   shared curve costs 1.8 fold-sd. Correctly shipped.

### The error map (phase0/error_map.py, data/error_map.json)

Ranking slices by total excess loss produced a **noise ordering** — almost every
slice has *negative* excess (the model beats a slice-label oracle nearly
everywhere). Two better instruments were built: shortfall against the dimension's
own mean, and an in-slice recalibration likelihood ratio whose null expectation
is exactly 1.0 nat per slice at any n.

**What a PERFECT slice-conditional recalibration would buy, net of that null:**

| dimension | MLB | NFL | NHL |
|---|---|---|---|
| probability decile | −0.000039 | −0.000763 | +0.000666 |
| month / week | +0.000017 | +0.000113 | +0.000005 |
| favourite side | −0.000045 | −0.000178 | −0.000203 |
| rest | −0.000023 | −0.000917 | −0.000322 |
| team | −0.000006 | −0.000034 | −0.000228 |

**One cell in the entire map clears a gate, and it is in-sample on the model's own
output.** Conclusion: the models are not mis-specified anywhere we can find. The
common theme in the shortfall lists is early-season cold start (MLB team-games
0-9 cal_gap +0.028±0.011; NFL weeks 1-2; NHL team-games 5-14 +0.029±0.015) — a
real, cross-league, mechanistically sensible pattern that still nets ~0 once the
null is subtracted.

### The floor — how far is the theoretical limit?

Odds used as an evaluation benchmark only (sanctioned).

| league | n | our DEV | market | entropy floor | us -> market |
|---|---|---|---|---|---|
| MLB 2010-15 | 13,930 | 0.679233 | 0.676351 | 0.678391 | **+0.00288** [0.0015, 0.0041] |
| NFL 2006-15 | 2,531 | 0.622080 | 0.610363 | 0.606654 | **+0.01172** [0.0039, 0.0190] |
| NHL | — | — | — | — | skipped — no historical closing line exists in the repo; inventing one would be worse than reporting nothing |

**The honest headline: the market-vs-floor comparison is nearly circular.** If the
close is calibrated then E[LL_mkt | p_mkt] = H(p_mkt) *identically* — the two
columns are one quantity measured twice, and both gaps are consistent with zero
(−0.0020±0.0014, +0.0037±0.0073). That is confirmation the market is calibrated,
NOT headroom. Also mean H(p_mkt) is an **upper bound** on the true irreducible
loss (the market's probability is a coarsening of the truth, H is concave), so the
real floor sits at or below these numbers and is not identified by the market
alone.

**So the answer to "how far from the theoretical limit": MLB ~0.0029, NFL ~0.0117
in log loss** — and since the market sits essentially *at* the floor, our gap to
the market IS our gap to the limit.

### Where that leaves the program

Rounds 1-2 closed all 9 axes x 3 leagues with zero adoptions. Combined with ~60
prior channel nulls and the error map showing no exploitable residual structure,
the evidence is consistent: **the remaining gap is missing INFORMATION, not
mis-specification.** No re-parameterization, calibration, ensembling or model
class recovers it. Future rounds should be aimed at information the public record
does not contain (the in-season lineup/injury archives now accruing), not at
further optimization of the existing components.
