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

Round 2 (axes G-I: calibration, training cadence, ensembling) remains open.
