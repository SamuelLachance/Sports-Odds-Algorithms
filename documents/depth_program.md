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

(rounds append here)
