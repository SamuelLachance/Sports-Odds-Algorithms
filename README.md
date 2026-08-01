# Sports-Odds-Algorithms — GLASSBOX

Market-blind prediction models for **MLB, NFL and NHL**, served as a static
site (GitHub Pages → sharpsheettips.com). Every model is built exclusively
from public play-by-play / game data — betting-market information is **never**
a model input. Odds appear in exactly two places: as an *evaluation benchmark*
(closing-line log-loss) and in a *post-process edge layer* that compares live
market prices to the model's finished output.

## The constitution

1. **Market-blind models.** No lines, odds, totals, or consensus anywhere in
   training or features. Salary/contract data (labor market) is allowed.
2. **Locked splits, one look.** Every league has a frozen DEV/TEST split.
   Ideas are screened on DEV against a pre-registered gate; each idea gets
   **one** TEST evaluation, recorded as a row in `data/*_test_ledger.csv`
   whether it ships or dies. No re-tuning after the look.
3. **Display follows log-loss.** The site shows what the best-scoring model
   says; there is no editorial override.
4. **Publication honesty.** A played game publishes the probability we held
   *before* it started, never the recomputed one. Picks are gated on
   information, small samples are not advertised as rates, and every
   published claim traces to a ledger row or a committed DEV report.

## Shipped models (TEST log-loss, lower is better)

| League | Model | TEST LL | Baseline (plain Elo) | Closing line |
|---|---|---|---|---|
| MLB | FIP-Elo + per-PA TrueSkill blend (14 features) | 0.67274 | 0.67861 | ~0.665 |
| NFL | 14-feature stack incl. 11v11 per-play TrueSkill | 0.61919 | 0.63984 | ~0.609 |
| NHL | Tuned Elo + rest/B2B + xG-team rating | 0.66418 | 0.66918 | — |

The NFL player-rating feature (participation TrueSkill over every snap:
β=60, weekly QB Kalman fusion, opponent quality, salary cohort priors) beats
team Elo *on its own* (0.62973). The NHL xG rating updates on **expected**
goal margin only, never realized goals.

## How far from the theoretical limit? (measured 2026-07-31)

A perfectly calibrated forecaster's log-loss equals the mean binary entropy of
the true win probabilities, so the devigged closing line gives an estimate of
the floor. Odds used as an evaluation benchmark only.

| League | Ours | Closing line | Gap | Verdict |
|---|---|---|---|---|
| MLB | 0.679233 | 0.676351 | +0.00288 | significant |
| NFL | 0.622080 | 0.610363 | +0.01172 (DEV) / **+0.01034** (full model) | significant |
| **NHL** | 0.672423 | 0.670717 | **+0.00171** | **not significant** |

Measured on DEV; NHL closing lines sourced 2026-07-31 (15,203 games, 99.97%
join, zero result mismatches). Three findings shape everything after:

- **The MLB closing line statistically *encompasses* our model.** Conditional
  on the close, our forecast carries no weight (b=+0.028, t=0.25) and adding it
  to a market-only forecast makes that forecast *worse*.
- **NHL ties the market but knows *different things*.** Its orthogonal
  information predicts outcomes at t=5.75 and is worth more than the whole gap
  — yet our own orthogonal component is also alive (t=2.48). Both sides hold
  what the other lacks; the weights say the market prices goalie confirmation
  and scratches, which we have no column for. That is why the in-season lineup
  archive exists.
- **The gap is diffuse, not concentrated.** Zero of sixteen heterogeneity tests
  reject uniformity, so there is no slice to target — which explains why ~60
  channel screens came back null.

`documents/depth_program.md` records the 9-axis × 3-league depth sweep
(joint hyperparameters, model class, functional form, interactions,
regularization, calibration, cadence, ensembling): **zero adoptions**, with
every candidate reported raw *and* nested. One NFL candidate looked like a
+0.005 win raw and was 82% search optimism.

**Research completeness (as of 2026-08-01):** every protocol-compatible
channel has been tested. MLB: platoon, fatigue, form, schedule, DER, BABIP,
bullpen, power, SIERA, baserunning (shipped); lineup-absence and catcher
framing (DEV nulls — the framing proxy itself validates against known elite
framers). NFL: 14 shipped features; TS2/TS3/TTT, PFF-WAR transplants,
coordinator/coach, SRS, RAPM all null or DEV-mirage. NHL: Elo+rest+B2B+xG
shipped; Glicko-2, goalie level *and* starter-delta, player-rating
aggregates, scratch-absence, PP/PK split, dead-game weighting all null
(scratch-absence is the closest miss and re-opens when announced-lineup
data accrues via the in-season archiver). Every claim traces to a ledger
row or a committed DEV report.

**Environment channels, screened 2026-08-01 (`documents/depth_program.md` Rounds 3–3c).**
The error map's `team` dimension pools each franchise's home *and* away games,
so it never sharply tested a venue effect. A `home_venue` dimension (home games
only, one slice per ballpark/stadium/arena) now does, in all three leagues:

| league | venue slices | survive Bonferroni | ll ceiling if every venue were perfectly recalibrated |
|---|---|---|---|
| MLB | 32 | **0** | ~0.00013 |
| NFL | 33 | **0** | ~0.0066 |
| NHL | 31 | **0** | ~0.0017 |

Null everywhere; ceilings are quoted to two significant figures because the
NFL/NHL harnesses fit through sklearn's iterative solver and reproduce to
~1e-4, not exactly (the DEV asserts carry a 5e-4 tolerance for the same
reason). MLB's ceiling caps the entire park channel at ~4.5% of its gap
to the closing line — and that is an in-sample bound that would itself be
overfitting, so park factors, weather and umpire data are not worth acquiring.
The one Bonferroni survivor anywhere, COL in the pooled `team` slice, is **not**
a Coors effect: at Coors itself it is n.s. (p=0.49), so it lives in Colorado's
road games — a team property. NFL broadcast slot was screened too and is
likewise null, with a *negative* recalibration ceiling.

## Pick policy

Full text in [`documents/pick_policy.md`](documents/pick_policy.md), written and
committed *before* the implementation. Every scheduled game is stamped with an
**information tier** on each strictly-pre-game write:

| Tier | Condition | Published as |
|---|---|---|
| CONFIRMED | official lineup posted | pick |
| PROJECTED | probable starter known, lineup not out | pick |
| EARLY | team ratings only | **scheduled, not a pick** |

Rules: only CONFIRMED/PROJECTED publish as picks; anything inside 55/45 is a
LEAN, not a play; the track record splits by tier and **never pools** them; a
graded pick is the last pre-game forecast, frozen; and no manufactured volume —
if the slate has nothing, the board says so. Before this, every scheduled game
was published as a "pick"; today's live board carries 36 picks against 386
merely-scheduled games.

The tier rule lives in `mlbwp/pred_ledger.py`, which also *emits the JavaScript*
the page uses, so the Python and the browser cannot drift apart — a silent drift
here would label a card one tier while the ledger stamped it another. The tier
is stamped at write time on every strictly-pre-game write; once a game starts
its number and tier are immutable.

The same gate now sits in front of the edge layer: `market/edges.py` will not
quote a game whose starter is unknown.

## Why each pick — and what the bars actually mean

Every game page shows a per-feature breakdown in home-probability points, taken
from the model's own coefficients. The three leagues do not give the same
*guarantee*, and the site now says so per panel rather than implying one rule:

| league | construction | guarantee |
|---|---|---|
| **NHL** | sequential deltas through a single logistic | **exact**: `0.5 + Σ(bars)/100 = the served probability` |
| **MLB** | exact deltas for the five feature terms | **exact** from `recal_prob`; the four baseline terms are a slope linearisation and are *not* additive with them |
| **NFL** | `coefficient × feature × scale` | **faithful, not additive** — Spearman 0.9999 against the served probability, but the bars do not sum to it |

NFL's cannot be exact: it is a logit-space attribution, so it omits the
intercept and the sigmoid's curvature. Its caption used to read "push on the
home win probability vs a 50/50 game", which invited readers to add the bars to
50% — only 11 of 272 games land within 1pp of that, with errors reaching
0.179. The caption now states the limitation, and a test asserts that only the
league whose construction *is* exact may advertise exactness.

Verified continuously: NHL's identity holds to 0.0014 across all 1,312 served
games (rounding budget 0.00205), and the NHL key order is generated from
`phase0/nhl_contributions.py` rather than retyped, so the payload and the page
cannot drift.

## Track record (`#/record`)

The site publishes its own grading. Every pick's pre-game probability is frozen
at write time — a played game publishes the number we had *before* kickoff, not
the recomputed one (`phase0/nfl_ph_freeze.py`, and the equivalent in the MLB
serve). Post-hoc numbers would flatter us silently, which is why that mechanism
is now an extracted, tested, mutation-checked module rather than an inline loop.

The page shows cumulative accuracy, per-tier rollups, HIT/MISS/**PENDING**
chips, and a collapsed scheduled block. Two honesty gates sit on the display:
the headline rate reports the frozen holdout number until the live ledger
reaches **n≥100** (`mlbwp/predict_slate.py`), and a tier whose sample is under
**n≥30** shows its record without claiming a rate (`mlbwp_site/build_site.py`). Small
samples are shown, never advertised.

Live as of 2026-08-01 — 435 rows, 13 graded, 36 pending, 386 scheduled:

| Tier | n | log-loss | acc |
|---|---|---|---|
| CONFIRMED | 10 | 0.67173 | 0.500 |
| PROJECTED | 3 | 0.66039 | 0.667 |
| EARLY | **0** | — | — |

Zero graded EARLY rows is the policy working, not an absence of data: 386 EARLY
games were forecast and none was published as a pick. Thirteen games is far too
few to mean anything — that is exactly why the n≥100 gate exists.

`market/edge_ledger.py` tracks the EV edges the same way, and records the
**CLV close-lag** — minutes between our last pre-game observation and first
pitch. A stale close biases measured CLV toward zero, so the lag is published
(median and p90) instead of being quietly assumed away.

## Repo map

```
mlbwp/           MLB model package (ratings, blend, serving, db)
  pred_ledger.py   information tiers — owns the rule AND emits the page's JS
phase0/          research scripts: eval harnesses, NFL/NHL engines, audits
  nfl_ph_freeze.py     pre-game probability freeze (the honesty mechanism)
  nfl_season_guards.py the two guards that first fire on opening day
market/          post-process EV edge layers (edges, nfl_edges, nhl_edges)
  edge_ledger.py   realized edge tracking + CLV close-lag
mlbwp_site/      build_site.py — the single-file 3-league SPA (incl. #/record)
site/            published output (index.html + data payloads)
data/            frozen models, spines, TEST ledgers, audit reports
documents/       paper archive (PDFs local-only; extracted .txt committed)
  pick_policy.md     information tiers and the 5 publication rules
  depth_program.md   the 9-axis × 3-league depth sweep, and the floor table
tests/           pytest suite incl. payload-contract tests
refresh.py       daily site refresh (MLB board + NHL incremental + rebuild)
refresh_odds.py  edge-layer refresh (needs ODDS_API_KEY, server-side only)
```

## Pipelines

- **`.github/workflows/refresh.yml`** — scheduled `refresh.py`: MLB finals +
  board, NHL spine append (`phase0/nhl_update.py`) + re-serve
  (`phase0/nhl_serve.py`), site rebuild. Stdlib-only critical path; NHL steps
  skip gracefully if numpy/sklearn are absent.
- **`.github/workflows/odds.yml`** — `refresh_odds.py` re-prices the three
  edge layers against frozen opening odds (≥20% EV, 7-day window).
- **`.github/workflows/pages.yml`** — deploys `site/`.

Run locally:

```bash
python refresh.py
```

```bash
python -m pytest tests -q
```

143 tests. Beyond unit coverage they enforce two things that are otherwise
invisible: **payload contracts** (the built JSON actually has the keys the SPA
reads — the site's missing build gate) and the honesty mechanisms above. The
guards that first fire on NFL opening day are covered specifically because a
bug there would surface in September, in production, on a game day.

## Betting research (paper pilot)

`phase0/pilot_report.py` emits the pilot health report (calibration-z per
Kaunitz eq. 8, odds-bucket ROI, fade/ride-luck arms, per-season z) to
`data/pilot_report.md`. Standing findings: the 15%-EV gate's mid-dog bucket
is broken (selection winner's curse, z −4.5); the 20% gate is healthy;
fade-luck bets dominate ride-luck. Promotion to live money requires
sustained **live** positive CLV — no historical backtest qualifies by itself.

## Attributions

Retrosheet, MLB Stats API, nflverse, NHL Stats API, MoneyPuck.
