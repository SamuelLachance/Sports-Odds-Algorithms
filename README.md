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

## Shipped models (TEST log-loss, lower is better)

| League | Model | TEST LL | Baseline (plain Elo) | Closing line |
|---|---|---|---|---|
| MLB | FIP-Elo + per-PA TrueSkill blend (14 features) | 0.67274 | 0.67861 | ~0.665 |
| NFL | 14-feature stack incl. 11v11 per-play TrueSkill | 0.61919 | 0.63984 | ~0.609 |
| NHL | Tuned Elo + rest/B2B + xG-team rating | 0.66418 | 0.66918 | — |

The NFL player-rating feature (participation TrueSkill over every snap:
β=60, weekly QB Kalman fusion, opponent quality, salary cohort priors) beats
team Elo *on its own* (0.62973). The NHL xG rating updates on **expected**
goal margin only, never realized goals. All three sit near the public-data
identifiability frontier: ~50 screened challengers (TrueSkill-2/3, TTT,
RAPM, external predictors, sheet signals) are recorded as nulls in the
ledgers and in `documents/`.

## Repo map

```
mlbwp/           MLB model package (ratings, blend, serving, db)
phase0/          research scripts: eval harnesses, NFL/NHL engines, audits
market/          post-process EV edge layers (edges, nfl_edges, nhl_edges)
mlbwp_site/      build_site.py — the single-file 3-league SPA
site/            published output (index.html + data payloads)
data/            frozen models, spines, TEST ledgers, audit reports
documents/       paper archive (PDFs local-only; extracted .txt committed)
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

## Betting research (paper pilot)

`phase0/pilot_report.py` emits the pilot health report (calibration-z per
Kaunitz eq. 8, odds-bucket ROI, fade/ride-luck arms, per-season z) to
`data/pilot_report.md`. Standing findings: the 15%-EV gate's mid-dog bucket
is broken (selection winner's curse, z −4.5); the 20% gate is healthy;
fade-luck bets dominate ride-luck. Promotion to live money requires
sustained **live** positive CLV — no historical backtest qualifies by itself.

## Attributions

Retrosheet, MLB Stats API, nflverse, NHL Stats API, MoneyPuck.
