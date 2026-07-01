# NBA ML pipeline (pilot) and rollout template

A leak-free, market-aware machine-learning prediction stack for NBA, validated
honestly against **real closing lines**. This is the reference template for
rolling the same approach out to the other sports.

## Why it exists

The legacy pick backtests reported implausible ROI (e.g. NBA "60%") because they
bet against a *synthetic* market derived from the model itself, tuned thresholds
in-sample, and leaked the future via a sport layer built through today's date.
This package replaces that with a rigorous, reproducible pipeline whose headline
metric is **CLV / ROI vs the real closing line**.

## Components

| File | Role |
|------|------|
| `web/nba_odds_espn.py` | Fetch real opening/closing spreads, moneylines, totals from ESPN's core odds API (median consensus across books). |
| `scripts/fetch_nba_odds.py` | Backfill `data/supplemental/closing-odds/nba.csv` (resumable, per-day cache). |
| `web/nba_ml/features.py` | Point-in-time features (Elo, rolling efficiency/pace, rest/B2B/3-in-4, travel + time-zone, de-vigged market line). State updates only *after* each game -> no lookahead. |
| `web/nba_ml/dataset.py` | Build the versioned training matrix from the odds+scores table. |
| `web/nba_ml/model.py` | XGBoost home-margin regression + home-win classifier; normal-model cover probability; market-aware ensemble. |
| `web/nba_ml/calibrate.py` | Isotonic calibration + log-loss / Brier / reliability. |
| `web/nba_ml/backtest.py` | Nested walk-forward: train on prior seasons, pick thresholds on a validation season, grade on an untouched test season at the **real closing line**; reports ROI, CLV, calibration. |
| `scripts/train_nba_ml.py` | Orchestrates dataset -> backtest -> deployable artifacts + honest report. |
| `web/nba_ml/predict.py` | Cheap inference from committed artifacts (no training at runtime). |

## Reproduce

```bash
python -m pip install -r requirements.txt
python scripts/fetch_nba_odds.py --start 2016-10-01 --end <today>
python scripts/train_nba_ml.py
```

Artifacts land in `data/models/nba/`; the honest report in
`data/supplemental/nba-ml/backtest_report.md`.

## Honest pilot result (2016-2026, out-of-sample vs real closing lines)

- ATS ROI ~ -1.5%, ML ROI ~ -3%, **negative CLV** (~34% positive).
- Model win-prob log-loss (~0.62) is close to but slightly worse than the
  market's (~0.60).

Conclusion: the model is well-calibrated but **does not beat the NBA closing
line** with public data + this feature set. Official NBA picks are therefore
**disabled** (`data/pick_strategy.json`), and the live ML layer is **opt-in and
off by default** (`NBA_ML_ENABLED=1`) so the public board is not silently
changed. This is the intended, truthful outcome of rigorous validation.

## Live integration

`web/blend_service.py` uses the NBA ML win probability as the basketball sport
layer only when `NBA_ML_ENABLED=1` and artifacts exist; otherwise it falls back
to the existing matrix model. The daily GitHub Pages build installs
`requirements.txt` but leaves the flag off.

## Rolling out to another sport

1. **Odds table**: adapt `nba_odds_espn.py` to the sport's ESPN `sport_path`
   (e.g. `hockey/nhl`, `baseball/mlb`, `football/nfl`). ESPN's core odds
   endpoint is league-agnostic.
2. **Features**: reuse `features.py` scaffolding; swap sport-specific signals
   (MLB: probable pitcher, park, weather, bullpen; NHL: starting goalie; NFL:
   QB, rest, weather). Keep everything point-in-time.
3. **Targets/markets**: NBA/NFL grade ATS; NHL/MLB grade moneyline; soccer 1X2.
4. **Model/backtest/predict**: the `model`, `calibrate`, `backtest`, `predict`
   modules are largely sport-agnostic; parameterize league.
5. **Ship**: keep the flag off until the walk-forward shows positive CLV/ROI vs
   real closing lines. Never advertise ROI that isn't graded at the close.
