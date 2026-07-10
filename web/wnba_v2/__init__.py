"""WNBA GradientBoost v2 prediction stack.

Full-history ESPN pipeline (scores 1997+, team box scores 2003+, multi-book
odds 2018+) feeding a walk-forward feature engine (Elo, four factors, pace,
efficiency EWMAs, rest/travel) and an XGBoost + logistic-regression ensemble
with isotonic calibration. Offline training scripts live in scripts/
(fetch_wnba_history.py, build_wnba_training_table.py, train_wnba_model.py,
backtest_wnba_bets.py); live inference is web/wnba_v2/live.py.
"""
