"""NBA GradientBoost v2 prediction stack.

Full-history ESPN pipeline (scores 1996+, team box scores 2003+, multi-book
odds via supplemental closing-odds CSV 2016+) feeding a walk-forward feature
engine (Elo, four factors, pace, efficiency EWMAs, rest/travel) and an
XGBoost + logistic-regression ensemble with isotonic calibration. Offline
training scripts live in scripts/ (fetch_nba_history.py,
build_nba_training_table.py, train_nba_model.py, backtest_nba_bets.py);
live inference is web/nba_v2/live.py.
"""
