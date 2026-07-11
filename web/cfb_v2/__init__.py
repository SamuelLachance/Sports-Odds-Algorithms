"""CFB GradientBoost v2 prediction stack.

Closing-odds FBS history (2022+) plus ESPN live scoreboards feed a
walk-forward feature engine (Elo, scoring EWMAs, rest/week/neutral,
score-based player proxies) and an XGBoost + logistic-regression
ensemble with isotonic calibration and a dedicated margin head.

Offline: scripts/build_cfb_training_table.py, train_cfb_model.py,
backtest_cfb_bets.py. Live: web/cfb_v2/live.py.
"""
