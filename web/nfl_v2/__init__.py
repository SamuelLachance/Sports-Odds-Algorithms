"""NFL GradientBoost v2 prediction stack.

nflverse closing-odds history (1999+) plus ESPN live scoreboards feed a
walk-forward feature engine (team Elo, QB/coach Elo, rest/weather/context,
score-based luck proxies) and an XGBoost + logistic-regression ensemble
with isotonic calibration and a dedicated margin head.

Offline: scripts/build_nfl_training_table.py, train_nfl_model.py,
backtest_nfl_bets.py. Live: web/nfl_v2/live.py.
"""
