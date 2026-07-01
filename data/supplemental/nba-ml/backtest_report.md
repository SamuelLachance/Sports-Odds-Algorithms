# NBA ML Honest Backtest Report

Generated: 2026-07-01T20:29:35.421560Z
Dataset: 13112 games, 2016-10-25 -> 2026-06-13
Out-of-sample seasons: [2019, 2020, 2021, 2022, 2023, 2024, 2025]

## Against the spread (graded at REAL closing line)
- Bets: 2781
- Record: 1415-1332-34 (51.51% win)
- Units: -40.71 | ROI: -1.46%

## Moneyline (graded at REAL closing line)
- Bets: 5587
- Record: 2253-3334 (40.33% win)
- Units: -174.92 | ROI: -3.13%

## Closing Line Value (ATS bets with opening data)
- Bets with opening line: 1322
- Mean CLV (points): -0.344
- Positive CLV: 34.3%

## Probability calibration (win prob)
- Model log-loss: 0.6216 | Brier: 0.214
- Market log-loss: 0.604 | Brier: 0.2084

Interpretation: beating the closing line is extremely hard. Positive CLV
and ROI at the close on a filtered subset are the real signals; a model
log-loss at or slightly above the market's is expected and honest.