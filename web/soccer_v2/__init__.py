"""Soccer GradientBoost v2 — top-5 European club league prediction stack.

Data: football-data.co.uk full history (results, match stats, Bet365 +
Pinnacle open/close odds) for E0/D1/SP1/I1/F1 plus second divisions for
promoted-team rating continuity. Feature engine tracks Elo, attack/defence
EWMAs, shots/SoT xG proxies, Dixon-Coles intensities, form, rest and
head-to-head — all walk-forward (no leakage). Model: pooled XGBoost
multi:softprob + multinomial logistic ensemble with per-class isotonic
calibration, benchmarked against the Pinnacle closing line.
"""
