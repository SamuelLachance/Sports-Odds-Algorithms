# NBA supplemental notes (CBB-lesson port)

| Source | Path / script | Notes |
|--------|---------------|-------|
| Closing odds | `closing-odds/nba.csv` | Close ML/spread/total; open spreads ~31% coverage |
| Steam features | `web/market_steam_features.py` | Wired in FE + `build_nba_training_table.py` (attach before emit) |
| Prior OVR | `data/ratings/2k_nba_*.csv` | `prev_ovr_diff` / `has_ext_rating` when prior-season file exists |

Never invent open=close. Missing opens → `has_open_line=0` / `has_steam=0`.
