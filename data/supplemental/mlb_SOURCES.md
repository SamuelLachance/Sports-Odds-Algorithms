# MLB supplemental data sources

Public sources wired into the MLB hybrid feature stack:

| Need | Source | Path / script |
|------|--------|----------------|
| Schedule, SP, team logs | MLB Stats API (`statsapi.mlb.com`) | `.build-cache/mlb-history/` via `scripts/fetch_mlb_history.py` |
| Opening / closing odds + steam | SportsbookReview historical archive (via `web/sbr_odds.py`) | `data/supplemental/closing-odds/mlb.csv` — `scripts/build_mlb_open_close_cache.py` |
| Weather (temp/humidity/precip/wind) | Open-Meteo Historical API | `data/supplemental/mlb-weather/game_weather.csv` — `scripts/build_mlb_weather.py` |
| Weather / roof fallback | MLB Stats API schedule hydrate | `data/supplemental/mlb-weather/mlb_api_weather.csv` — `scripts/build_mlb_schedule_context.py` |
| Home-plate umpires | MLB Stats API schedule hydrate=`officials` | `data/supplemental/mlb-umpires/game_umpires.csv` — `scripts/build_mlb_schedule_context.py` |
| IL / injured-list burden | MLB Stats API transactions | `data/supplemental/mlb-injuries/team_il_daily.csv` — `scripts/build_mlb_il_burden.py` |
| Statcast xERA / xwOBA priors | Baseball Savant expected-statistics CSV (prior season only) | `data/supplemental/mlb-statcast/` — `scripts/build_mlb_statcast_priors.py` |

Training join: `scripts/build_mlb_training_table.py` (hard-gates `FEATURE_COLUMNS` — zero NaNs; missing fields use `has_*` flags, never invent open=close).
