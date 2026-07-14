# MLB Statcast sources

| Lake | Script | Notes |
|------|--------|-------|
| `mlb-statcast/pitcher_prev_xera.csv` | `scripts/build_mlb_statcast_priors.py` | Prior-season pitcher xERA (`for_season`) |
| `mlb-statcast/team_prev_xwoba.csv` | `scripts/build_mlb_statcast_priors.py` | Prior-season team xwOBA |
| `mlb-statcast/team_xwoba_rolling.csv` | `scripts/build_mlb_statcast_rolling.py` | Season-end snapshots; join `asof_date < game_date` → `has_statcast_rolling` |

See also `ROLLING_SOURCES.md` in this folder after the rolling builder runs.
