# Soccer supplemental sources

| Lake | Script | Notes |
|------|--------|-------|
| `soccer-xg/team_match_xg_priors.csv` | `scripts/build_soccer_xg_priors.py` | Understat public league pages; rolling xG for/against exclude current match |
| football-data CSVs | `web/soccer_v2/data.py` | Open (`PS*`) / close (`PSC*` / `B365C*`) 1X2; steam via `web/market_steam_features.soccer_steam_from_row` |

Missing xG → `has_xg=0`. Missing open≠close → `has_steam=0` (never invent open=close).
