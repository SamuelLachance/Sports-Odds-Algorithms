# CFB supplemental sources

| Lake | Script | Notes |
|------|--------|-------|
| `cfb-rankings/{ap,cfp}_{season}.csv` | `scripts/build_cfb_rankings.py` | ESPN Core weekly polls; PIT join via `web/cfb_v2/rankings.py` |
| `cfb-pbp/game_team_epa.csv` | `scripts/build_cfb_pbp_epa.py` | cfbfastR sportsdataverse parquet; season-to-date priors only (`web/cfb_v2/epa.py`) |
| `closing-odds/cfb.csv` | `scripts/fetch_cfb_odds.py` | Open/close spreads + MLs; never invent open=close |

Missing lakes → `has_rank=0` / `has_epa=0` / `has_steam=0` (finite defaults, zero NaNs).
