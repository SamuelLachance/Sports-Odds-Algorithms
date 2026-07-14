"""Supplemental CBB public data lakes (no paid KenPom / no NA fills).

| Path | Source | Coverage | Used for |
|---|---|---|---|
| `torvik-asof/full_YYYY.csv` | [toRvik-data](https://github.com/andreweatherman/toRvik-data) ratings archive | seasons ending 2015–2023 | PIT adjO/adjD/tempo/Barthag |
| `torvik-game-factors/game_factors_YYYY.csv` | same repo `game_factors/` | 2008–2023 | rolling four-factor / adj O-D |
| `torvik-pregame/pregame_YYYY.csv` | same repo `pregame_prob/` | 2008–2023 | Torvik published pregame WP |
| `cbb-rankings/ap_YYYY.csv` | ESPN Core API AP Top 25 | 2018–2026 | PIT poll rank (unranked=40) |
| `closing-odds/cbb.csv` | sportsbook closing lines (project fetch) | 2021– | market / open-spread steam |

Fetchers:

```powershell
python scripts/fetch_torvik_asof.py
python scripts/fetch_torvik_game_factors.py
python scripts/fetch_torvik_pregame.py
python scripts/fetch_cbb_ap_rankings.py
```

For 2024+ without Torvik archive years, the feature engine uses ESPN walk-forward
efficiency / Elo as real PIT proxies (`torvik_known=0`, `tf_known=0`) — never blank cells.
"""
