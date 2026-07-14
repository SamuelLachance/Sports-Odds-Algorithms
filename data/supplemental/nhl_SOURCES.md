"""Supplemental NHL public data lakes (no paid Evolving-Hockey / no NA fills).

| Path | Source | Coverage | Used for |
|---|---|---|---|
| `closing-odds/nhl.csv` | ESPN Core odds + SBR archives | close ML ~98%; open ML ~20% overall (near-complete 2024+); puck line ~84%; totals ~44% | market / steam / spread / total features |
| MoneyPuck team + goalie CSVs | [MoneyPuck](https://moneypuck.com/data.htm) via `web/nhl_v2/data.py` | 2008+ | xG / Corsi / flurry / PP-PK xG / goalie GSAx |
| `.build-cache/nhl-history/{season}/` | NHL official API team game logs + goalies | 2008–current | PP%/PK%/faceoff, scores, rest/schedule |

Feature-engine notes:

- Market columns use `has_*` flags; missing opens leave `has_steam=0` / `ml_steam_pp=0`
  (never invent open lines). Open totals are sparse in ESPN history → `has_total` only
  when `close_total` exists.
- Arena altitude proxies (COL/CGY/UTA) are fixed public geography constants.
- Historical injury lakes are not multi-year public from ESPN; do not train on
  injury burden until a PIT snapshot archive exists.

Rebuild:

```powershell
python scripts/build_nhl_training_table.py --start-season 2015 --end-season 2025
python scripts/train_all_hybrids.py --leagues nhl --ship --force-ship
```
"""
