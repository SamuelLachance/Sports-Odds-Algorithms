# NFL open/close line cache

Public sources used by `scripts/build_nfl_open_close_cache.py`:

| Source | Coverage | Fields |
|---|---|---|
| [SBR 10Y archive](https://github.com/flancast90/sportsbookreview-scraper) (`nfl_archive_10Y.json`) | 2011 – early 2022 | open/close spread + total |
| [nflverse games.csv](https://github.com/nflverse/nfldata) | 1999 – current | closing ML / spread / total |
| [Odds API season snapshots](https://github.com/bobby-king3/nfl-market-movement-tracker) (DuckDB release) | 2025 season | open/close spread + total |

Merged output: `data/supplemental/closing-odds/nfl.csv`.

**No invented opens.** When an open line is missing, training features set `has_open_line=0` / `has_steam=0` (zeros, not NaN fills). Free public open-line archives for **2022–2024** are not available after SBR stopped publishing season pages; those seasons stay close-only via nflverse.

Rebuild:

```powershell
# optional: download DuckDB release into data/supplemental/nfl-opens/nfl_odds.duckdb
python scripts/build_nfl_open_close_cache.py
python scripts/build_nfl_training_table.py
```

Do not commit the large `nfl_odds.duckdb` binary; commit `opens_2025.csv` and `nfl.csv` only.
