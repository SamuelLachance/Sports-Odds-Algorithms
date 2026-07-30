# NFL 2026 in-season weekly refresh — runbook

Written 2026-07-30 (offseason reconnaissance, iteration 15). Purpose: when the
season starts (~Sept 10), the weekly refresh must be a known procedure, not an
archaeology project. Implementation target: `phase0/nfl_weekly.py` +
`.github/workflows/nfl-weekly.yml`, mirroring the NHL weekly pattern.

## Weekly data assets (URLs verified 2026-07-30)

| Asset | URL | Status |
|---|---|---|
| Schedule + results | `https://github.com/nflverse/nfldata/raw/master/data/games.csv` | 200 — evergreen single file, updates within hours of each game; refresh `data/nfl_games.csv` from it |
| Play-by-play | `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet` | will appear week 1 (2025 file verified 200); nightly updates in-season |
| Injuries | `https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.csv` | will appear ~week 1 (2025 verified 200); save as `data/inj_2026.csv` — **this arms the dormant lineup-availability MC** in `nfl_lineups.py` (currently `absences = 0`) |
| Depth charts | `https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.csv` | 200 already live; refresh `data/depth_charts_2026.csv` |

`habitatring.com/games.csv` mirror is dead (000) — use the nfldata GitHub raw URL.

## The weekly chain (order matters)

1. **games**: re-download games.csv → rewrite `data/nfl_games.csv` (adds the
   week's finals; the serve consumes results + schedule from here).
2. **pbp 2026**: download `play_by_play_2026.parquet` → feature walks (EPA,
   pass/run units, per-play TrueSkill via `nfl_trueskill_players.py` /
   `nfl_micro_pull.py` pattern) must advance THROUGH played 2026 games.
3. **injuries + depth charts**: refresh `data/inj_2026.csv` +
   `data/depth_charts_2026.csv` → `nfl_lineups.py` builds expected lineups
   with real absences.
4. **serve**: `phase0/nfl_season_serve.py` → `site/data/nfl.json`
   (272-game schedule with p_home, 20k-sim standings/playoff odds).
5. **site**: `mlbwp_site/build_site.py` rebuild (daily refresh.yml already does
   this; the weekly job only needs steps 1-4 + commit).

## Known engineering gap (the September task)

`nfl_season_serve.py` freezes **end-2025** states ("all games <= 2025") and
predicts all 272 games from there. In-season it must instead:
- walk every feature THROUGH the played 2026 games (pbp + finals available),
- refit nothing (blend stays the adopted training protocol: <= 2025,
  hl=3, C=100 — one-look discipline; only STATES advance),
- fill `absences` from the injury report for the upcoming week,
- keep already-played 2026 games' pre-game predictions FROZEN in the payload
  (graded HIT/MISS like NHL) rather than re-predicting them post-hoc.

The NHL serve solved the same problem with `final_states()` + an
upcoming-slate block (`phase0/nhl_serve.py`) — same architecture applies.

## Cadence

- Tuesday morning UTC (after MNF, before Thursday game): full weekly chain.
- The existing daily `refresh.yml` + 20-min `odds.yml` handle board + edges;
  no change needed there.
