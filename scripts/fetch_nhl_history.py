"""Fetch NHL history: MoneyPuck team xG + Stats API results/goalie logs.

Caches under .build-cache/nhl-history/:
  mp_all_teams.csv          MoneyPuck game-by-game team file (all seasons)
  {season}/team_games.json  per-team per-game results (regular + playoffs)
  {season}/goalies.json     per-goalie per-game logs (starters, saves)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nhl_v2.data import (  # noqa: E402
    CACHE_ROOT,
    download_moneypuck_teams,
    fetch_goalie_games,
    fetch_team_games,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch NHL v2 history")
    parser.add_argument("--start-season", type=int, default=2008)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--skip-moneypuck", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="refetch cached seasons")
    args = parser.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_moneypuck:
        mp_path = CACHE_ROOT / "mp_all_teams.csv"
        if args.refresh or not mp_path.is_file():
            t0 = time.time()
            download_moneypuck_teams(mp_path)
            print(f"MoneyPuck all_teams.csv: {mp_path.stat().st_size / 1e6:.1f} MB "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        else:
            print("MoneyPuck all_teams.csv cached", flush=True)

    for season in range(args.start_season, args.end_season + 1):
        season_dir = CACHE_ROOT / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)
        team_path = season_dir / "team_games.json"
        goalie_path = season_dir / "goalies.json"
        if not args.refresh and team_path.is_file() and goalie_path.is_file():
            print(f"season {season}: cached", flush=True)
            continue
        t0 = time.time()
        team_rows = fetch_team_games(season)
        goalie_rows = fetch_goalie_games(season)
        team_path.write_text(json.dumps(team_rows, separators=(",", ":")), encoding="utf-8")
        goalie_path.write_text(json.dumps(goalie_rows, separators=(",", ":")), encoding="utf-8")
        print(
            f"season {season}: {len(team_rows)} team rows, {len(goalie_rows)} goalie rows "
            f"[{time.time() - t0:.1f}s]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
