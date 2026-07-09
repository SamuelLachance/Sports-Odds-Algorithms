"""Fetch full MLB history (2011+) from the public MLB Stats API.

Downloads, per season:
  - regular-season schedule with probable pitchers + final scores
  - pitcher game logs for every probable starter (bulk people hydrate)
  - team hitting + pitching game logs (30 teams)

Raw JSON is cached under .build-cache/mlb-history/{season}/ so re-runs are
incremental. Completed past seasons are never refetched; the current season
is refreshed when its cache is older than today.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_v2.statsapi_data import (  # noqa: E402
    fetch_pitcher_logs,
    fetch_season_games,
    fetch_team_ids,
    fetch_team_logs,
)

CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "mlb-history"
FIRST_SEASON = 2011


def _season_dir(season: int) -> Path:
    path = CACHE_ROOT / str(season)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _is_fresh(path: Path, season: int, today: date) -> bool:
    if not path.is_file():
        return False
    if season < today.year:
        return True
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < 20 * 3600


def fetch_season(season: int, *, force: bool = False) -> dict[str, int]:
    today = date.today()
    out_dir = _season_dir(season)
    games_path = out_dir / "games.json"
    pitchers_path = out_dir / "pitchers.json"
    hitting_path = out_dir / "team_hitting.json"
    pitching_path = out_dir / "team_pitching.json"

    if not force and all(
        _is_fresh(p, season, today)
        for p in (games_path, pitchers_path, hitting_path, pitching_path)
    ):
        games = json.loads(games_path.read_text(encoding="utf-8"))
        return {"season": season, "games": len(games), "cached": 1}

    games = fetch_season_games(season)
    _write(games_path, games)

    pitcher_ids = sorted(
        {
            int(pid)
            for game in games
            for pid in (game.get("home_pp_id"), game.get("away_pp_id"))
            if pid
        }
    )
    pitchers = fetch_pitcher_logs(season, pitcher_ids)
    _write(pitchers_path, pitchers)

    team_ids = fetch_team_ids(season)
    _write(hitting_path, fetch_team_logs(season, team_ids, "hitting"))
    _write(pitching_path, fetch_team_logs(season, team_ids, "pitching"))

    return {
        "season": season,
        "games": len(games),
        "pitchers": len(pitchers),
        "teams": len(team_ids),
        "cached": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch MLB history from statsapi.mlb.com")
    parser.add_argument("--start-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for season in range(args.start_season, args.end_season + 1):
        started = time.time()
        stats = fetch_season(season, force=args.force)
        flag = "cache" if stats.get("cached") else f"{time.time() - started:5.1f}s"
        print(
            f"season {season}: {stats['games']} games"
            + (f", {stats.get('pitchers', '?')} pitchers" if not stats.get("cached") else "")
            + f" [{flag}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
