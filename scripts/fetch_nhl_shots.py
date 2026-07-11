"""Fetch MoneyPuck shot-level data and aggregate per-goalie per-game xG faced.

Produces .build-cache/nhl-history/goalie_xg_games.csv with columns:
  gameId, goalieId, team, shots_faced, xg_faced, goals_allowed,
  hd_shots_faced, hd_xg_faced, hd_goals_allowed

High-danger shots are those with xGoal >= HD_XG_THRESHOLD (MoneyPuck proxy).
gameId matches NHL Stats API ids (e.g. 2023020001). Empty-net rows excluded.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nhl_v2.data import CACHE_ROOT, get_bytes  # noqa: E402

SHOTS_URL = "https://peter-tanner.com/moneypuck/downloads/shots_{season}.zip"
OUT_PATH = CACHE_ROOT / "goalie_xg_games.csv"
ZIP_DIR = CACHE_ROOT / "shots"
HD_XG_THRESHOLD = 0.20


def season_game_id(season: int, mp_game_id: str, is_playoff: int) -> int | None:
    """MoneyPuck game_id is like 20500 (regular) / 30111 (playoff) or full id."""
    raw = str(mp_game_id).strip()
    if not raw:
        return None
    try:
        value = int(float(raw))
    except ValueError:
        return None
    if value > 2000000000:
        return value
    return int(f"{season}0{value:05d}") if value < 100000 else None


def aggregate_zip(payload: bytes, season: int) -> dict[tuple[int, int, str], list[float]]:
    """(gameId, goalieId, team) -> [shots, xg, goals, hd_shots, hd_xg, hd_goals]."""
    out: dict[tuple[int, int, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(
            (n for n in archive.namelist() if n.lower().endswith(".csv")), None
        )
        if not name:
            return out
        with archive.open(name) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
            for row in reader:
                goalie_raw = row.get("goalieIdForShot") or ""
                try:
                    goalie_id = int(float(goalie_raw))
                except ValueError:
                    continue
                if goalie_id <= 0:
                    continue
                if row.get("shotOnEmptyNet") in ("1", "1.0"):
                    continue
                game_id = season_game_id(
                    int(float(row.get("season") or season)),
                    row.get("game_id") or "",
                    int(float(row.get("isPlayoffGame") or 0)),
                )
                if not game_id:
                    continue
                shooter_team = str(row.get("teamCode") or "")
                home_team = str(row.get("homeTeamCode") or "")
                away_team = str(row.get("awayTeamCode") or "")
                goalie_team = away_team if shooter_team == home_team else home_team
                try:
                    xg = float(row.get("xGoal") or 0.0)
                except ValueError:
                    xg = 0.0
                goal = 1.0 if row.get("goal") in ("1", "1.0") else 0.0
                key = (game_id, goalie_id, goalie_team)
                bucket = out[key]
                bucket[0] += 1.0
                bucket[1] += xg
                bucket[2] += goal
                if xg >= HD_XG_THRESHOLD:
                    bucket[3] += 1.0
                    bucket[4] += xg
                    bucket[5] += goal
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch MoneyPuck shots -> goalie xG games")
    parser.add_argument("--start-season", type=int, default=2008)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    totals: dict[tuple[int, int, str], list[float]] = {}

    for season in range(args.start_season, args.end_season + 1):
        zip_path = ZIP_DIR / f"shots_{season}.zip"
        if args.refresh or not zip_path.is_file():
            t0 = time.time()
            payload = get_bytes(SHOTS_URL.format(season=season))
            zip_path.write_bytes(payload)
            print(f"downloaded shots_{season}.zip {len(payload) / 1e6:.1f} MB "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        chunk = aggregate_zip(zip_path.read_bytes(), season)
        totals.update(chunk)
        print(f"season {season}: {len(chunk)} goalie-game rows", flush=True)

    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "gameId", "goalieId", "team",
            "shots_faced", "xg_faced", "goals_allowed",
            "hd_shots_faced", "hd_xg_faced", "hd_goals_allowed",
        ])
        for (game_id, goalie_id, team), vals in sorted(totals.items()):
            shots, xg, goals, hd_shots, hd_xg, hd_goals = vals
            writer.writerow([
                game_id, goalie_id, team,
                int(shots), round(xg, 4), int(goals),
                int(hd_shots), round(hd_xg, 4), int(hd_goals),
            ])
    print(f"wrote {OUT_PATH} ({len(totals)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
