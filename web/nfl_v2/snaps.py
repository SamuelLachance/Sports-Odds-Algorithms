"""Attach nflverse snap-share aggregates onto NFL game dicts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SNAP_PATHS = (
    PROJECT_ROOT / "data" / "supplemental" / "nfl-snaps" / "game_team_snaps.csv",
    PROJECT_ROOT / ".build-cache" / "nfl-snaps" / "game_team_snaps.csv",
)

SNAP_FIELDS = ("wr1_snap_share", "skill_snap_share", "ol_starter_share", "ol_starters")


@lru_cache(maxsize=1)
def load_game_team_snaps() -> pd.DataFrame:
    for path in SNAP_PATHS:
        if path.is_file():
            frame = pd.read_csv(path)
            frame["team"] = frame["team"].astype(str).str.lower().str.strip()
            frame["game_id"] = frame["game_id"].astype(str)
            return frame
    return pd.DataFrame()


def attach_snaps_to_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snaps = load_game_team_snaps()
    if snaps.empty:
        return games
    by_game: dict[str, dict[str, dict[str, float]]] = {}
    for row in snaps.itertuples(index=False):
        gid = str(getattr(row, "game_id"))
        team = str(getattr(row, "team"))
        stats: dict[str, float] = {}
        for field in SNAP_FIELDS:
            raw = getattr(row, field, None)
            if raw is not None and pd.notna(raw):
                stats[field] = float(raw)
        by_game.setdefault(gid, {})[team] = stats
    for game in games:
        gid = str(game.get("game_id") or "")
        if not gid or gid not in by_game:
            continue
        teams = by_game[gid]
        for side, key in (("home", str(game.get("home") or "").lower()), ("away", str(game.get("away") or "").lower())):
            for field, value in (teams.get(key) or {}).items():
                game[f"{side}_{field}"] = value
    return games
