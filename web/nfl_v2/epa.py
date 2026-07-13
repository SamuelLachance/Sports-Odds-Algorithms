"""Helpers to attach nflverse PBP EPA aggregates onto NFL game dicts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EPA_PATHS = (
    PROJECT_ROOT / "data" / "supplemental" / "nfl-pbp" / "game_team_epa.csv",
    PROJECT_ROOT / ".build-cache" / "nfl-pbp" / "game_team_epa.csv",
)

EPA_FIELDS = (
    "epa_off",
    "epa_def",
    "sr_off",
    "sr_def",
    "explosive_off",
    "pass_epa_off",
    "rush_epa_off",
    "sack_rate_off",
    "sack_rate_def",
    "qb_hit_rate_off",
    "qb_hit_rate_def",
    "early_down_epa_off",
    "third_down_sr_off",
    "redzone_epa_off",
)


@lru_cache(maxsize=1)
def load_game_team_epa() -> pd.DataFrame:
    for path in EPA_PATHS:
        if path.is_file():
            frame = pd.read_csv(path)
            frame["team"] = frame["team"].astype(str).str.lower().str.strip()
            frame["game_id"] = frame["game_id"].astype(str)
            return frame
    return pd.DataFrame()


def attach_epa_to_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mutate game dicts with home_/away_ EPA fields when game_id is known."""
    epa = load_game_team_epa()
    if epa.empty:
        return games
    by_game: dict[str, dict[str, dict[str, float]]] = {}
    for row in epa.itertuples(index=False):
        gid = str(getattr(row, "game_id"))
        team = str(getattr(row, "team"))
        stats: dict[str, float | None] = {}
        for field in EPA_FIELDS:
            raw = getattr(row, field, None)
            stats[field] = float(raw) if raw is not None and pd.notna(raw) else None
        by_game.setdefault(gid, {})[team] = stats  # type: ignore[assignment]
    for game in games:
        gid = str(game.get("game_id") or "")
        if not gid or gid not in by_game:
            continue
        teams = by_game[gid]
        home = str(game.get("home") or "").lower()
        away = str(game.get("away") or "").lower()
        for side, key in (("home", home), ("away", away)):
            stats = teams.get(key) or {}
            for field, value in stats.items():
                if value is not None:
                    game[f"{side}_{field}"] = value
    return games
