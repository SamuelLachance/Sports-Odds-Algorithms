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
        by_game.setdefault(gid, {})[team] = {
            "epa_off": float(getattr(row, "epa_off")) if pd.notna(getattr(row, "epa_off", None)) else None,
            "epa_def": float(getattr(row, "epa_def")) if pd.notna(getattr(row, "epa_def", None)) else None,
            "sr_off": float(getattr(row, "sr_off")) if pd.notna(getattr(row, "sr_off", None)) else None,
            "sr_def": float(getattr(row, "sr_def")) if pd.notna(getattr(row, "sr_def", None)) else None,
            "explosive_off": (
                float(getattr(row, "explosive_off"))
                if pd.notna(getattr(row, "explosive_off", None))
                else None
            ),
            "pass_epa_off": (
                float(getattr(row, "pass_epa_off"))
                if pd.notna(getattr(row, "pass_epa_off", None))
                else None
            ),
        }
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
