"""Point-in-time injury burden join for NFL training.

Sources (no NA fills — missing → injury_known=0):
  1. nflverse weekly releases → data/supplemental/nfl-injuries/weekly_team_burden.csv
  2. Optional daily ESPN snapshots (*.csv / *.json) for live as-of joins
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INJ_DIR = PROJECT_ROOT / "data" / "supplemental" / "nfl-injuries"
WEEKLY_PATH = INJ_DIR / "weekly_team_burden.csv"

_ABBR_ALIASES = {
    "wsh": "was",
    "was": "was",
    "lar": "la",
    "la": "la",
    "jac": "jax",
}


def _canon(team: str) -> str:
    key = str(team or "").lower().strip()
    return _ABBR_ALIASES.get(key, key)


@lru_cache(maxsize=1)
def _load_weekly_burden() -> dict[tuple[int, int, str], dict[str, float]]:
    if not WEEKLY_PATH.is_file():
        return {}
    try:
        frame = pd.read_csv(WEEKLY_PATH)
    except (OSError, ValueError):
        return {}
    out: dict[tuple[int, int, str], dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        try:
            season = int(getattr(row, "season"))
            week = int(getattr(row, "week"))
        except (TypeError, ValueError):
            continue
        team = _canon(str(getattr(row, "team") or ""))
        if not team:
            continue
        out[(season, week, team)] = {
            "injuries": float(getattr(row, "injuries") or 0.0),
            "out": float(getattr(row, "out") or 0.0),
            "ol_out": float(getattr(row, "ol_out") or 0.0),
            "skill_out": float(getattr(row, "skill_out") or 0.0),
        }
    return out


@lru_cache(maxsize=1)
def _load_daily_panel() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not INJ_DIR.is_dir():
        return pd.DataFrame()
    for path in sorted(INJ_DIR.glob("*.csv")):
        if path.name == "weekly_team_burden.csv":
            continue
        try:
            part = pd.read_csv(path)
        except (OSError, ValueError):
            continue
        if part.empty or "date" not in part.columns:
            continue
        rows.append(part)
    for path in sorted(INJ_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        day = str(payload.get("as_of") or path.stem)[:10]
        for team in payload.get("teams") or []:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "date": day,
                            "team": str(team.get("team") or "").lower(),
                            "injuries": team.get("injuries", 0),
                            "out": team.get("out", 0),
                            "ol_out": team.get("ol_out", 0),
                            "skill_out": team.get("skill_out", 0),
                        }
                    ]
                )
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    frame["date"] = frame["date"].astype(str).str[:10]
    frame["team"] = frame["team"].astype(str).str.lower().str.strip().map(_canon)
    return frame.sort_values(["team", "date"]).reset_index(drop=True)


def _asof_daily(panel: pd.DataFrame, team: str, day: str) -> dict[str, float] | None:
    empty = None
    if panel.empty:
        return empty
    key = _canon(team)
    sub = panel[(panel["team"] == key) & (panel["date"] < day)]
    if sub.empty:
        return empty
    row = sub.iloc[-1]
    return {
        "injuries": float(row.get("injuries") or 0),
        "out": float(row.get("out") or 0),
        "ol_out": float(row.get("ol_out") or 0),
        "skill_out": float(row.get("skill_out") or 0),
    }


def attach_injuries_to_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weekly = _load_weekly_burden()
    daily = _load_daily_panel()
    for game in games:
        day = str(game.get("date") or "")[:10]
        season = int(game.get("season") or 0)
        week = int(game.get("week") or 0)
        known = 0.0
        for side, key in (
            ("home", str(game.get("home") or "").lower()),
            ("away", str(game.get("away") or "").lower()),
        ):
            stats = None
            if season and week and key:
                stats = weekly.get((season, week, _canon(key)))
            if stats is None:
                stats = _asof_daily(daily, key, day)
            if stats is None:
                stats = {"injuries": 0.0, "out": 0.0, "ol_out": 0.0, "skill_out": 0.0}
            else:
                known = 1.0
            for field, value in stats.items():
                game[f"{side}_{field}"] = float(value)
        game["injury_known"] = known
    return games
