"""Point-in-time injury burden join for NFL training (as-of snapshots)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INJ_DIR = PROJECT_ROOT / "data" / "supplemental" / "nfl-injuries"

# ESPN abbr -> nflverse / engine keys
_ABBR_ALIASES = {
    "wsh": "was",
    "was": "was",
    "lar": "lar",
    "la": "lar",
    "jac": "jax",
}


@lru_cache(maxsize=1)
def _load_injury_panel() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not INJ_DIR.is_dir():
        return pd.DataFrame()
    for path in sorted(INJ_DIR.glob("*.csv")):
        try:
            part = pd.read_csv(path)
        except (OSError, ValueError):
            continue
        if part.empty:
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
    frame["team"] = (
        frame["team"].astype(str).str.lower().str.strip().map(lambda t: _ABBR_ALIASES.get(t, t))
    )
    return frame.sort_values(["team", "date"]).reset_index(drop=True)


def _asof_row(panel: pd.DataFrame, team: str, day: str) -> dict[str, float]:
    """Latest snapshot strictly before game day (fail closed → zeros)."""
    empty = {"injuries": 0.0, "out": 0.0, "ol_out": 0.0, "skill_out": 0.0}
    if panel.empty:
        return empty
    key = _ABBR_ALIASES.get(team, team)
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
    panel = _load_injury_panel()
    if panel.empty:
        for game in games:
            for side in ("home", "away"):
                game[f"{side}_injuries"] = 0.0
                game[f"{side}_out"] = 0.0
                game[f"{side}_ol_out"] = 0.0
                game[f"{side}_skill_out"] = 0.0
            game["injury_known"] = 0.0
        return games
    for game in games:
        day = str(game.get("date") or "")[:10]
        known = 0.0
        for side, key in (("home", str(game.get("home") or "").lower()), ("away", str(game.get("away") or "").lower())):
            stats = _asof_row(panel, key, day)
            for field, value in stats.items():
                game[f"{side}_{field}"] = value
            if any(stats.values()):
                known = 1.0
        game["injury_known"] = known
    return games
