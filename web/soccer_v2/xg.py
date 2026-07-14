"""Helpers to attach Understat rolling xG priors onto soccer match rows."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
XG_PATH = PROJECT_ROOT / "data" / "supplemental" / "soccer-xg" / "team_match_xg_priors.csv"

# football-data short names → Understat normalized keys
_NAME_ALIASES: dict[str, str] = {
    "mancity": "manchestercity",
    "manunited": "manchesterunited",
    "manutd": "manchesterunited",
    "newcastle": "newcastleunited",
    "nottmforest": "nottinghamforest",
    "wolves": "wolverhamptonwanderers",
    "spurs": "tottenham",
    "tottenhamhotspur": "tottenham",
    "westbrom": "westbromwichalbion",
    "qpr": "queensparkrangers",
    "sheffieldunited": "sheffieldunited",
    "sheffutd": "sheffieldunited",
    "sheffieldwednesday": "sheffieldwednesday",
    "sheffwed": "sheffieldwednesday",
    "nottinghamforest": "nottinghamforest",
    "parisg": "parissaintgermain",
    "psg": "parissaintgermain",
    "athmadrid": "atleticomadrid",
    "atleticomadrid": "atleticomadrid",
    "betis": "realbetis",
    "sociedad": "realsociedad",
    "vallecano": "rayovallecano",
    "inter": "inter",
    "verona": "hellasverona",
    "milan": "milan",
    "bayernmunich": "bayernmunich",
    "dortmund": "borussiadortmund",
    "frankfurt": "eintrachtfrankfurt",
    "mgladbach": "borussiamonchengladbach",
    "monchengladbach": "borussiamonchengladbach",
}


def _norm(name: str) -> str:
    key = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
    return _NAME_ALIASES.get(key, key)


@lru_cache(maxsize=1)
def _xg_index() -> dict[tuple[str, str, str], dict[str, float]]:
    if not XG_PATH.is_file():
        return {}
    frame = pd.read_csv(XG_PATH)
    if frame.empty:
        return {}
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in frame.to_dict(orient="records"):
        day = str(row.get("date") or "")[:10]
        home = _norm(row.get("home_key") or row.get("home") or "")
        away = _norm(row.get("away_key") or row.get("away") or "")
        if not (day and home and away):
            continue
        out[(day, home, away)] = {
            "home_xg_for": float(row["home_xg_for"]) if pd.notna(row.get("home_xg_for")) else None,  # type: ignore[dict-item]
            "away_xg_for": float(row["away_xg_for"]) if pd.notna(row.get("away_xg_for")) else None,  # type: ignore[dict-item]
            "home_xg_against": float(row["home_xg_against"]) if pd.notna(row.get("home_xg_against")) else None,  # type: ignore[dict-item]
            "away_xg_against": float(row["away_xg_against"]) if pd.notna(row.get("away_xg_against")) else None,  # type: ignore[dict-item]
        }
    return out  # type: ignore[return-value]


def attach_xg_to_row(row: dict[str, Any]) -> None:
    idx = _xg_index()
    if not idx:
        return
    day = str(row.get("date") or "")[:10]
    home = _norm(row.get("home") or "")
    away = _norm(row.get("away") or "")
    hit = idx.get((day, home, away))
    if not hit:
        # ±1 day skew
        try:
            from datetime import date, timedelta

            base = date.fromisoformat(day)
        except ValueError:
            return
        for delta in (-1, 1):
            alt = (base + timedelta(days=delta)).isoformat()
            hit = idx.get((alt, home, away))
            if hit:
                break
    if not hit:
        return
    for key, value in hit.items():
        if value is not None:
            row[key] = value


def attach_xg_to_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        attach_xg_to_row(row)
