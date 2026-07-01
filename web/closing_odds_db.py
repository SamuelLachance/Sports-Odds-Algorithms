"""Local closing-odds database for walk-forward backtests and CLV grading."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.football_data_uk import load_football_data_uk_games
from web.league_profiles import is_soccer_league

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ODDS_DIR = PROJECT_ROOT / "data" / "supplemental" / "closing-odds"
TEAM_MAP_PATH = PROJECT_ROOT / "data" / "_meta" / "closing_odds_teams.json"

US_ODDS_LEAGUES = frozenset({"nba", "nfl", "nhl", "mlb", "wnba", "cbb", "cfb", "ncaah", "ncaabb"})


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _team_maps() -> dict[str, dict[str, str]]:
    if not TEAM_MAP_PATH.is_file():
        return {}
    try:
        payload = json.loads(TEAM_MAP_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        league: {key: value.lower() for key, value in mapping.items()}
        for league, mapping in payload.items()
        if isinstance(mapping, dict)
    }


def normalize_team_key(league: str, label: str) -> str | None:
    league = league.lower()
    label = str(label or "").strip()
    if not label:
        return None
    mapping = _team_maps().get(league, {})
    if label in mapping:
        return mapping[label]
    lowered = label.lower()
    if lowered in mapping.values():
        return lowered
    if label.upper() in mapping:
        return mapping[label.upper()]
    return lowered.replace(" ", "")


def _iso_date(raw: str) -> str:
    raw = str(raw or "").strip()
    if not raw:
        return ""
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw[:10]


@lru_cache(maxsize=16)
def _load_us_odds_index(league: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    league = league.lower()
    path = ODDS_DIR / f"{league}.csv"
    if not path.is_file():
        return {}
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            game_date = _iso_date(row.get("date", ""))
            home = str(row.get("home_key", "")).lower()
            away = str(row.get("away_key", "")).lower()
            if not game_date or not home or not away:
                continue
            index[(game_date, home, away)] = {
                "home_close_ml": _parse_int(row.get("home_close_ml")),
                "away_close_ml": _parse_int(row.get("away_close_ml")),
                "home_close_spread": _parse_float(row.get("home_close_spread")),
                "away_close_spread": _parse_float(row.get("away_close_spread")),
                "home_spread_odds": _parse_int(row.get("home_spread_odds")),
                "away_spread_odds": _parse_int(row.get("away_spread_odds")),
                "source": row.get("source") or "closing-odds-db",
            }
    return index


@lru_cache(maxsize=8)
def _load_soccer_odds_index(league: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    league = league.lower()
    if league not in {"epl", "bundesliga", "laliga", "seriea", "ligue1"}:
        return {}
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in load_football_data_uk_games(league):
        game_date = record["date"]
        home = record["home_key"]
        away = record["away_key"]
        index[(game_date, home, away)] = {
            "home_close_ml": record.get("home_odds"),
            "draw_close_ml": record.get("draw_odds"),
            "away_close_ml": record.get("away_odds"),
            "source": "football-data.co.uk",
        }
    return index


def closing_odds_lookup(
    league: str,
    game_date: str,
    home_key: str,
    away_key: str,
) -> dict[str, Any] | None:
    """Return closing market lines for a completed game when cached locally."""
    league = league.lower()
    home = home_key.lower()
    away = away_key.lower()
    game_date = _iso_date(game_date)
    if is_soccer_league(league):
        return _load_soccer_odds_index(league).get((game_date, home, away))
    return _load_us_odds_index(league).get((game_date, home, away))


def closing_odds_coverage(league: str) -> dict[str, Any]:
    league = league.lower()
    if is_soccer_league(league):
        count = len(_load_soccer_odds_index(league))
        source = "football-data.co.uk"
    else:
        count = len(_load_us_odds_index(league))
        source = f"data/supplemental/closing-odds/{league}.csv"
    return {"league": league, "rows": count, "source": source}


def clear_closing_odds_cache() -> None:
    _team_maps.cache_clear()
    _load_us_odds_index.cache_clear()
    _load_soccer_odds_index.cache_clear()
