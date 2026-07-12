"""Local closing-odds database for walk-forward backtests and CLV grading."""

from __future__ import annotations

import csv
import json
from datetime import date as date_cls, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.football_data_uk import load_football_data_uk_games
from web.international_soccer_odds import load_international_odds_index
from web.league_profiles import INTERNATIONAL_SOCCER_LEAGUES, is_soccer_league
from web.league_profiles import is_soccer_league

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ODDS_DIR = PROJECT_ROOT / "data" / "supplemental" / "closing-odds"
TEAM_MAP_PATH = PROJECT_ROOT / "data" / "_meta" / "closing_odds_teams.json"

US_ODDS_LEAGUES = frozenset({"nba", "nfl", "nhl", "mlb", "wnba", "cbb", "cfb", "ncaah", "ncaabb"})

# SBR / legacy NHL CSV keys -> ESPN abbreviations used in scoreboards.
NHL_ODDS_KEY_ALIASES: dict[str, str] = {
    "st.louis": "stl",
    "phoenix": "ari",
    "arizonas": "ari",
    "winnipegjets": "wpg",
    "tampabay": "tb",
    "tampa": "tb",
    "nyislanders": "nyi",
    "seattlekraken": "sea",
    "uta": "uta",
}

# SBR archive keys -> ESPN abbreviations used in scoreboards / datasets.
MLB_ODDS_KEY_ALIASES: dict[str, str] = {
    "sfo": "sf",
    "sdg": "sd",
    "sfg": "sf",
    "kan": "kc",
    "los": "lad",
    "cub": "chc",
    "brs": "bos",
    "tam": "tb",
    "ath": "oak",
    "cws": "chw",
    "was": "wsh",
}


def _canonical_odds_team_key(league: str, key: str) -> str:
    league = league.lower()
    key = str(key or "").strip().lower()
    if league == "nhl":
        return NHL_ODDS_KEY_ALIASES.get(key, key)
    if league == "mlb":
        return MLB_ODDS_KEY_ALIASES.get(key, key)
    return key


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    # ESPN/SBR EVEN money arrives as 0 — normalize like live odds paths.
    if number == 0:
        return 100
    # American odds require |x| >= 100; reject garbage magnitudes.
    if abs(number) < 100:
        return None
    return number


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


def _flip_two_way_odds(row: dict[str, Any]) -> dict[str, Any]:
    """Swap home/away orientation for a matched odds row."""
    flipped = dict(row)
    flipped["home_close_ml"] = row.get("away_close_ml")
    flipped["away_close_ml"] = row.get("home_close_ml")
    flipped["home_open_ml"] = row.get("away_open_ml")
    flipped["away_open_ml"] = row.get("home_open_ml")
    home_spread = row.get("home_close_spread")
    away_spread = row.get("away_close_spread")
    if home_spread is not None:
        flipped["home_close_spread"] = -float(home_spread)
    if away_spread is not None:
        flipped["away_close_spread"] = -float(away_spread)
    flipped["home_spread_odds"] = row.get("away_spread_odds")
    flipped["away_spread_odds"] = row.get("home_spread_odds")
    return flipped


def _lookup_us_odds_row(
    index: dict[tuple[str, str, str], dict[str, Any]],
    game_date: str,
    home: str,
    away: str,
) -> dict[str, Any] | None:
    """Exact or +/-1 day match; tolerate same-day home/away swaps only.

    Fuzzy ±1 day is for timezone/date-label drift on the *same* matchup.
    Combining a fuzzy date with a home/away flip can attach the wrong game's
    line on back-to-backs (adjacent day, flipped venue).
    """
    try:
        base_date = date_cls.fromisoformat(game_date)
    except ValueError:
        base_date = None

    direct = index.get((game_date, home, away))
    if direct:
        return dict(direct)
    swapped = index.get((game_date, away, home))
    if swapped:
        return _flip_two_way_odds(swapped)

    if base_date is None:
        return None
    for offset in (-1, 1):
        candidate_date = (base_date + timedelta(days=offset)).isoformat()
        fuzzy = index.get((candidate_date, home, away))
        if fuzzy:
            return dict(fuzzy)
    return None


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
            home = _canonical_odds_team_key(league, row.get("home_key", ""))
            away = _canonical_odds_team_key(league, row.get("away_key", ""))
            if not game_date or not home or not away:
                continue
            index[(game_date, home, away)] = {
                "home_close_ml": _parse_int(row.get("home_close_ml")),
                "away_close_ml": _parse_int(row.get("away_close_ml")),
                "home_open_ml": _parse_int(row.get("home_open_ml")),
                "away_open_ml": _parse_int(row.get("away_open_ml")),
                "home_close_spread": _parse_float(row.get("home_close_spread")),
                "away_close_spread": _parse_float(row.get("away_close_spread")),
                "home_spread_odds": _parse_int(row.get("home_spread_odds")),
                "away_spread_odds": _parse_int(row.get("away_spread_odds")),
                "close_total": _parse_float(row.get("close_total")),
                "open_total": _parse_float(row.get("open_total")),
                "source": row.get("source") or "closing-odds-db",
            }
    return index


@lru_cache(maxsize=16)
def _load_soccer_odds_index(league: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    league = league.lower()
    if league in INTERNATIONAL_SOCCER_LEAGUES or league == "fifa_friendlies":
        index: dict[tuple[str, str, str], dict[str, Any]] = {}
        for (game_date, home, away), odds in load_international_odds_index().items():
            index[(game_date, home, away)] = {
                "home_close_ml": odds.get("home_odds"),
                "draw_close_ml": odds.get("draw_odds"),
                "away_close_ml": odds.get("away_odds"),
                "source": "football-data.co.uk/international",
            }
        return index
    if league not in {"epl", "bundesliga", "laliga", "seriea", "ligue1"}:
        return {}
    index = {}
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
    home = _canonical_odds_team_key(league, home_key)
    away = _canonical_odds_team_key(league, away_key)
    game_date = _iso_date(game_date)
    if is_soccer_league(league):
        return _load_soccer_odds_index(league).get((game_date, home, away))
    return _lookup_us_odds_row(_load_us_odds_index(league), game_date, home, away)


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
