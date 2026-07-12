"""Load completed games from bundled nba/nhl/mlb team_data CSV archives."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from web.espn_client import ESPN_ABBR_ALIASES
from web.live_data import _parse_cutoff
from web.team_registry import load_team_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_LEAGUES = ("nba", "nhl", "mlb")
ARCHIVE_MAX_GAMES = 1200
GameTuple = tuple[str, str, str, str, int, int]


def _normalize_abbr(league: str, abbr: str) -> str:
    upper = abbr.upper()
    aliases = ESPN_ABBR_ALIASES.get(league, {})
    return aliases.get(upper, upper).lower()


def _slug_to_abbr(league: str, slug: str) -> str | None:
    registry = load_team_registry(league)
    for abbr, entry in registry.items():
        if len(entry) >= 2 and entry[1] == slug:
            return abbr
    return None


def _parse_archive_date(raw: str) -> datetime | None:
    text = raw.replace("/", "-").strip()
    for fmt in ("%m-%d-%Y", "%m-%d-%y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


@lru_cache(maxsize=8)
def load_archive_csv_games(league: str, cutoff_date: str) -> list[GameTuple]:
    """Parse historical team_data CSVs for calibration/backtest enrichment only."""
    league = league.lower()
    if league not in ARCHIVE_LEAGUES:
        return []

    cutoff = _parse_cutoff(cutoff_date)
    root = PROJECT_ROOT / league / "team_data"
    if not root.is_dir():
        return []

    merged: dict[str, GameTuple] = {}
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for csv_path in year_dir.glob("*.csv"):
            slug = csv_path.stem
            team_abbr = _slug_to_abbr(league, slug)
            if not team_abbr:
                continue
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if len(row) < 5:
                        continue
                    game_date = _parse_archive_date(row[0])
                    if game_date is None or game_date >= cutoff:
                        continue
                    opp_abbr = _normalize_abbr(league, row[1])
                    home_away = (row[2] or "").strip().lower()
                    try:
                        team_score = int(str(row[3]).split()[0])
                        opp_score = int(str(row[4]).split()[0])
                    except ValueError:
                        continue
                    team_name = slug.replace("-", " ").title()
                    opp_name = opp_abbr.upper()
                    if home_away == "home":
                        home_key, away_key = team_abbr, opp_abbr
                        home_name, away_name = team_name, opp_name
                    elif home_away == "away":
                        home_key, away_key = opp_abbr, team_abbr
                        home_name, away_name = opp_name, team_name
                    else:
                        continue
                    key = f"{game_date.date().isoformat()}:{home_key}:{away_key}"
                    merged[key] = (
                        home_key,
                        away_key,
                        home_name,
                        away_name,
                        team_score if home_away == "home" else opp_score,
                        opp_score if home_away == "home" else team_score,
                    )
    games = [
        game_tuple
        for _key, game_tuple in sorted(merged.items())
    ]
    # Cap keeps the newest window (keys are ISO date ascending).
    if len(games) > ARCHIVE_MAX_GAMES:
        return games[-ARCHIVE_MAX_GAMES:]
    return games
