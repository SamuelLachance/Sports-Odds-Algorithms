"""Build in-memory team datasets from ESPN schedules for live predictions."""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from web.espn_client import (  # noqa: E402
    ESPN_ABBR_ALIASES,
    fetch_team_schedule,
    guess_season_years,
    iso_to_project_date,
)

NUM_PERIODS = {"nba": 4, "nhl": 3, "mlb": 9}


def _ensure_project_root() -> None:
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _normalize_abbr(league: str, espn_abbr: str) -> str:
    upper = espn_abbr.upper()
    aliases = ESPN_ABBR_ALIASES.get(league, {})
    return aliases.get(upper, upper).lower()


@lru_cache(maxsize=8)
def _load_team_registry(league: str) -> dict[str, list[str]]:
    _ensure_project_root()
    from universal_functions import Universal_Functions

    universal = Universal_Functions(league)
    registry: dict[str, list[str]] = {}
    for abbr, slug in universal.load_league_teams():
        registry[abbr.lower()] = [abbr, slug]
        registry[slug] = [abbr, slug]
    return registry


def resolve_team(
    league: str,
    espn_abbr: str,
    display_name: str | None = None,
) -> list[str] | None:
    registry = _load_team_registry(league)
    normalized = _normalize_abbr(league, espn_abbr)
    if normalized in registry:
        return registry[normalized]

    slug_source = display_name or normalized
    slug = slug_source.lower().replace(".", "").replace("'", "").replace(" ", "-")
    return [normalized, slug]


def _parse_cutoff(cutoff_date: str) -> datetime:
    month, day, year = cutoff_date.split("-")
    return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)


def _event_before_cutoff(event: dict[str, Any], cutoff: datetime) -> bool:
    competition = (event.get("competitions") or [{}])[0]
    status = (competition.get("status") or {}).get("type") or {}
    if not status.get("completed"):
        return False

    event_date = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    if event_date.tzinfo is None:
        event_date = event_date.replace(tzinfo=timezone.utc)
    return event_date < cutoff


def _build_team_entry(
    league: str,
    team_abbr: str,
    espn_team_id: str,
    cutoff_date: str,
) -> list[dict[str, Any]]:
    cutoff = _parse_cutoff(cutoff_date)
    cutoff_day = date(cutoff.year, cutoff.month, cutoff.day)
    seasons = guess_season_years(league, cutoff_day)

    dates: list[str] = []
    opponents: list[str] = []
    home_away: list[str] = []
    game_scores: list[list[int]] = []
    period_scores: list[list[list[int]]] = []
    years_used: list[str] = []

    for season in seasons:
        events = fetch_team_schedule(league, espn_team_id, season)
        year_label = str(season)
        year_has_games = False

        for event in events:
            if not _event_before_cutoff(event, cutoff):
                continue

            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors") or []
            team_comp = next(
                (
                    c
                    for c in competitors
                    if _normalize_abbr(league, (c.get("team") or {}).get("abbreviation", ""))
                    == team_abbr.lower()
                ),
                None,
            )
            opp_comp = next(
                (
                    c
                    for c in competitors
                    if c is not team_comp
                ),
                None,
            )
            if not team_comp or not opp_comp:
                continue

            opp_abbr = _normalize_abbr(
                league, (opp_comp.get("team") or {}).get("abbreviation", "")
            )
            team_score = int((team_comp.get("score") or {}).get("value") or 0)
            opp_score = int((opp_comp.get("score") or {}).get("value") or 0)
            dates.append(iso_to_project_date(event["date"]))
            opponents.append(opp_abbr)
            home_away.append("home" if team_comp.get("homeAway") == "home" else "away")
            game_scores.append([team_score, opp_score])

            periods = NUM_PERIODS[league]
            period_scores.append([[0] * periods, [0] * periods])
            year_has_games = True

        if year_has_games:
            years_used.append(year_label)

    if not dates:
        return []

    year_key = years_used[-1] if years_used else str(cutoff.year)
    return [
        {
            "year": year_key,
            "dates": dates,
            "other_team": opponents,
            "home_away": home_away,
            "game_scores": game_scores,
            "period_scores": period_scores,
        }
    ]


def load_live_team_data(
    league: str,
    team: list[str],
    espn_team_id: str,
    cutoff_date: str,
) -> list[dict[str, Any]]:
    return _build_team_entry(league, team[0], espn_team_id, cutoff_date)
