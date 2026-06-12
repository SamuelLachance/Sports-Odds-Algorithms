"""League-wide completed game collection from ESPN for power ratings."""

from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime
from functools import lru_cache
from typing import Any

from web.espn_client import ESPN_ABBR_ALIASES, current_season_year, fetch_team_schedule, prior_season_year
from web.league_profiles import MIN_GAMES_FOR_MODEL, get_league_profile
from web.live_data import _event_before_cutoff, _parse_cutoff, _score_value


def _normalize_abbr(league: str, espn_abbr: str) -> str:
    upper = espn_abbr.upper()
    aliases = ESPN_ABBR_ALIASES.get(league, {})
    return aliases.get(upper, upper).lower()


def _load_espn_team_ids(league: str) -> list[tuple[str, str, str]]:
    """Fetch ESPN team ids from the teams API payload."""
    profile = get_league_profile(league)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{profile['sport_path']}/teams?limit=500"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except OSError:
        return []

    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add_team(team: dict[str, Any]) -> None:
        team_id = str(team.get("id") or "")
        abbr = (team.get("abbreviation") or team.get("shortDisplayName") or "").lower()
        name = team.get("displayName") or team.get("name") or abbr
        if not team_id or not abbr or team_id in seen:
            return
        seen.add(team_id)
        rows.append((team_id, abbr, name))

    for entry in payload.get("teams") or []:
        add_team(entry.get("team") or entry)

    for sport in payload.get("sports") or []:
        for league_block in sport.get("leagues") or []:
            for entry in league_block.get("teams") or []:
                add_team(entry.get("team") or entry)

    return rows


def _collect_events_for_season(
    league: str,
    espn_team_id: str,
    season: int,
    cutoff: datetime,
) -> dict[str, tuple[str, str, str, str, int, int]]:
    """Return event_id -> game tuple before cutoff."""
    games: dict[str, tuple[str, str, str, str, int, int]] = {}
    for event in fetch_team_schedule(league, espn_team_id, season):
        if not _event_before_cutoff(event, cutoff):
            continue
        event_id = str(event.get("id") or "")
        if not event_id or event_id in games:
            continue

        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors") or []
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), None)
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), None)
        if not away_comp or not home_comp:
            continue

        away_abbr = _normalize_abbr(
            league, (away_comp.get("team") or {}).get("abbreviation", "")
        )
        home_abbr = _normalize_abbr(
            league, (home_comp.get("team") or {}).get("abbreviation", "")
        )
        away_name = (away_comp.get("team") or {}).get("displayName") or away_abbr
        home_name = (home_comp.get("team") or {}).get("displayName") or home_abbr
        away_score = _score_value(away_comp)
        home_score = _score_value(home_comp)

        games[event_id] = (
            home_abbr,
            away_abbr,
            home_name,
            away_name,
            home_score,
            away_score,
        )
    return games


def load_league_completed_games(
    league: str,
    cutoff_date: str,
) -> list[tuple[str, str, str, str, int, int]]:
    """
    Load deduplicated completed games for a league before cutoff.

    Returns list of (home_key, away_key, home_name, away_name, home_score, away_score).
    """
    league = league.lower()
    cutoff = _parse_cutoff(cutoff_date)
    cutoff_day = date(cutoff.year, cutoff.month, cutoff.day)
    current_season = current_season_year(league, cutoff_day)
    prior_season = prior_season_year(league, cutoff_day)

    merged: dict[str, tuple[str, str, str, str, int, int]] = {}
    enriched_ids = _load_espn_team_ids(league)

    for espn_id, _abbr, _label in enriched_ids:
        for season in (current_season, prior_season):
            season_games = _collect_events_for_season(league, espn_id, season, cutoff)
            merged.update(season_games)

    return list(merged.values())


@lru_cache(maxsize=32)
def get_league_power_context(
    league: str,
    cutoff_date: str,
):
    """Cached power ratings for a league at a cutoff date."""
    from web.power_model import build_power_ratings

    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_GAMES_FOR_MODEL:
        return None

    teams, _total, param = build_power_ratings(games)
    if param is None:
        return None

    return teams, games, param
