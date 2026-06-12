"""League-wide completed game collection from ESPN for power ratings."""

from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

from web.espn_client import (
    ESPN_ABBR_ALIASES,
    current_season_year,
    fetch_team_schedule,
    prior_season_year,
)
from web.league_profiles import (
    FRIENDLIES_SUPPLEMENT_LEAGUE,
    INTERNATIONAL_TOURNAMENT_LEAGUES,
    LARGE_ROSTER_LEAGUES,
    MIN_GAMES_FOR_POWER,
    POWER_SCOREBOARD_LOOKBACK,
    get_league_profile,
)
from web.live_data import _event_before_cutoff, _parse_cutoff, _score_value

GameTuple = tuple[str, str, str, str, int, int]
GameMap = dict[str, GameTuple]


def _normalize_abbr(league: str, espn_abbr: str) -> str:
    upper = espn_abbr.upper()
    aliases = ESPN_ABBR_ALIASES.get(league, {})
    return aliases.get(upper, upper).lower()


def _parse_event_to_game(
    league: str,
    event: dict[str, Any],
    cutoff: datetime,
) -> tuple[str, GameTuple] | None:
    if not _event_before_cutoff(event, cutoff):
        return None

    event_id = str(event.get("id") or "")
    if not event_id:
        return None

    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    away_comp = next((c for c in competitors if c.get("homeAway") == "away"), None)
    home_comp = next((c for c in competitors if c.get("homeAway") == "home"), None)
    if not away_comp or not home_comp:
        return None

    away_abbr = _normalize_abbr(
        league, (away_comp.get("team") or {}).get("abbreviation", "")
    )
    home_abbr = _normalize_abbr(
        league, (home_comp.get("team") or {}).get("abbreviation", "")
    )
    if not away_abbr or not home_abbr:
        return None

    away_name = (away_comp.get("team") or {}).get("displayName") or away_abbr
    home_name = (home_comp.get("team") or {}).get("displayName") or home_abbr
    away_score = _score_value(away_comp)
    home_score = _score_value(home_comp)

    return event_id, (
        home_abbr,
        away_abbr,
        home_name,
        away_name,
        home_score,
        away_score,
    )


def _fetch_scoreboard_payload(league: str, day: date) -> dict[str, Any]:
    profile = get_league_profile(league)
    date_param = day.strftime("%Y%m%d")
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{profile['sport_path']}/scoreboard?dates={date_param}"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _collect_from_scoreboards(
    league: str,
    cutoff: datetime,
    lookback_days: int,
) -> GameMap:
    """Collect completed games by walking daily scoreboards (fast for large rosters)."""
    games: GameMap = {}
    cutoff_day = date(cutoff.year, cutoff.month, cutoff.day)

    for offset in range(1, lookback_days + 1):
        day = cutoff_day - timedelta(days=offset)
        try:
            payload = _fetch_scoreboard_payload(league, day)
        except OSError:
            continue

        for event in payload.get("events") or []:
            parsed = _parse_event_to_game(league, event, cutoff)
            if parsed:
                event_id, game_tuple = parsed
                games[event_id] = game_tuple

    return games


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
) -> GameMap:
    """Return event_id -> game tuple before cutoff."""
    games: GameMap = {}
    for event in fetch_team_schedule(league, espn_team_id, season):
        parsed = _parse_event_to_game(league, event, cutoff)
        if parsed:
            event_id, game_tuple = parsed
            games[event_id] = game_tuple
    return games


def _collect_from_team_schedules(
    league: str,
    cutoff: datetime,
    cutoff_day: date,
) -> GameMap:
    """Collect games from per-team schedules (efficient for small pro leagues)."""
    current_season = current_season_year(league, cutoff_day)
    prior_season = prior_season_year(league, cutoff_day)

    merged: GameMap = {}
    for espn_id, _abbr, _label in _load_espn_team_ids(league):
        for season in (current_season, prior_season):
            merged.update(_collect_events_for_season(league, espn_id, season, cutoff))
    return merged


def _scoreboard_lookback_days(league: str) -> int:
    return POWER_SCOREBOARD_LOOKBACK.get(league, POWER_SCOREBOARD_LOOKBACK["default"])


def _load_friendlies_supplement(cutoff_date: str) -> list[GameTuple]:
    """Recent FIFA friendlies for sparse international tournaments."""
    cutoff = _parse_cutoff(cutoff_date)
    lookback = _scoreboard_lookback_days(FRIENDLIES_SUPPLEMENT_LEAGUE)
    return list(_collect_from_scoreboards(FRIENDLIES_SUPPLEMENT_LEAGUE, cutoff, lookback).values())


def load_league_completed_games(
    league: str,
    cutoff_date: str,
) -> list[GameTuple]:
    """
    Load deduplicated completed games for a league before cutoff.

    Returns list of (home_key, away_key, home_name, away_name, home_score, away_score).
    """
    league = league.lower()
    cutoff = _parse_cutoff(cutoff_date)
    cutoff_day = date(cutoff.year, cutoff.month, cutoff.day)

    if league in LARGE_ROSTER_LEAGUES:
        merged = _collect_from_scoreboards(league, cutoff, _scoreboard_lookback_days(league))
    else:
        merged = _collect_from_team_schedules(league, cutoff, cutoff_day)

    if league in INTERNATIONAL_TOURNAMENT_LEAGUES:
        for index, game in enumerate(_load_friendlies_supplement(cutoff_date)):
            merged[f"friendlies:{index}:{game[0]}:{game[1]}"] = game

    return list(merged.values())


def power_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    """Human-readable reason when power blend cannot run."""
    league = league.lower()
    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_GAMES_FOR_POWER:
        return (
            f"Insufficient completed games ({len(games)} < {MIN_GAMES_FOR_POWER}) "
            "— likely off-season or sparse schedule."
        )

    from web.power_model import build_power_ratings, predict_matchup

    teams, _total, param = build_power_ratings(games)
    if param is None:
        return "Could not fit power-rating logistic curve on available games."

    home_key = _normalize_abbr(league, home_abbr)
    away_key = _normalize_abbr(league, away_abbr)
    if not predict_matchup(teams, param, home_key, away_key):
        missing = []
        if home_key not in teams:
            missing.append(home_key)
        if away_key not in teams:
            missing.append(away_key)
        if missing:
            return f"Teams not found in power ratings: {', '.join(missing)}."
        return "Teams have insufficient games in the power-rating sample."
    return "Power model unavailable."


@lru_cache(maxsize=32)
def get_league_power_context(
    league: str,
    cutoff_date: str,
):
    """Cached power ratings for a league at a cutoff date."""
    from web.power_model import build_power_ratings

    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_GAMES_FOR_POWER:
        return None

    teams, _total, param = build_power_ratings(games)
    if param is None:
        return None

    return teams, games, param


def prewarm_league_power(league: str, cutoff_date: str) -> None:
    """Load/cache power ratings once per league before per-game slate analysis."""
    get_league_power_context(league, cutoff_date)
