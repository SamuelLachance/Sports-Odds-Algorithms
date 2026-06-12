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
    current_season_year,
    fetch_team_schedule,
    iso_to_project_date,
    prior_season_year,
)
from web.league_profiles import (  # noqa: E402
    FRIENDLIES_SUPPLEMENT_LEAGUE,
    INTERNATIONAL_TOURNAMENT_LEAGUES,
    MIN_GAMES_FOR_MODEL,
    NUM_PERIODS,
)
from web.team_registry import load_team_registry  # noqa: E402

GameRows = tuple[list[str], list[str], list[str], list[list[int]], list[list[list[int]]]]


def _ensure_project_root() -> None:
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _normalize_abbr(league: str, espn_abbr: str) -> str:
    upper = espn_abbr.upper()
    aliases = ESPN_ABBR_ALIASES.get(league, {})
    return aliases.get(upper, upper).lower()


@lru_cache(maxsize=32)
def _load_team_registry(league: str) -> dict[str, list[str]]:
    return load_team_registry(league)


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


def _score_value(comp: dict[str, Any]) -> int:
    raw = (comp.get("score") or {}).get("value")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _collect_season_games(
    league: str,
    team_abbr: str,
    espn_team_id: str,
    season: int,
    cutoff: datetime,
) -> GameRows:
    events = fetch_team_schedule(league, espn_team_id, season)

    dates: list[str] = []
    opponents: list[str] = []
    home_away: list[str] = []
    game_scores: list[list[int]] = []
    period_scores: list[list[list[int]]] = []

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
        opp_comp = next((c for c in competitors if c is not team_comp), None)
        if not team_comp or not opp_comp:
            continue

        opp_abbr = _normalize_abbr(
            league, (opp_comp.get("team") or {}).get("abbreviation", "")
        )
        team_score = _score_value(team_comp)
        opp_score = _score_value(opp_comp)
        dates.append(iso_to_project_date(event["date"]))
        opponents.append(opp_abbr)
        home_away.append("home" if team_comp.get("homeAway") == "home" else "away")
        game_scores.append([team_score, opp_score])

        periods = NUM_PERIODS[league]
        period_scores.append([[0] * periods, [0] * periods])

    return dates, opponents, home_away, game_scores, period_scores


def _merge_game_rows(left: GameRows, right: GameRows) -> GameRows:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
        left[4] + right[4],
    )


def _parse_row_date(date_str: str) -> date:
    month, day, year = date_str.split("-")
    return date(int(year), int(month), int(day))


def _sort_game_rows(rows: GameRows) -> GameRows:
    dates, opponents, home_away, game_scores, period_scores = rows
    if not dates:
        return rows
    order = sorted(range(len(dates)), key=lambda index: _parse_row_date(dates[index]))
    return (
        [dates[index] for index in order],
        [opponents[index] for index in order],
        [home_away[index] for index in order],
        [game_scores[index] for index in order],
        [period_scores[index] for index in order],
    )


def _walk_back_season_games(
    league: str,
    team_abbr: str,
    espn_team_id: str,
    current_season: int,
    cutoff: datetime,
    *,
    start_offset: int = 0,
    max_offset: int = 8,
) -> tuple[GameRows, list[str]]:
    merged: GameRows = ([], [], [], [], [])
    seasons_used: list[str] = []
    for offset in range(start_offset, max_offset + 1):
        season = current_season - offset
        season_rows = _collect_season_games(
            league, team_abbr, espn_team_id, season, cutoff
        )
        if not season_rows[0]:
            continue
        merged = _merge_game_rows(season_rows, merged)
        seasons_used.append(str(season))
        if len(merged[0]) >= MIN_GAMES_FOR_MODEL:
            break
    return merged, seasons_used


def _build_team_entry(
    league: str,
    team_abbr: str,
    espn_team_id: str,
    cutoff_date: str,
) -> list[dict[str, Any]]:
    cutoff = _parse_cutoff(cutoff_date)
    cutoff_day = date(cutoff.year, cutoff.month, cutoff.day)
    current_season = current_season_year(league, cutoff_day)
    prior_season = prior_season_year(league, cutoff_day)

    current_rows = _collect_season_games(
        league, team_abbr, espn_team_id, current_season, cutoff
    )
    dates, opponents, home_away, game_scores, period_scores = current_rows
    seasons_used = [str(current_season)] if dates else []

    if len(dates) < MIN_GAMES_FOR_MODEL:
        prior_rows = _collect_season_games(
            league, team_abbr, espn_team_id, prior_season, cutoff
        )
        if prior_rows[0]:
            dates, opponents, home_away, game_scores, period_scores = _merge_game_rows(
                prior_rows, current_rows
            )
            seasons_used = [str(prior_season)]
            if current_rows[0]:
                seasons_used.append(str(current_season))

    if league in INTERNATIONAL_TOURNAMENT_LEAGUES and len(dates) < MIN_GAMES_FOR_MODEL:
        older_rows, older_seasons = _walk_back_season_games(
            league,
            team_abbr,
            espn_team_id,
            current_season,
            cutoff,
            start_offset=2,
        )
        if older_seasons:
            dates, opponents, home_away, game_scores, period_scores = _merge_game_rows(
                older_rows,
                (dates, opponents, home_away, game_scores, period_scores),
            )
            seasons_used = older_seasons + seasons_used

    if league in INTERNATIONAL_TOURNAMENT_LEAGUES and len(dates) < MIN_GAMES_FOR_MODEL:
        friendly_rows, friendly_seasons = _walk_back_season_games(
            FRIENDLIES_SUPPLEMENT_LEAGUE,
            team_abbr,
            espn_team_id,
            current_season,
            cutoff,
        )
        if friendly_seasons:
            dates, opponents, home_away, game_scores, period_scores = _merge_game_rows(
                (dates, opponents, home_away, game_scores, period_scores),
                friendly_rows,
            )
            seasons_used = seasons_used + [
                f"{FRIENDLIES_SUPPLEMENT_LEAGUE}:{season}" for season in friendly_seasons
            ]

    if not dates:
        return []

    if league in INTERNATIONAL_TOURNAMENT_LEAGUES:
        dates, opponents, home_away, game_scores, period_scores = _sort_game_rows(
            (dates, opponents, home_away, game_scores, period_scores)
        )

    year_key = str(current_season)
    return [
        {
            "year": year_key,
            "dates": dates,
            "other_team": opponents,
            "home_away": home_away,
            "game_scores": game_scores,
            "period_scores": period_scores,
            "seasons_used": seasons_used,
            "used_prior_season": len(seasons_used) > 1
            or (len(seasons_used) == 1 and seasons_used[0] == str(prior_season)),
        }
    ]


def load_live_team_data(
    league: str,
    team: list[str],
    espn_team_id: str,
    cutoff_date: str,
) -> list[dict[str, Any]]:
    return _build_team_entry(league, team[0], espn_team_id, cutoff_date)
