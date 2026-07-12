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
    """Parse slate cutoffs as M-D-YYYY / MM-DD-YYYY, or ISO YYYY-MM-DD.

    NFL/CFB v2 live passes ISO ``day_iso`` into ``_load_league_game_map``; legacy
    callers still use project slate labels like ``9-15-2025``.
    """
    parts = str(cutoff_date).strip().split("-")
    if len(parts) != 3:
        raise ValueError(f"invalid cutoff date: {cutoff_date!r}")
    a, b, c = (int(p) for p in parts)
    if a >= 1900:  # YYYY-MM-DD
        year, month, day = a, b, c
    else:  # M-D-YYYY or MM-DD-YYYY
        month, day, year = a, b, c
    return datetime(year, month, day, tzinfo=timezone.utc)


def _event_before_cutoff(event: dict[str, Any], cutoff: datetime) -> bool:
    competition = (event.get("competitions") or [{}])[0]
    status = (competition.get("status") or {}).get("type") or {}
    if not status.get("completed"):
        return False

    # Compare Toronto calendar days (same as iso_to_project_date / slate labels).
    # A UTC-midnight cutoff wrongly drops late-ET games whose ISO stamp is the
    # next UTC day but still the prior Toronto day.
    from datetime import date as date_cls

    event_label = iso_to_project_date(event["date"])
    em, ed, ey = event_label.split("-")
    event_day = date_cls(int(ey), int(em), int(ed))
    cutoff_day = date_cls(cutoff.year, cutoff.month, cutoff.day)
    return event_day < cutoff_day


def _score_value(comp: dict[str, Any]) -> int | None:
    """Parse ESPN competitor score; None when missing/unparseable (never invent 0)."""
    score = comp.get("score")
    if isinstance(score, dict):
        raw = score.get("value")
    else:
        raw = score
    if raw is None or raw == "":
        return None
    try:
        # ESPN may send int, float, or numeric strings ("110" / "110.0").
        return int(float(raw))
    except (TypeError, ValueError):
        return None


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
        if team_score is None or opp_score is None:
            continue
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

    if not dates:
        return []

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
