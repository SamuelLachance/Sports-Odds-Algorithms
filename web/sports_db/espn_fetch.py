"""ESPN public API fetchers for the sports database layer."""

from __future__ import annotations

from typing import Any

from web.espn_client import _fetch_json, current_season_year
from web.league_profiles import get_league_profile


def _site_base(league: str) -> str:
    sport_path = get_league_profile(league)["sport_path"]
    return f"https://site.api.espn.com/apis/site/v2/sports/{sport_path}"


def _v2_base(league: str) -> str:
    sport_path = get_league_profile(league)["sport_path"]
    return f"https://site.api.espn.com/apis/v2/sports/{sport_path}"


def season_year_for_league(league: str) -> int:
    from datetime import date

    return current_season_year(league, date.today())


def fetch_standings(league: str, season_year: int | None = None) -> dict[str, Any] | None:
    year = season_year or season_year_for_league(league)
    try:
        return _fetch_json(f"{_v2_base(league)}/standings?season={year}", timeout=25)
    except Exception:
        try:
            return _fetch_json(f"{_site_base(league)}/standings", timeout=25)
        except Exception:
            return None


def fetch_league_news(league: str, limit: int = 12) -> dict[str, Any] | None:
    try:
        payload = _fetch_json(f"{_site_base(league)}/news?limit={limit}", timeout=20)
        return payload if payload.get("articles") else None
    except Exception:
        return None


def fetch_rankings(league: str) -> dict[str, Any] | None:
    try:
        payload = _fetch_json(f"{_site_base(league)}/rankings", timeout=20)
        return payload if payload.get("rankings") else None
    except Exception:
        return None


def fetch_team_roster(league: str, espn_team_id: str) -> dict[str, Any] | None:
    try:
        return _fetch_json(f"{_site_base(league)}/teams/{espn_team_id}/roster", timeout=25)
    except Exception:
        return None


def fetch_team_statistics(league: str, espn_team_id: str, season_year: int | None = None) -> dict[str, Any] | None:
    year = season_year or season_year_for_league(league)
    try:
        return _fetch_json(
            f"{_site_base(league)}/teams/{espn_team_id}/statistics?season={year}",
            timeout=25,
        )
    except Exception:
        return None


def fetch_team_detail(league: str, espn_team_id: str) -> dict[str, Any] | None:
    try:
        return _fetch_json(
            f"{_site_base(league)}/teams/{espn_team_id}?enable=roster,stats,projection",
            timeout=25,
        )
    except Exception:
        return None


def _common_v3_base(league: str) -> str:
    sport_path = get_league_profile(league)["sport_path"]
    return f"https://site.api.espn.com/apis/common/v3/sports/{sport_path}"


def fetch_athlete_overview(league: str, athlete_id: str) -> dict[str, Any] | None:
    try:
        return _fetch_json(f"{_common_v3_base(league)}/athletes/{athlete_id}/overview", timeout=25)
    except Exception:
        return None


def fetch_athlete_stats(league: str, athlete_id: str) -> dict[str, Any] | None:
    try:
        return _fetch_json(f"{_common_v3_base(league)}/athletes/{athlete_id}/stats", timeout=25)
    except Exception:
        return None
