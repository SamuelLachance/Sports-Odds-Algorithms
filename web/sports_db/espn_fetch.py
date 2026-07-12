"""ESPN public API fetchers for the sports database layer."""

from __future__ import annotations

import threading
from typing import Any

from web.espn_client import _fetch_json, current_season_year
from web.league_profiles import get_league_profile

_FETCH_CACHE: dict[str, dict[str, Any] | None] = {}
_CACHE_LOCK = threading.Lock()


def clear_fetch_cache() -> None:
    with _CACHE_LOCK:
        _FETCH_CACHE.clear()


def _cached_fetch(url: str, timeout: int = 15) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        if url in _FETCH_CACHE:
            return _FETCH_CACHE[url]
    try:
        payload = _fetch_json(url, timeout=timeout)
    except Exception:
        payload = None
    with _CACHE_LOCK:
        _FETCH_CACHE[url] = payload
    return payload


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
    payload = _cached_fetch(f"{_v2_base(league)}/standings?season={year}", timeout=15)
    if payload:
        return payload
    return _cached_fetch(f"{_site_base(league)}/standings", timeout=15)


def fetch_league_news(league: str, limit: int = 12) -> dict[str, Any] | None:
    payload = _cached_fetch(f"{_site_base(league)}/news?limit={limit}", timeout=15)
    return payload if payload and payload.get("articles") else None


def fetch_rankings(league: str) -> dict[str, Any] | None:
    payload = _cached_fetch(f"{_site_base(league)}/rankings", timeout=15)
    return payload if payload and payload.get("rankings") else None


def fetch_team_roster(league: str, espn_team_id: str) -> dict[str, Any] | None:
    return _cached_fetch(f"{_site_base(league)}/teams/{espn_team_id}/roster", timeout=15)


def fetch_team_statistics(league: str, espn_team_id: str, season_year: int | None = None) -> dict[str, Any] | None:
    year = season_year or season_year_for_league(league)
    return _cached_fetch(
        f"{_site_base(league)}/teams/{espn_team_id}/statistics?season={year}",
        timeout=15,
    )


def fetch_team_detail(league: str, espn_team_id: str) -> dict[str, Any] | None:
    return _cached_fetch(
        f"{_site_base(league)}/teams/{espn_team_id}?enable=roster,stats,projection",
        timeout=15,
    )


def _common_v3_base(league: str) -> str:
    sport_path = get_league_profile(league)["sport_path"]
    return f"https://site.api.espn.com/apis/common/v3/sports/{sport_path}"


def fetch_athlete_overview(league: str, athlete_id: str) -> dict[str, Any] | None:
    return _cached_fetch(f"{_common_v3_base(league)}/athletes/{athlete_id}/overview", timeout=15)


def fetch_athlete_stats(league: str, athlete_id: str) -> dict[str, Any] | None:
    return _cached_fetch(f"{_common_v3_base(league)}/athletes/{athlete_id}/stats", timeout=15)
