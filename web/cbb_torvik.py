"""Fetch and cache BartTorvik team efficiency ratings for CBB."""

from __future__ import annotations

import csv
import io
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from web.season_games import _load_espn_team_ids

USER_AGENT = "Sports-Odds-Algorithms/2.0"
TORVIK_URL = "https://barttorvik.com/{year}_team_results.csv"

# In-memory cache: season_year -> abbr -> ratings
_TORVIK_CACHE: dict[int, dict[str, TorvikTeamRatings]] = {}
_ESPN_TEAM_CACHE: dict[str, list[tuple[str, str, str]]] = {}


@dataclass(frozen=True)
class TorvikTeamRatings:
    adj_o: float
    adj_d: float
    adj_t: float
    barthag: float
    torvik_name: str


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label or "").lower())


def _fetch_csv(url: str) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def _column_value(row: dict[str, str], *candidates: str) -> float | None:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in candidates:
        raw = lowered.get(name.lower())
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            continue
    return None


def _team_name_from_row(row: dict[str, str]) -> str:
    for key in ("team", "name", "school"):
        if key in row and row[key]:
            return str(row[key]).strip()
    for key, value in row.items():
        if key.lower() in {"team", "name", "school"} and value:
            return str(value).strip()
    return ""


def _espn_teams(league: str = "cbb") -> list[tuple[str, str, str]]:
    if league not in _ESPN_TEAM_CACHE:
        _ESPN_TEAM_CACHE[league] = _load_espn_team_ids(league)
    return _ESPN_TEAM_CACHE[league]


def _map_torvik_name_to_abbr(team_name: str, espn_teams: list[tuple[str, str, str]]) -> str | None:
    if not team_name:
        return None
    norm = _normalize_label(team_name)
    if not norm:
        return None

    best_abbr: str | None = None
    best_score = 0

    for _espn_id, abbr, display_name in espn_teams:
        candidates = [
            _normalize_label(display_name),
            _normalize_label(abbr),
            _normalize_label(display_name.split()[-1] if display_name else ""),
        ]
        for cand in candidates:
            if not cand:
                continue
            if norm == cand:
                return abbr.lower()
            if norm in cand or cand in norm:
                score = min(len(norm), len(cand))
                if score > best_score:
                    best_score = score
                    best_abbr = abbr.lower()
    return best_abbr


def _parse_torvik_csv(text: str, season_year: int) -> dict[str, TorvikTeamRatings]:
    espn_teams = _espn_teams("cbb")
    ratings: dict[str, TorvikTeamRatings] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        team_name = _team_name_from_row(row)
        adj_o = _column_value(row, "adjoe", "adj_o", "adjo", "adj off")
        adj_d = _column_value(row, "adjde", "adj_d", "adjd", "adj def")
        adj_t = _column_value(row, "adjt", "adj_t", "tempo", "adj tempo")
        barthag = _column_value(row, "barthag", "barthag", "bart")
        if adj_o is None or adj_d is None:
            continue
        if adj_t is None:
            adj_t = 68.0
        if barthag is None:
            barthag = 0.5
        abbr = _map_torvik_name_to_abbr(team_name, espn_teams)
        if not abbr:
            continue
        ratings[abbr] = TorvikTeamRatings(
            adj_o=adj_o,
            adj_d=adj_d,
            adj_t=adj_t,
            barthag=barthag,
            torvik_name=team_name,
        )
    if ratings:
        _TORVIK_CACHE[season_year] = ratings
    return ratings


def fetch_torvik_ratings(season_year: int) -> dict[str, TorvikTeamRatings]:
    """Return Torvik ratings keyed by ESPN abbreviation (lower case)."""
    if season_year in _TORVIK_CACHE:
        return _TORVIK_CACHE[season_year]

    url = TORVIK_URL.format(year=season_year)
    text = _fetch_csv(url)
    if not text:
        return {}

    try:
        return _parse_torvik_csv(text, season_year)
    except (csv.Error, ValueError):
        return {}


def clear_torvik_cache() -> None:
    """Clear in-memory Torvik and ESPN team caches (tests)."""
    _TORVIK_CACHE.clear()
    _ESPN_TEAM_CACHE.clear()
