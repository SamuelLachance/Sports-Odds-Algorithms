"""Lightweight client for the public MLB Stats API (statsapi.mlb.com)."""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from typing import Any

USER_AGENT = "Sports-Odds-Algorithms/2.0"
BASE_URL = "https://statsapi.mlb.com/api/v1"

# ESPN lower-case abbr -> MLB API teamId (2025 rosters).
ESPN_TO_MLB_TEAM_ID: dict[str, int] = {
    "ari": 109,
    "atl": 144,
    "bal": 110,
    "bos": 111,
    "chc": 112,
    "chw": 145,
    "cin": 113,
    "cle": 114,
    "col": 115,
    "det": 116,
    "hou": 117,
    "kc": 118,
    "laa": 108,
    "lad": 119,
    "mia": 146,
    "mil": 158,
    "min": 142,
    "nym": 121,
    "nyy": 147,
    "oak": 133,
    "phi": 143,
    "pit": 134,
    "sd": 135,
    "sea": 136,
    "sf": 137,
    "stl": 138,
    "tb": 139,
    "tex": 140,
    "tor": 141,
    "wsh": 120,
    "ath": 133,
}


def _fetch_json(url: str, timeout: int = 20) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except OSError:
        return None


# MLB Stats API fileCode / abbreviation → ESPN slate keys used elsewhere.
_MLB_STANDINGS_KEY_ALIASES: dict[str, str] = {
    "cws": "chw",  # White Sox (MLB fileCode) vs ESPN chw
    "ath": "oak",  # Athletics legacy key
    "was": "wsh",
}


def _standings_team_key(team: dict[str, Any]) -> str | None:
    abbr = (
        (team.get("fileCode") or team.get("abbreviation") or "")
        .lower()
        .replace(" ", "")
    )
    if not abbr:
        return None
    return _MLB_STANDINGS_KEY_ALIASES.get(abbr, abbr)


@lru_cache(maxsize=8)
def fetch_mlb_season_standings(season: int) -> dict[str, dict[str, float]]:
    """Team-level season stats keyed by lower-case ESPN-compatible abbr."""
    url = (
        f"{BASE_URL}/standings?leagueId=103,104&season={season}"
        f"&standingsTypes=regularSeason"
    )
    payload = _fetch_json(url)
    if not payload:
        return {}

    stats: dict[str, dict[str, float]] = {}
    for record in payload.get("records") or []:
        for team_rec in record.get("teamRecords") or []:
            team = team_rec.get("team") or {}
            abbr = _standings_team_key(team)
            if not abbr:
                continue
            wins = float(team_rec.get("wins") or 0)
            losses = float(team_rec.get("losses") or 0)
            games = max(wins + losses, 1.0)
            rs = float(team_rec.get("runsScored") or 0)
            ra = float(team_rec.get("runsAllowed") or 0)
            stats[abbr] = {
                "win_pct": wins / games,
                "runs_per_game": rs / games,
                "runs_allowed_per_game": ra / games,
                "run_diff_per_game": (rs - ra) / games,
                "games": games,
            }
    return stats


def mlb_api_matchup_edge(
    home_abbr: str,
    away_abbr: str,
    season: int,
) -> float | None:
    """Signed margin signal from MLB Stats API season run differential + win%."""
    home_key = home_abbr.lower()
    away_key = away_abbr.lower()
    home_id = ESPN_TO_MLB_TEAM_ID.get(home_key)
    away_id = ESPN_TO_MLB_TEAM_ID.get(away_key)
    standings = fetch_mlb_season_standings(season)
    if not standings:
        return None

    home_stats = standings.get(home_key)
    away_stats = standings.get(away_key)
    if not home_stats or not away_stats:
        return None

    run_skill = home_stats["run_diff_per_game"] - away_stats["run_diff_per_game"]
    win_skill = home_stats["win_pct"] - away_stats["win_pct"]
    return 2.8 * run_skill + 1.4 * win_skill


def mlb_team_season_snapshot(abbr: str, season: int) -> dict[str, float] | None:
    standings = fetch_mlb_season_standings(season)
    return standings.get(abbr.lower())
