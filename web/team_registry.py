"""Dynamic team lists from ESPN for leagues without bundled CSV rosters."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from functools import lru_cache

from web.league_profiles import LEAGUE_PROFILES, SUPPORTED_LEAGUES, get_league_profile

PROJECT_ROOT_IMPORT = True


def _slugify(name: str) -> str:
    text = name.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text or "team"


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _teams_from_payload(payload: dict) -> list[dict[str, str]]:
    teams: list[dict[str, str]] = []

    direct = payload.get("teams") or []
    for entry in direct:
        team = entry.get("team") or entry
        abbr = (team.get("abbreviation") or team.get("shortDisplayName") or "").lower()
        slug = _slugify(team.get("displayName") or team.get("name") or abbr)
        if abbr:
            teams.append({"abbr": abbr, "slug": slug, "label": team.get("displayName") or slug})

    sports = payload.get("sports") or []
    for sport in sports:
        for league in sport.get("leagues") or []:
            for entry in league.get("teams") or []:
                team = entry.get("team") or entry
                abbr = (team.get("abbreviation") or team.get("shortDisplayName") or "").lower()
                slug = _slugify(team.get("displayName") or team.get("name") or abbr)
                if abbr:
                    teams.append(
                        {
                            "abbr": abbr,
                            "slug": slug,
                            "label": team.get("displayName") or slug,
                        }
                    )

    deduped: dict[str, dict[str, str]] = {}
    for team in teams:
        deduped[team["abbr"]] = team
    return list(deduped.values())


@lru_cache(maxsize=32)
def fetch_espn_teams(league: str) -> list[dict[str, str]]:
    league = league.lower()
    if league not in SUPPORTED_LEAGUES:
        return []

    profile = get_league_profile(league)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{profile['sport_path']}/teams?limit=500"
    )
    try:
        payload = _fetch_json(url)
    except urllib.error.URLError:
        return []

    teams = _teams_from_payload(payload)
    teams.sort(key=lambda item: item["label"])
    return teams


def load_team_registry(league: str) -> dict[str, list[str]]:
    """Return abbr/slug registry compatible with legacy universal_functions."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    teams_file = project_root / league / f"{league}_teams.txt"
    registry: dict[str, list[str]] = {}

    if teams_file.is_file():
        for line in teams_file.read_text(encoding="utf-8").splitlines():
            if "|" not in line:
                continue
            abbr, slug = [part.strip() for part in line.split("|", 1)]
            registry[abbr.lower()] = [abbr.lower(), slug]
            registry[slug] = [abbr.lower(), slug]

    for team in fetch_espn_teams(league):
        abbr = team["abbr"].lower()
        slug = team["slug"]
        if abbr not in registry:
            registry[abbr] = [abbr, slug]
        if slug not in registry:
            registry[slug] = [abbr, slug]

    return registry
