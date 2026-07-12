"""Cross-sport availability / injury nudges from public ESPN endpoints."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from web.league_profiles import get_league_profile

USER_AGENT = "Sports-Odds-Algorithms/2.0"
MAX_AVAILABILITY_SHIFT_PP = 2.0


@dataclass
class AvailabilitySnapshot:
    home_injuries: int = 0
    away_injuries: int = 0
    home_out: int = 0
    away_out: int = 0
    sources: list[str] = field(default_factory=list)


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def _count_team_injuries(league: str, team_espn_id: str) -> tuple[int, int]:
    profile = get_league_profile(league)
    sport_path = profile["sport_path"]
    parts = sport_path.split("/")
    if len(parts) < 2:
        return 0, 0
    sport, league_slug = parts[0], parts[1]
    url = (
        f"https://sports.core.api.espn.com/v2/sports/{sport}/leagues/"
        f"{league_slug}/teams/{team_espn_id}/injuries"
    )
    payload = _fetch_json(url)
    if not isinstance(payload, dict):
        return 0, 0
    items = payload.get("items") or []
    total = len(items)
    out = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or item.get("type") or "").lower()
        if "out" in status or "injured" in status or "suspended" in status:
            out += 1
    return total, out


def fetch_availability_snapshot(
    league: str,
    *,
    home_espn_id: str | None,
    away_espn_id: str | None,
) -> AvailabilitySnapshot:
    snap = AvailabilitySnapshot()
    if home_espn_id:
        home_total, home_out = _count_team_injuries(league, home_espn_id)
        snap.home_injuries = home_total
        snap.home_out = home_out
        if home_total:
            snap.sources.append("espn_injuries_home")
    if away_espn_id:
        away_total, away_out = _count_team_injuries(league, away_espn_id)
        snap.away_injuries = away_total
        snap.away_out = away_out
        if away_total:
            snap.sources.append("espn_injuries_away")
    return snap


def availability_home_prob_shift(snapshot: AvailabilitySnapshot) -> float:
    """Signed percentage-point shift to home win probability."""
    home_burden = snapshot.home_out + 0.35 * max(snapshot.home_injuries - snapshot.home_out, 0)
    away_burden = snapshot.away_out + 0.35 * max(snapshot.away_injuries - snapshot.away_out, 0)
    diff = away_burden - home_burden
    shift = max(min(diff * 0.35, MAX_AVAILABILITY_SHIFT_PP), -MAX_AVAILABILITY_SHIFT_PP)
    return round(shift, 2)
