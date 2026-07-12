"""Cross-sport availability / injury nudges from public ESPN endpoints."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from web.league_profiles import get_league_profile

USER_AGENT = "Sports-Odds-Algorithms/2.0"
MAX_AVAILABILITY_SHIFT_PP = 2.0
# ESPN team injury lists are $ref stubs; cap detail fetches per team.
MAX_INJURY_DETAIL_FETCHES = 40


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


def _injury_status_text(item: dict[str, Any]) -> str:
    """Normalize ESPN injury status from status / type / fantasy fields."""
    status = item.get("status")
    if status is not None and str(status).strip():
        return str(status).strip().lower()
    type_block = item.get("type")
    if isinstance(type_block, dict):
        for key in ("description", "name", "abbreviation"):
            val = type_block.get(key)
            if val is not None and str(val).strip():
                return str(val).strip().lower()
    elif type_block is not None and str(type_block).strip():
        return str(type_block).strip().lower()
    details = item.get("details")
    if isinstance(details, dict):
        fantasy = details.get("fantasyStatus")
        if isinstance(fantasy, dict):
            for key in ("description", "abbreviation"):
                val = fantasy.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip().lower()
    return ""


def _status_means_out(status: str) -> bool:
    """True for Out / injured / suspended — not negations or incidental substrings."""
    if not status:
        return False
    s = status.strip().lower()
    # Negated phrases must not count as OUT (substring "injured"/"out" would).
    if "not injured" in s or "not out" in s or "no longer injured" in s:
        return False
    # ESPN short codes / enum tails (word-boundary alone misses injury_status_out).
    if s in {"o", "out", "ir", "sus", "suspended", "injured"}:
        return True
    if s.endswith("_out") or "status_out" in s:
        return True
    # Word-boundary match: avoid "timeout", "about", "scout", etc.
    return bool(
        re.search(r"\bout\b", s)
        or re.search(r"\binjured\b", s)
        or re.search(r"\bsuspended\b", s)
    )


def _resolve_injury_item(item: Any) -> dict[str, Any] | None:
    """ESPN list endpoints return ``{$ref: ...}`` stubs without status inline.

    A nested ``type: {$ref: ...}`` dict is *not* enough — that still has no
    readable status, so we must fetch the detail URL when present.
    """
    if not isinstance(item, dict):
        return None
    # Usable inline status/type text → no fetch needed.
    if _injury_status_text(item):
        return item
    ref = item.get("$ref")
    if not isinstance(ref, str) or not ref.strip():
        return None
    payload = _fetch_json(ref.strip())
    return payload if isinstance(payload, dict) else None


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
    if not isinstance(items, list):
        return 0, 0
    try:
        total = int(payload.get("count") if payload.get("count") is not None else len(items))
    except (TypeError, ValueError):
        total = len(items)
    total = max(total, 0)
    out = 0
    fetched = 0
    for item in items:
        if fetched >= MAX_INJURY_DETAIL_FETCHES:
            break
        resolved = _resolve_injury_item(item)
        fetched += 1
        if not isinstance(resolved, dict):
            continue
        if _status_means_out(_injury_status_text(resolved)):
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
