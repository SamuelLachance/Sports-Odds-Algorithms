"""Probable-pitcher enrichment from the public MLB Stats API."""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from functools import lru_cache
from typing import Any

from web.mlb_stats_api import BASE_URL, ESPN_TO_MLB_TEAM_ID, USER_AGENT

# Lower ERA / WHIP = better pitcher.
LEAGUE_ERA_BASELINE = 4.25
LEAGUE_WHIP_BASELINE = 1.28
MAX_PITCHER_MARGIN = 0.85


def _fetch_json(url: str, timeout: int = 20) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _stat_from_block(stat: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(stat, dict) or not stat:
        return None
    try:
        innings = float(stat.get("inningsPitched") or stat.get("ip") or 0)
    except (TypeError, ValueError):
        return None
    if innings < 10:
        return None
    try:
        era = float(stat.get("era") or LEAGUE_ERA_BASELINE)
        whip = float(stat.get("whip") or LEAGUE_WHIP_BASELINE)
    except (TypeError, ValueError):
        return None
    return {"era": era, "whip": whip, "innings": innings}


def _pitching_snapshot(person: dict[str, Any] | None) -> dict[str, float] | None:
    """Extract season pitching from a probablePitcher person blob.

    Schedule hydrate returns season numbers on the block itself
    (``statsSingleSeason`` / ``stats`` dict, empty ``splits``). People-API
    hydrate uses ``splits[].stat``. Prefer season blocks over gameLog.
    """
    if not person:
        return None
    stats_blocks = person.get("stats") or []
    season_first = sorted(
        stats_blocks,
        key=lambda b: 0
        if str((b.get("type") or {}).get("displayName") or "")
        .lower()
        .find("season")
        >= 0
        else 1,
    )
    for block in season_first:
        if block.get("group", {}).get("displayName") != "pitching":
            continue
        # People API: splits[].stat
        for split in block.get("splits") or []:
            snap = _stat_from_block(split.get("stat") if isinstance(split, dict) else None)
            if snap:
                return snap
        # Schedule hydrate: season totals live directly on block["stats"].
        direct = block.get("stats")
        if isinstance(direct, dict) and "splits" not in direct:
            snap = _stat_from_block(direct)
            if snap:
                return snap
    return None


def _team_id_for_abbr(abbr: str) -> int | None:
    return ESPN_TO_MLB_TEAM_ID.get(abbr.lower())


@lru_cache(maxsize=64)
def fetch_probable_pitchers(game_date: str) -> dict[tuple[int, int, int], dict[str, dict[str, float]]]:
    """Map (home_team_id, away_team_id) -> {home: stats, away: stats} for a calendar date."""
    url = (
        f"{BASE_URL}/schedule?sportId=1&date={game_date}"
        "&hydrate=probablePitcher(note),stats(group=[pitching],type=[season])"
    )
    payload = _fetch_json(url)
    if not payload:
        return {}

    # Key includes gameNumber so doubleheaders do not last-win overwrite G1.
    lookup: dict[tuple[int, int, int], dict[str, dict[str, float]]] = {}
    for day in payload.get("dates") or []:
        for game in day.get("games") or []:
            teams = game.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            home_id = int((home.get("team") or {}).get("id") or 0)
            away_id = int((away.get("team") or {}).get("id") or 0)
            if not home_id or not away_id:
                continue
            try:
                game_number = int(game.get("gameNumber") or 1)
            except (TypeError, ValueError):
                game_number = 1
            home_pitcher = _pitching_snapshot(home.get("probablePitcher"))
            away_pitcher = _pitching_snapshot(away.get("probablePitcher"))
            if not home_pitcher and not away_pitcher:
                continue
            lookup[(home_id, away_id, game_number)] = {
                "home": home_pitcher or {"era": LEAGUE_ERA_BASELINE, "whip": LEAGUE_WHIP_BASELINE},
                "away": away_pitcher or {"era": LEAGUE_ERA_BASELINE, "whip": LEAGUE_WHIP_BASELINE},
            }
    return lookup


def _pitcher_skill(stats: dict[str, float]) -> float:
    era_skill = (LEAGUE_ERA_BASELINE - stats["era"]) / LEAGUE_ERA_BASELINE
    whip_skill = (LEAGUE_WHIP_BASELINE - stats["whip"]) / LEAGUE_WHIP_BASELINE
    return 0.65 * era_skill + 0.35 * whip_skill


def pitcher_matchup_margin(
    home_abbr: str,
    away_abbr: str,
    *,
    game_date: str | None = None,
    game_number: int | None = None,
) -> float | None:
    """Signed run margin from probable starter quality (home minus away)."""
    home_id = _team_id_for_abbr(home_abbr)
    away_id = _team_id_for_abbr(away_abbr)
    if not home_id or not away_id:
        return None
    when = game_date or date.today().isoformat()
    lookup = fetch_probable_pitchers(when)
    try:
        gn = int(game_number) if game_number is not None else 1
    except (TypeError, ValueError):
        gn = 1
    matchup = lookup.get((home_id, away_id, gn))
    if not matchup:
        return None
    diff = _pitcher_skill(matchup["home"]) - _pitcher_skill(matchup["away"])
    return max(min(diff * MAX_PITCHER_MARGIN, MAX_PITCHER_MARGIN), -MAX_PITCHER_MARGIN)
