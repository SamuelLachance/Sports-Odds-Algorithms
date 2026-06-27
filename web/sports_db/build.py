"""Build static sports database JSON for GitHub Pages."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from web.espn_client import current_season_year
from web.league_profiles import LEAGUE_PROFILES, SUPPORTED_LEAGUES
from web.live_data import load_live_team_data, resolve_team
from web.sports_db.espn_fetch import (
    fetch_league_news,
    fetch_rankings,
    fetch_standings,
    fetch_team_roster,
    fetch_team_statistics,
    season_year_for_league,
)
from web.sports_db.normalize import (
    SCHEMA_VERSION,
    build_projection,
    build_trends,
    parse_news,
    parse_rankings,
    parse_roster,
    parse_standings,
    parse_team_statistics,
    parse_team_summary,
)
from web.sports_db.ratings import league_ratings_snapshot, team_rating_slice
from web.team_service import fetch_espn_team_ids


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _recent_games_from_live(league: str, abbr: str, espn_id: str, cutoff_date: str) -> list[dict[str, Any]]:
    resolved = resolve_team(league, abbr)
    if not resolved:
        return []
    try:
        entries = load_live_team_data(league, resolved, espn_id, cutoff_date)
        entry = entries[0] if entries else None
    except Exception:
        return []
    if not entry:
        return []

    rows: list[dict[str, Any]] = []
    dates = entry.get("dates") or []
    scores = entry.get("game_scores") or []
    home_away = entry.get("home_away") or []
    opponents = entry.get("opponents") or []
    for idx in range(min(len(dates), len(scores))):
        scored, allowed = scores[idx]
        loc = home_away[idx] if idx < len(home_away) else "home"
        opp = opponents[idx] if idx < len(opponents) else "?"
        if scored > allowed:
            result = "W"
        elif scored < allowed:
            result = "L"
        else:
            result = "T"
        rows.append(
            {
                "date": dates[idx],
                "opponent": opp,
                "location": loc,
                "score": f"{scored}-{allowed}",
                "result": result,
            }
        )
    return list(reversed(rows[-10:]))


def _standing_row_for_team(standings: dict[str, Any], abbr: str) -> dict[str, Any] | None:
    target = abbr.lower()
    for row in standings.get("teams") or []:
        if (row.get("abbr") or "").lower() == target:
            return row
    return None


def build_league_snapshot(
    league: str,
    cutoff_date: str,
    *,
    team_keys: set[str] | None = None,
    include_all_teams: bool = False,
    include_ratings: bool = True,
) -> dict[str, Any]:
    profile = LEAGUE_PROFILES[league]
    season_year = season_year_for_league(league)
    standings_raw = fetch_standings(league, season_year)
    standings = parse_standings(standings_raw)
    news = parse_news(fetch_league_news(league))
    rankings = parse_rankings(fetch_rankings(league))
    ratings = league_ratings_snapshot(league, cutoff_date) if include_ratings else {"source": [], "skipped": True}
    team_ids = fetch_espn_team_ids(league)

    teams_built: list[str] = []
    if include_all_teams:
        targets = set(team_ids.keys())
    elif team_keys:
        targets = {k.lower() for k in team_keys if k.lower() in team_ids}
    else:
        targets = set()

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "league": {
            "id": league,
            "name": profile["name"],
            "category": profile["category"],
            "sport_path": profile["sport_path"],
        },
        "season_year": season_year,
        "cutoff_date": cutoff_date,
        "standings": standings,
        "news": news,
        "rankings": rankings,
        "ratings": ratings,
        "team_ids": team_ids,
        "team_targets": sorted(targets),
    }


def build_team_snapshot(
    league: str,
    abbr: str,
    espn_team_id: str,
    cutoff_date: str,
    standings: dict[str, Any],
    ratings: dict[str, Any],
) -> dict[str, Any]:
    season_year = season_year_for_league(league)
    roster_raw = fetch_team_roster(league, espn_team_id)
    stats_raw = fetch_team_statistics(league, espn_team_id, season_year)
    summary = parse_team_summary(roster_raw, stats_raw)
    roster = parse_roster(roster_raw)
    stats = parse_team_statistics(stats_raw)
    standing_row = _standing_row_for_team(standings, abbr)
    recent = _recent_games_from_live(league, abbr, espn_team_id, cutoff_date)
    rating = team_rating_slice(ratings, abbr)
    power = (rating.get("power") or {}).get("power")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "league": league,
        "cutoff_date": cutoff_date,
        "team": summary,
        "standing": standing_row,
        "stats": stats,
        "roster": roster,
        "trends": build_trends(standing_row, recent),
        "recent_games": recent,
        "ratings": rating,
        "projection": build_projection(standing_row, power),
    }


def _league_team_keys_from_slate(slate: dict[str, Any] | None, league: str) -> set[str]:
    keys: set[str] = set()
    if not slate:
        return keys
    for game in slate.get("games") or []:
        if game.get("league") != league:
            continue
        matchup = game.get("matchup") or {}
        for side in ("home", "away"):
            abbr = (matchup.get(side) or {}).get("abbr")
            if abbr:
                keys.add(abbr.lower())
    return keys


def build_sports_database(
    output_dir: Path,
    *,
    slate: dict[str, Any] | None = None,
    fast: bool = True,
    max_workers: int = 6,
) -> dict[str, Any]:
    """Write docs/api/db snapshots for all supported leagues."""
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    cutoff_date = slate.get("date_label") if slate else today.isoformat()
    if slate and "-" in str(cutoff_date):
        parts = str(cutoff_date).split("-")
        if len(parts) == 3 and len(parts[0]) == 4:
            y, m, d = parts
            cutoff_date = f"{int(m)}-{int(d)}-{y}"

    manifest_leagues: list[dict[str, Any]] = []
    built_teams = 0

    def build_one_league(league: str) -> tuple[str, dict[str, Any], int]:
        slate_keys = _league_team_keys_from_slate(slate, league)
        snapshot = build_league_snapshot(
            league,
            cutoff_date,
            team_keys=slate_keys,
            include_all_teams=not fast,
            include_ratings=bool(slate_keys) or league in {"nba", "nfl", "nhl", "mlb", "mls", "epl"},
        )
        league_dir = output_dir / league
        league_dir.mkdir(parents=True, exist_ok=True)
        _write_json(league_dir / "league.json", snapshot)

        team_ids = snapshot.get("team_ids") or {}
        targets = set(snapshot.get("team_targets") or [])
        if fast and not targets:
            targets = set()

        teams_dir = league_dir / "teams"
        name_by_abbr = {
            (row.get("abbr") or "").lower(): row.get("name")
            for row in snapshot["standings"].get("teams") or []
        }
        local_built = 0
        for abbr in sorted(targets):
            espn_id = team_ids.get(abbr)
            if not espn_id:
                continue
            team_payload = build_team_snapshot(
                league,
                abbr,
                espn_id,
                cutoff_date,
                snapshot["standings"],
                snapshot["ratings"],
            )
            _write_json(teams_dir / f"{abbr}.json", team_payload)
            local_built += 1

        team_index = [
            {
                "abbr": abbr,
                "espn_id": team_ids.get(abbr),
                "name": name_by_abbr.get(abbr, abbr.upper()),
            }
            for abbr in sorted(team_ids.keys())
        ]
        _write_json(teams_dir / "index.json", {"teams": team_index})

        return league, {
            "id": league,
            "name": snapshot["league"]["name"],
            "category": snapshot["league"]["category"],
            "team_count": len(team_ids),
            "teams_built": local_built,
            "news_count": len(snapshot.get("news") or []),
            "standings_groups": len((snapshot.get("standings") or {}).get("groups") or []),
        }, local_built

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(build_one_league, league): league for league in SUPPORTED_LEAGUES}
        for future in as_completed(futures):
            league, meta, count = future.result()
            manifest_leagues.append(meta)
            built_teams += count
            print(f"Sports DB: {league} ({count} teams)")

    manifest_leagues.sort(key=lambda row: row["name"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_date": cutoff_date,
        "fast_build": fast,
        "league_count": len(manifest_leagues),
        "teams_built": built_teams,
        "leagues": manifest_leagues,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
