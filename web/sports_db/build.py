"""Build static sports database JSON for GitHub Pages."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from web.espn_client import clear_schedule_cache
from web.league_profiles import (
    LARGE_ROSTER_LEAGUES,
    LEAGUE_PROFILES,
    SCOREBOARD_ONLY_LEAGUES,
    SUPPORTED_LEAGUES,
    list_leagues_metadata,
)
from web.live_data import load_live_team_data, resolve_team
from web.sports_db.betting_context import league_betting_context, team_betting_context
from web.sports_db.espn_fetch import (
    clear_fetch_cache,
    fetch_athlete_overview,
    fetch_athlete_stats,
    fetch_league_news,
    fetch_rankings,
    fetch_standings,
    fetch_team_roster,
    fetch_team_statistics,
    season_year_for_league,
)
from web.sports_db.normalize import (
    SCHEMA_VERSION,
    build_player_roster_snapshot,
    build_player_snapshot,
    build_player_stats_snapshot,
    build_projection,
    build_trends,
    parse_news,
    parse_rankings,
    parse_roster,
    parse_standings,
    parse_team_statistics,
    parse_team_summary,
    roster_index_row,
)
from web.sports_db.player_ratings import enrich_team_roster_ratings
from web.sports_db.ratings import league_ratings_snapshot, team_rating_slice
from web.team_service import fetch_espn_team_ids

# Deep ESPN athlete fetches (overview + stats) per team in fast daily builds.
FAST_DEEP_SLATE_CAP = 30
FAST_DEEP_OTHER = 5
FAST_DEEP_LARGE_LEAGUE = 3
TEAM_BUILD_WORKERS = 4
PLAYER_FETCH_WORKERS = 4

_RATING_LEAGUES = {"nba", "nfl", "nhl", "mlb", "mls", "epl"}


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _league_profile_meta(league: str) -> dict[str, Any]:
    for row in list_leagues_metadata():
        if row.get("id") == league:
            return row
    profile = LEAGUE_PROFILES[league]
    return {
        "id": league,
        "name": profile["name"],
        "category": profile["category"],
        "description": "",
    }


def _team_name_lookup(standings: dict[str, Any] | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in (standings or {}).get("teams") or []:
        abbr = (row.get("abbr") or "").lower()
        name = row.get("name")
        if abbr and name:
            lookup[abbr] = name
    return lookup


def _recent_games_from_live(
    league: str,
    abbr: str,
    espn_id: str,
    cutoff_date: str,
    team_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
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

    names = team_names or {}
    rows: list[dict[str, Any]] = []
    dates = entry.get("dates") or []
    scores = entry.get("game_scores") or []
    home_away = entry.get("home_away") or []
    opponents = entry.get("other_team") or entry.get("opponents") or []
    for idx in range(min(len(dates), len(scores))):
        scored, allowed = scores[idx]
        loc = home_away[idx] if idx < len(home_away) else "home"
        opp_abbr = opponents[idx] if idx < len(opponents) else ""
        if scored > allowed:
            result = "W"
        elif scored < allowed:
            result = "L"
        else:
            result = "T"
        opp_name = names.get(opp_abbr.lower()) if opp_abbr else None
        if not opp_name and opp_abbr:
            opp_name = opp_abbr.upper()
        rows.append(
            {
                "date": dates[idx],
                "opponent": opp_name or "?",
                "opponent_abbr": opp_abbr.lower() if opp_abbr else None,
                "location": loc,
                "score": f"{scored}-{allowed}",
                "result": result,
            }
        )
    return list(reversed(rows[-10:]))


def _load_cached_team(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _recent_games_for_team(
    league: str,
    abbr: str,
    espn_id: str,
    cutoff_date: str,
    team_names: dict[str, str] | None,
    *,
    fast: bool,
    on_slate: bool,
    cached_team: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not should_fetch_recent_games(fast=fast, on_slate=on_slate):
        if cached_team:
            return cached_team.get("recent_games") or []
        return []
    return _recent_games_from_live(league, abbr, espn_id, cutoff_date, team_names)


def _standing_row_for_team(standings: dict[str, Any], abbr: str) -> dict[str, Any] | None:
    target = abbr.lower()
    for row in standings.get("teams") or []:
        if (row.get("abbr") or "").lower() == target:
            return row
    return None


def _injuries_from_roster(roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    injuries: list[dict[str, Any]] = []
    for player in roster:
        status = (player.get("status") or "").strip()
        if status and status.lower() not in {"active", "starter", "active roster"}:
            injuries.append(
                {
                    "id": player.get("id"),
                    "name": player.get("name"),
                    "position": player.get("position"),
                    "status": status,
                }
            )
    return injuries


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


def team_build_targets(
    league: str,
    team_ids: dict[str, str],
    slate_keys: set[str],
    standings: dict[str, Any],
    *,
    fast: bool,
) -> set[str]:
    """Teams that receive full team sheets (roster + card JSON)."""
    all_abbrs = set(team_ids.keys())
    if not fast:
        return all_abbrs
    if league in LARGE_ROSTER_LEAGUES or league in SCOREBOARD_ONLY_LEAGUES:
        standing_abbrs = {
            (row.get("abbr") or "").lower()
            for row in (standings.get("teams") or [])
        }
        return (slate_keys & all_abbrs) | (standing_abbrs & all_abbrs)
    return all_abbrs


def should_fetch_recent_games(*, fast: bool, on_slate: bool) -> bool:
    """Full-season schedule walks are expensive; limit them on fast daily builds."""
    return not fast or on_slate


def deep_player_targets(
    roster: list[dict[str, Any]],
    abbr: str,
    league: str,
    slate_keys: set[str],
    *,
    fast: bool,
    games_today: int = 1,
) -> set[str]:
    """Player ids that receive full ESPN overview/stats enrichment."""
    ids = [str(player["id"]) for player in roster if player.get("id")]
    if not ids:
        return set()
    if not fast:
        return set(ids)

    on_slate = abbr.lower() in slate_keys
    if games_today == 0 and not on_slate:
        return set()

    large = league in LARGE_ROSTER_LEAGUES
    if on_slate:
        cap = FAST_DEEP_SLATE_CAP if large else len(ids)
    elif large:
        cap = FAST_DEEP_LARGE_LEAGUE
    else:
        cap = FAST_DEEP_OTHER
    return set(ids[:cap])


def stats_only_player_targets(
    roster: list[dict[str, Any]],
    deep_ids: set[str],
    *,
    fast: bool,
) -> set[str]:
    """Roster players that receive stats-only enrichment (no overview/game log)."""
    if not fast:
        return set()
    return {
        str(player["id"])
        for player in roster
        if player.get("id") and str(player["id"]) not in deep_ids
    }


def _write_player_json(players_dir: Path, player_id: str, payload: dict[str, Any]) -> None:
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(players_dir / f"{player_id}.json", payload)


def build_league_snapshot(
    league: str,
    cutoff_date: str,
    slate: dict[str, Any] | None,
    *,
    team_keys: set[str] | None = None,
    include_ratings: bool = True,
) -> dict[str, Any]:
    profile = LEAGUE_PROFILES[league]
    profile_meta = _league_profile_meta(league)
    season_year = season_year_for_league(league)
    standings_raw = fetch_standings(league, season_year)
    standings = parse_standings(standings_raw)
    news = parse_news(fetch_league_news(league))
    rankings = parse_rankings(fetch_rankings(league))
    ratings = league_ratings_snapshot(league, cutoff_date) if include_ratings else {"source": [], "skipped": True}
    team_ids = fetch_espn_team_ids(league)
    betting = league_betting_context(slate, league)

    targets = team_keys or set()

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "league": {
            "id": league,
            "name": profile["name"],
            "category": profile["category"],
            "sport_path": profile["sport_path"],
        },
        "profile": profile_meta,
        "season_year": season_year,
        "cutoff_date": cutoff_date,
        "standings": standings,
        "news": news,
        "rankings": rankings,
        "ratings": ratings,
        "betting": betting,
        "team_ids": team_ids,
        "team_targets": sorted(targets),
    }


def _build_one_deep_player(
    league: str,
    abbr: str,
    player: dict[str, Any],
    players_dir: Path,
    cutoff_date: str,
) -> str:
    player_id = str(player["id"])
    overview = fetch_athlete_overview(league, player_id)
    stats_payload = fetch_athlete_stats(league, player_id)
    payload = build_player_snapshot(
        league=league,
        player_id=player_id,
        roster_row=player,
        overview=overview,
        stats_payload=stats_payload,
        team_abbr=abbr,
        cutoff_date=cutoff_date,
    )
    _write_player_json(players_dir, player_id, payload)
    return player_id


def _build_one_stats_player(
    league: str,
    abbr: str,
    player: dict[str, Any],
    players_dir: Path,
    cutoff_date: str,
) -> str:
    player_id = str(player["id"])
    stats_payload = fetch_athlete_stats(league, player_id)
    payload = build_player_stats_snapshot(
        league=league,
        player_id=player_id,
        roster_row=player,
        stats_payload=stats_payload,
        team_abbr=abbr,
        cutoff_date=cutoff_date,
    )
    _write_player_json(players_dir, player_id, payload)
    return player_id


def _build_one_roster_player(
    league: str,
    abbr: str,
    player: dict[str, Any],
    players_dir: Path,
    cutoff_date: str,
) -> str:
    player_id = str(player["id"])
    payload = build_player_roster_snapshot(
        league=league,
        player_id=player_id,
        roster_row=player,
        team_abbr=abbr,
        cutoff_date=cutoff_date,
    )
    _write_player_json(players_dir, player_id, payload)
    return player_id


def build_players_for_team(
    league: str,
    abbr: str,
    roster: list[dict[str, Any]],
    players_dir: Path,
    slate_keys: set[str],
    *,
    fast: bool,
    games_today: int = 1,
    cutoff_date: str,
) -> tuple[list[dict[str, Any]], int, int, dict[str, dict[str, Any]]]:
    """Write player JSON for every roster member (deep, stats, or roster tier)."""
    deep_ids = deep_player_targets(
        roster, abbr, league, slate_keys, fast=fast, games_today=games_today
    )
    stats_ids = stats_only_player_targets(roster, deep_ids, fast=fast)
    ratings_by_id: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    built_deep = 0

    def _record_rating(pid: str) -> None:
        player_path = players_dir / f"{pid}.json"
        if not player_path.is_file():
            return
        try:
            saved = json.loads(player_path.read_text(encoding="utf-8"))
            if saved.get("algo_rating") is not None:
                ratings_by_id[pid] = {
                    "algo_rating": saved["algo_rating"],
                    "rating_source": saved.get("rating_source"),
                    "rating_layer": saved.get("rating_layer"),
                }
        except (json.JSONDecodeError, OSError):
            pass

    roster_only = [
        player
        for player in roster
        if player.get("id") and str(player["id"]) not in deep_ids and str(player["id"]) not in stats_ids
    ]
    for player in roster_only:
        pid = str(player["id"])
        _build_one_roster_player(league, abbr, player, players_dir, cutoff_date)
        _record_rating(pid)

    stats_players = [player for player in roster if str(player.get("id") or "") in stats_ids]
    if stats_players:
        workers = min(PLAYER_FETCH_WORKERS, len(stats_players))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_build_one_stats_player, league, abbr, player, players_dir, cutoff_date): player
                for player in stats_players
            }
            for future in as_completed(futures):
                player = futures[future]
                future.result()
                _record_rating(str(player.get("id")))

    deep_players = [player for player in roster if str(player.get("id") or "") in deep_ids]
    if deep_players:
        workers = min(PLAYER_FETCH_WORKERS, len(deep_players))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_build_one_deep_player, league, abbr, player, players_dir, cutoff_date): player
                for player in deep_players
            }
            for future in as_completed(futures):
                player = futures[future]
                pid = str(player.get("id"))
                future.result()
                built_deep += 1
                _record_rating(pid)

    for player in roster:
        player_id = player.get("id")
        if not player_id:
            continue
        pid = str(player_id)
        row = roster_index_row(player)
        if pid in ratings_by_id:
            cached = ratings_by_id[pid]
            row["algo_rating"] = cached.get("algo_rating")
            if cached.get("rating_source") is not None:
                row["rating_source"] = cached["rating_source"]
            if cached.get("rating_layer") is not None:
                row["rating_layer"] = cached["rating_layer"]
        index_rows.append(row)

    return index_rows, built_deep, len(index_rows), ratings_by_id


def build_team_snapshot(
    league: str,
    abbr: str,
    espn_team_id: str,
    cutoff_date: str,
    standings: dict[str, Any],
    ratings: dict[str, Any],
    slate: dict[str, Any] | None,
    slate_keys: set[str],
    *,
    fast: bool,
    players_dir: Path,
    games_today: int = 1,
    cached_team: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, int]:
    season_year = season_year_for_league(league)
    roster_raw = fetch_team_roster(league, espn_team_id)
    stats_raw = fetch_team_statistics(league, espn_team_id, season_year)
    summary = parse_team_summary(roster_raw, stats_raw)
    roster = parse_roster(roster_raw)
    stats = parse_team_statistics(stats_raw)
    standing_row = _standing_row_for_team(standings, abbr)
    on_slate = abbr.lower() in slate_keys
    recent = _recent_games_for_team(
        league,
        abbr,
        espn_team_id,
        cutoff_date,
        _team_name_lookup(standings),
        fast=fast,
        on_slate=on_slate,
        cached_team=cached_team,
    )
    rating = team_rating_slice(ratings, abbr)
    power = (rating.get("power") or {}).get("power")
    player_index, players_built, roster_profiles, ratings_by_id = build_players_for_team(
        league,
        abbr,
        roster,
        players_dir,
        slate_keys,
        fast=fast,
        games_today=games_today,
        cutoff_date=cutoff_date,
    )
    enriched_roster, avg_player_rating = enrich_team_roster_ratings(
        league, roster, ratings_by_id, team_abbr=abbr, cutoff_date=cutoff_date
    )
    deep_ids = deep_player_targets(
        roster, abbr, league, slate_keys, fast=fast, games_today=games_today
    )
    rating["avg_player_rating"] = avg_player_rating
    if avg_player_rating is not None:
        rating["rating_diff"] = round(avg_player_rating - 50.0, 1)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "league": league,
        "cutoff_date": cutoff_date,
        "team": summary,
        "standing": standing_row,
        "stats": stats,
        "roster": enriched_roster,
        "injuries": _injuries_from_roster(roster),
        "trends": build_trends(standing_row, recent),
        "recent_games": recent,
        "ratings": rating,
        "projection": build_projection(standing_row, power),
        "betting": team_betting_context(slate, league, abbr),
        "players_built": sorted(deep_ids),
        "players_index": player_index,
        "roster_count": len(roster),
        "roster_profiles": roster_profiles,
    }
    return payload, players_built, roster_profiles


def build_sports_database(
    output_dir: Path,
    *,
    slate: dict[str, Any] | None = None,
    fast: bool = True,
    max_workers: int = 4,
    team_workers: int = TEAM_BUILD_WORKERS,
) -> dict[str, Any]:
    """Write docs/api/db snapshots for all supported leagues."""
    clear_fetch_cache()
    clear_schedule_cache()
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
    built_players = 0
    roster_profiles = 0

    def build_one_league(league: str) -> tuple[str, dict[str, Any], int, int, int]:
        slate_keys = _league_team_keys_from_slate(slate, league)
        season_year = season_year_for_league(league)
        standings_raw = fetch_standings(league, season_year)
        standings = parse_standings(standings_raw)
        team_ids = fetch_espn_team_ids(league)
        targets = team_build_targets(
            league,
            team_ids,
            slate_keys,
            standings,
            fast=fast,
        )
        snapshot = build_league_snapshot(
            league,
            cutoff_date,
            slate,
            team_keys=targets,
            include_ratings=bool(slate_keys) or league in _RATING_LEAGUES,
        )
        snapshot["standings"] = standings
        snapshot["team_ids"] = team_ids
        snapshot["team_targets"] = sorted(targets)

        league_dir = output_dir / league
        league_dir.mkdir(parents=True, exist_ok=True)
        _write_json(league_dir / "league.json", snapshot)

        teams_dir = league_dir / "teams"
        players_dir = league_dir / "players"
        games_today = (snapshot.get("betting") or {}).get("game_count", 0)
        name_by_abbr = {
            (row.get("abbr") or "").lower(): row.get("name")
            for row in snapshot["standings"].get("teams") or []
        }
        local_teams = 0
        local_players = 0

        def build_one_team(abbr: str) -> tuple[int, int, int]:
            espn_id = team_ids.get(abbr)
            if not espn_id:
                return 0, 0, 0
            cached_team = _load_cached_team(teams_dir / f"{abbr}.json")
            team_payload, player_count, roster_count = build_team_snapshot(
                league,
                abbr,
                espn_id,
                cutoff_date,
                standings,
                snapshot["ratings"],
                slate,
                slate_keys,
                fast=fast,
                players_dir=players_dir,
                games_today=games_today,
                cached_team=cached_team,
            )
            _write_json(teams_dir / f"{abbr}.json", team_payload)
            _write_json(
                teams_dir / abbr / "players.json",
                {"players": team_payload.get("players_index") or []},
            )
            return 1, player_count, roster_count

        target_list = sorted(targets)
        local_roster_profiles = 0
        if target_list:
            workers = min(team_workers, len(target_list))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(build_one_team, abbr): abbr for abbr in target_list}
                for future in as_completed(futures):
                    team_count, player_count, roster_count = future.result()
                    local_teams += team_count
                    local_players += player_count
                    local_roster_profiles += roster_count

        team_index = [
            {
                "abbr": abbr,
                "espn_id": team_ids.get(abbr),
                "name": name_by_abbr.get(abbr, abbr.upper()),
                "has_sheet": abbr in targets,
            }
            for abbr in sorted(team_ids.keys())
        ]
        _write_json(teams_dir / "index.json", {"teams": team_index})

        return league, {
            "id": league,
            "name": snapshot["league"]["name"],
            "category": snapshot["league"]["category"],
            "team_count": len(team_ids),
            "teams_built": local_teams,
            "players_built": local_players,
            "roster_profiles": local_roster_profiles,
            "games_today": (snapshot.get("betting") or {}).get("game_count", 0),
            "news_count": len(snapshot.get("news") or []),
            "standings_groups": len((snapshot.get("standings") or {}).get("groups") or []),
        }, local_teams, local_players, local_roster_profiles

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(build_one_league, league): league for league in SUPPORTED_LEAGUES}
        for future in as_completed(futures):
            league, meta, team_count, player_count, roster_count = future.result()
            manifest_leagues.append(meta)
            built_teams += team_count
            built_players += player_count
            roster_profiles += roster_count
            print(
                f"Sports DB: {league} ({team_count} teams, "
                f"{roster_count} roster profiles, {player_count} deep)"
            )

    manifest_leagues.sort(key=lambda row: row["name"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_date": cutoff_date,
        "fast_build": fast,
        "build_mode": "fast_tiered" if fast else "full",
        "league_count": len(manifest_leagues),
        "teams_built": built_teams,
        "players_built": built_players,
        "roster_profiles": roster_profiles,
        "leagues": manifest_leagues,
    }
    _write_json(output_dir / "manifest.json", manifest)
    clear_fetch_cache()
    clear_schedule_cache()
    return manifest
