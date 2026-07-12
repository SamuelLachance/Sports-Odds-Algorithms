"""Team profiles and season stats for all supported leagues."""

from __future__ import annotations

from datetime import date
from typing import Any

from web.espn_client import current_season_year
from web.league_profiles import LEAGUE_PROFILES, SUPPORTED_LEAGUES, get_league_profile
from web.live_data import load_live_team_data, resolve_team
from web.predict_service import get_teams


def _fetch_json(url: str) -> dict:
    import json
    import urllib.request

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_espn_team_ids(league: str) -> dict[str, str]:
    """Map team abbreviation to ESPN team id."""
    profile = get_league_profile(league)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{profile['sport_path']}/teams?limit=500"
    )
    try:
        payload = _fetch_json(url)
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    for entry in payload.get("teams") or []:
        team = entry.get("team") or entry
        abbr = (team.get("abbreviation") or "").lower()
        team_id = str(team.get("id") or "")
        if abbr and team_id:
            mapping[abbr] = team_id

    sports = payload.get("sports") or []
    for sport in sports:
        for lg in sport.get("leagues") or []:
            for entry in lg.get("teams") or []:
                team = entry.get("team") or entry
                abbr = (team.get("abbreviation") or "").lower()
                team_id = str(team.get("id") or "")
                if abbr and team_id:
                    mapping[abbr] = team_id
    return mapping


def build_league_sidebar_index(slate: dict | None = None) -> dict[str, Any]:
    """Lightweight league list for the sidebar (no per-league ESPN/CSV team fetches)."""
    game_counts: dict[str, int] = {}
    if slate:
        for game in slate.get("games") or []:
            league_id = game.get("league")
            if league_id:
                game_counts[league_id] = game_counts.get(league_id, 0) + 1

    leagues = []
    for league_id in SUPPORTED_LEAGUES:
        profile = LEAGUE_PROFILES[league_id]
        leagues.append(
            {
                "id": league_id,
                "name": profile["name"],
                "category": profile["category"],
                "team_count": game_counts.get(league_id, 0),
                "teams": [],
            }
        )
    return {
        "generated_at": date.today().isoformat(),
        "league_count": len(leagues),
        "leagues": leagues,
        "sidebar_only": True,
    }


def build_teams_index() -> dict[str, Any]:
    leagues = []
    for league in SUPPORTED_LEAGUES:
        teams = get_teams(league)
        leagues.append(
            {
                "id": league,
                "name": LEAGUE_PROFILES[league]["name"],
                "category": LEAGUE_PROFILES[league]["category"],
                "team_count": len(teams),
                "teams": teams,
            }
        )
    return {
        "generated_at": date.today().isoformat(),
        "league_count": len(leagues),
        "leagues": leagues,
    }


def get_team_profile(league: str, abbr: str) -> dict[str, Any]:
    league = league.lower()
    abbr = abbr.lower()
    teams = get_teams(league)
    team_row = next((t for t in teams if t["abbr"] == abbr), None)
    if not team_row:
        raise ValueError(f"Unknown team {abbr} in {league}")

    ids = fetch_espn_team_ids(league)
    espn_id = ids.get(abbr)
    if not espn_id:
        return {
            "league": league,
            "league_name": LEAGUE_PROFILES[league]["name"],
            **team_row,
            "season_stats": None,
            "note": "ESPN team id unavailable; roster listed only.",
        }

    resolved = resolve_team(league, abbr.upper(), team_row["label"])
    if not resolved:
        raise ValueError(f"Cannot resolve team {abbr}")

    today = date.today()
    cutoff = f"{today.month}-{today.day}-{today.year}"
    data = load_live_team_data(league, resolved, espn_id, cutoff)
    season_year = current_season_year(league, today)

    wins = losses = ties = 0
    recent: list[dict[str, Any]] = []
    if data:
        entry = data[-1]
        scores = entry.get("game_scores") or []
        opponents = entry.get("other_team") or []
        dates = entry.get("dates") or []
        for game in scores:
            scored, allowed = game[0], game[1]
            if scored > allowed:
                wins += 1
            elif scored < allowed:
                losses += 1
            else:
                ties += 1
        home_away = entry.get("home_away") or []
        team_labels = {t["abbr"]: t["label"] for t in teams}
        for i in range(max(0, len(scores) - 5), len(scores)):
            opp_abbr = opponents[i] if i < len(opponents) else ""
            opp_name = team_labels.get(opp_abbr.lower(), opp_abbr.upper() if opp_abbr else "")
            scored, allowed = scores[i][0], scores[i][1]
            if scored > allowed:
                result = "W"
            elif scored < allowed:
                result = "L"
            else:
                result = "T"
            recent.append(
                {
                    "date": dates[i] if i < len(dates) else "",
                    "opponent": opp_name,
                    "opponent_abbr": opp_abbr.lower() if opp_abbr else None,
                    "location": home_away[i] if i < len(home_away) else "",
                    "score": scores[i],
                    "result": result,
                }
            )

    seasons_used = data[-1].get("seasons_used") if data else []
    games_played = wins + losses + ties
    return {
        "league": league,
        "league_name": LEAGUE_PROFILES[league]["name"],
        "abbr": team_row["abbr"],
        "slug": team_row["slug"],
        "label": team_row["label"],
        "espn_id": espn_id,
        "season_year": season_year,
        "seasons_used": seasons_used,
        "season_stats": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "games_played": games_played,
            "win_pct": round(wins / games_played * 100, 1) if games_played else 0,
        },
        "recent_games": list(reversed(recent)),
        "cutoff_date": cutoff,
    }


def build_team_profiles_for_slate(slate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build profiles for teams appearing on today's slate (build-time)."""
    seen: set[tuple[str, str]] = set()
    profiles: dict[str, dict[str, Any]] = {}

    for game in slate.get("games") or []:
        league = game["league"]
        for side in ("away", "home"):
            abbr = game["matchup"][side]["abbr"].lower()
            key = (league, abbr)
            if key in seen:
                continue
            seen.add(key)
            try:
                profile = get_team_profile(league, abbr)
                profiles[f"{league}/{abbr}"] = profile
            except Exception:
                continue
    return profiles
