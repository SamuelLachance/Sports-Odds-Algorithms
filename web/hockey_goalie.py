"""Goalie-aware xG multipliers for NHL and college hockey."""

from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime
from functools import lru_cache
from typing import Any

USER_AGENT = "Sports-Odds-Algorithms/2.0"
NHL_WEB_API = "https://api-web.nhle.com/v1"
NHL_STATS_API = "https://api.nhle.com/stats/rest/en/team/summary"

ESPN_TO_NHL_CODE: dict[str, str] = {
    "ana": "ANA",
    "ari": "ARI",
    "bos": "BOS",
    "buf": "BUF",
    "car": "CAR",
    "cbj": "CBJ",
    "cgy": "CGY",
    "chi": "CHI",
    "col": "COL",
    "dal": "DAL",
    "det": "DET",
    "edm": "EDM",
    "fla": "FLA",
    "la": "LAK",
    "min": "MIN",
    "mtl": "MTL",
    "nj": "NJD",
    "nsh": "NSH",
    "nyi": "NYI",
    "nyr": "NYR",
    "ott": "OTT",
    "phi": "PHI",
    "pit": "PIT",
    "sea": "SEA",
    "sj": "SJS",
    "stl": "STL",
    "tb": "TBL",
    "tor": "TOR",
    "uta": "UTA",
    "van": "VAN",
    "vgk": "VGK",
    "wpg": "WPG",
    "wsh": "WSH",
}

MAX_GOALIE_XG_SHIFT = 0.18
LEAGUE_AVG_SV_PCT = 0.905


def _fetch_json(url: str, timeout: int = 20) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _nhl_tri_code(abbr: str) -> str:
    return ESPN_TO_NHL_CODE.get(abbr.lower(), abbr.upper())


@lru_cache(maxsize=4)
def _fetch_goalie_save_rates() -> dict[str, float]:
    """Goalie save% keyed by playerId string."""
    rates: dict[str, float] = {}
    payload = _fetch_json(
        f"{NHL_STATS_API}?cayenneExp=gameTypeId=2&reportType=goalie"
        "&limit=100&sort=[{\"property\":\"savePct\",\"direction\":\"DESC\"}]"
        "&isAggregate=false&isGame=false&start=0"
    )
    if not isinstance(payload, dict):
        return rates
    for row in payload.get("data") or []:
        player_id = str(row.get("playerId") or "")
        sv = row.get("savePct")
        if player_id and sv is not None:
            try:
                rates[player_id] = float(sv)
            except (TypeError, ValueError):
                continue
    return rates


@lru_cache(maxsize=4)
def _fetch_team_save_pct() -> dict[str, float]:
    stats: dict[str, float] = {}
    payload = _fetch_json(f"{NHL_WEB_API}/club-stats/now")
    if isinstance(payload, dict):
        for row in payload.get("data") or []:
            tri = str(row.get("teamAbbrev") or "").lower()
            save_pct = row.get("savePct")
            if tri and save_pct is not None:
                try:
                    stats[tri] = float(save_pct)
                except (TypeError, ValueError):
                    continue
    if stats:
        return stats
    fallback = _fetch_json(
        f"{NHL_STATS_API}?cayenneExp=gameTypeId=2&limit=50"
        '&sort=[{"property":"savePct","direction":"DESC"}]'
    )
    if isinstance(fallback, dict):
        for row in fallback.get("data") or []:
            tri = str(row.get("teamAbbrev") or row.get("teamTriCode") or "").lower()
            save_pct = row.get("savePct")
            if tri and save_pct is not None:
                try:
                    stats[tri] = float(save_pct)
                except (TypeError, ValueError):
                    continue
    return stats


def _parse_game_date(raw: str | None) -> date | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if "T" in text:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


@lru_cache(maxsize=32)
def _fetch_scoreboard_now() -> list[dict[str, Any]]:
    payload = _fetch_json(f"{NHL_WEB_API}/score/now")
    if not isinstance(payload, dict):
        return []
    games = payload.get("games") or []
    return [g for g in games if isinstance(g, dict)]


def _find_matchup_starters(
    home_abbr: str,
    away_abbr: str,
    game_date: date | None,
) -> dict[str, Any]:
    """Resolve confirmed/probable starters from score/now or club schedule."""
    home_tri = _nhl_tri_code(home_abbr)
    away_tri = _nhl_tri_code(away_abbr)
    metadata: dict[str, Any] = {
        "home_starter": None,
        "away_starter": None,
        "home_confirmed": False,
        "away_confirmed": False,
        "source": None,
    }

    for game in _fetch_scoreboard_now():
        home_team = ((game.get("homeTeam") or {}).get("abbrev") or "").upper()
        away_team = ((game.get("awayTeam") or {}).get("abbrev") or "").upper()
        if home_team != home_tri or away_team != away_tri:
            continue
        start = _parse_game_date(game.get("startTimeUTC") or game.get("gameDate"))
        if game_date and start and start != game_date:
            continue
        metadata["source"] = "score/now"
        for side, tri, key in (
            ("home", home_tri, "home_starter"),
            ("away", away_tri, "away_starter"),
        ):
            team_block = game.get(f"{side}Team") or {}
            goalie = team_block.get("goalie") or team_block.get("probableGoalie") or {}
            player_id = str(goalie.get("playerId") or goalie.get("id") or "")
            name = goalie.get("name") or goalie.get("default") or goalie.get("lastName")
            confirmed = bool(goalie.get("confirmed") or goalie.get("starterConfirmed"))
            if player_id or name:
                metadata[key] = {"id": player_id or None, "name": name, "confirmed": confirmed}
                metadata[f"{key.split('_')[0]}_confirmed"] = confirmed
        if metadata["home_starter"] or metadata["away_starter"]:
            return metadata

    for tri, key in ((home_tri, "home_starter"), (away_tri, "away_starter")):
        projected = _project_starter_from_schedule(tri, game_date)
        if projected:
            metadata[key] = projected
            metadata["source"] = metadata.get("source") or "club-schedule"
    return metadata


def _project_starter_from_schedule(
    team_tri: str,
    game_date: date | None,
) -> dict[str, Any] | None:
    payload = _fetch_json(f"{NHL_WEB_API}/club-schedule-season/{team_tri}/now")
    if not isinstance(payload, dict):
        return None
    games = payload.get("games") or []
    last_starter: dict[str, Any] | None = None
    for game in reversed(games):
        if not isinstance(game, dict):
            continue
        gdate = _parse_game_date(game.get("gameDate") or game.get("startTimeUTC"))
        if game_date and gdate and gdate >= game_date:
            continue
        goalie = (game.get("homeTeam") or {}).get("goalie") if (
            (game.get("homeTeam") or {}).get("abbrev", "").upper() == team_tri
        ) else (game.get("awayTeam") or {}).get("goalie")
        if not goalie:
            continue
        player_id = str(goalie.get("playerId") or goalie.get("id") or "")
        if not player_id:
            continue
        last_starter = {
            "id": player_id,
            "name": goalie.get("name") or goalie.get("default"),
            "confirmed": False,
            "projected": True,
            "last_start": gdate.isoformat() if gdate else None,
        }
        if game_date and gdate:
            rest_days = (game_date - gdate).days
            last_starter["rest_days"] = rest_days
        break
    return last_starter


def _goalie_sv_pct(starter: dict[str, Any] | None, team_abbr: str) -> float | None:
    if not starter:
        team_sv = _fetch_team_save_pct().get(team_abbr.lower())
        return team_sv
    player_id = str(starter.get("id") or "")
    rates = _fetch_goalie_save_rates()
    if player_id and player_id in rates:
        return rates[player_id]
    return _fetch_team_save_pct().get(team_abbr.lower())


def _sv_to_xg_multiplier(goalie_sv: float, team_sv: float, shots_against_pg: float = 30.0) -> float:
    """GSAx proxy: better goalie suppresses opponent xG."""
    # Positive GSAx when the starter saves more than the team baseline.
    gsax = (goalie_sv - team_sv) * shots_against_pg
    shift = gsax * 0.012
    shift = max(-MAX_GOALIE_XG_SHIFT, min(MAX_GOALIE_XG_SHIFT, shift))
    return max(0.88, min(1.12, 1.0 - shift))


def _college_goalie_proxy(team_ga_pg: float, league_avg_ga: float) -> float:
    if league_avg_ga <= 0:
        return 1.0
    ratio = team_ga_pg / league_avg_ga
    return max(0.92, min(1.08, ratio))


def goalie_xg_multipliers(
    league: str,
    home_abbr: str,
    away_abbr: str,
    game_date: str | None = None,
    *,
    team_ga_rates: dict[str, float] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """
    Return (home_xg_mult, away_xg_mult, metadata).

    ``home_xg_mult`` scales home expected goals (away goalie / away GA);
    ``away_xg_mult`` scales away expected goals (home goalie / home GA).
    """
    league = league.lower()
    metadata: dict[str, Any] = {"league": league, "method": None}
    parsed_date = _parse_game_date(game_date)

    if league == "nhl":
        metadata["method"] = "nhl_goalie_api"
        starters = _find_matchup_starters(home_abbr, away_abbr, parsed_date)
        metadata.update(starters)
        team_sv_map = _fetch_team_save_pct()
        home_tri = _nhl_tri_code(home_abbr).lower()
        away_tri = _nhl_tri_code(away_abbr).lower()
        home_team_sv = team_sv_map.get(home_tri, LEAGUE_AVG_SV_PCT)
        away_team_sv = team_sv_map.get(away_tri, LEAGUE_AVG_SV_PCT)

        home_goalie_sv = _goalie_sv_pct(starters.get("home_starter"), home_tri) or home_team_sv
        away_goalie_sv = _goalie_sv_pct(starters.get("away_starter"), away_tri) or away_team_sv

        away_xg_mult = _sv_to_xg_multiplier(home_goalie_sv, home_team_sv)
        home_xg_mult = _sv_to_xg_multiplier(away_goalie_sv, away_team_sv)
        metadata["home_goalie_sv"] = round(home_goalie_sv, 4)
        metadata["away_goalie_sv"] = round(away_goalie_sv, 4)
        return home_xg_mult, away_xg_mult, metadata

    if league in ("ncaah", "ncaawh"):
        metadata["method"] = "college_ga_proxy"
        rates = team_ga_rates or {}
        home_ga = rates.get(home_abbr.lower())
        away_ga = rates.get(away_abbr.lower())
        if home_ga is None or away_ga is None:
            values = [v for v in rates.values() if v > 0]
            league_avg = sum(values) / len(values) if values else 3.0
            # Same opponent-defense convention as the both-present branch.
            home_mult = _college_goalie_proxy(away_ga or league_avg, league_avg)
            away_mult = _college_goalie_proxy(home_ga or league_avg, league_avg)
            metadata["league_avg_ga"] = round(league_avg, 3)
            return home_mult, away_mult, metadata
        league_avg = sum(rates.values()) / len(rates)
        return (
            _college_goalie_proxy(away_ga, league_avg),
            _college_goalie_proxy(home_ga, league_avg),
            metadata,
        )

    return 1.0, 1.0, metadata


def goalie_xg_adjustment(home_abbr: str, away_abbr: str) -> tuple[float, float] | None:
    """Backward-compatible wrapper used by legacy imports."""
    home_mult, away_mult, _meta = goalie_xg_multipliers("nhl", home_abbr, away_abbr)
    if home_mult == 1.0 and away_mult == 1.0:
        save_rates = _fetch_team_save_pct()
        if not save_rates:
            return None
        home_sv = save_rates.get(_nhl_tri_code(home_abbr).lower())
        away_sv = save_rates.get(_nhl_tri_code(away_abbr).lower())
        if home_sv is None or away_sv is None:
            return None
        home_shift = (home_sv - LEAGUE_AVG_SV_PCT) * 0.8
        away_shift = (away_sv - LEAGUE_AVG_SV_PCT) * 0.8
        return (
            max(0.92, min(1.08, 1.0 - away_shift)),
            max(0.92, min(1.08, 1.0 - home_shift)),
        )
    return home_mult, away_mult
