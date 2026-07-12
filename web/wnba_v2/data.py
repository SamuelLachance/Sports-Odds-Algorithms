"""WNBA v2 data layer: league results (1997+), ESPN box scores (2003+), odds (2018+).

Sources (stdlib urllib, CI-safe):
  - data.wnba.com full-schedule JSON: complete final scores for every season
    since 1997 (ESPN's old scoreboards lack scores before 2002).
  - ESPN site.api scoreboard ranges: event ids, season type, neutral-site.
  - ESPN site.api summary per event: team box scores (four factors inputs).
  - ESPN sports.core.api odds per event: multi-book moneylines/spreads/totals;
    2024+ ESPN BET items also expose open/close.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "wnba-history"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
    "?dates={start}-{end}&limit=1000"
)
SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={event_id}"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba/"
    "events/{event_id}/competitions/{event_id}/odds"
)
LEAGUE_SCHEDULE_URL = (
    "https://data.wnba.com/data/10s/v2015/json/mobile_teams/wnba/{season}"
    "/league/10_full_schedule.json"
)

FIRST_SEASON = 1997
BOX_FIRST_SEASON = 2003
ODDS_FIRST_SEASON = 2018

# data.wnba.com tricodes -> canonical franchise key (relocations chained).
# The franchise key keeps Elo/state continuous across moves and rebrands.
FRANCHISE: dict[str, str] = {
    # active
    "atl": "atl", "chi": "chi", "conn": "con", "con": "con", "dal": "dal",
    "gsw": "gsv", "gsv": "gsv", "gs": "gsv", "ind": "ind", "lva": "lva",
    "lv": "lva", "las": "las", "la": "las", "min": "min", "nyl": "nyl",
    "ny": "nyl", "phx": "phx", "pho": "phx", "sea": "sea", "was": "was",
    "wsh": "was", "tor": "tor", "por": "por",
    # relocation chains
    "utah": "lva", "uta": "lva", "sas": "lva", "san": "lva",  # Starzz -> Silver Stars -> Aces
    "orl": "con",                                             # Miracle -> Sun
    "det": "dal", "tul": "dal",                               # Shock -> Tulsa -> Wings
    # defunct
    "hou": "hou", "sac": "sac", "cle": "cle", "cha": "cha", "mia": "mia",
    "prt": "prt",
}

# ESPN team ids -> canonical franchise key (ids stay stable across rebrands).
ESPN_ID_TO_FRANCHISE: dict[str, str] = {
    "20": "atl", "19": "chi", "18": "con", "3": "dal", "129689": "gsv",
    "5": "ind", "17": "lva", "6": "las", "8": "min", "9": "nyl",
    "11": "phx", "14": "sea", "16": "was", "131935": "tor", "133378": "por",
    "1": "cha", "2": "cle", "4": "det", "7": "mia", "10": "orl",
    "12": "prt", "13": "sac", "15": "hou", "21": "utah",
}


def canon_franchise(tricode_or_abbr: str) -> str:
    key = str(tricode_or_abbr or "").lower()
    return FRANCHISE.get(key, key)


def franchise_for_espn_id(team_id: str, abbr: str = "") -> str:
    mapped = ESPN_ID_TO_FRANCHISE.get(str(team_id))
    if mapped:
        return mapped
    return canon_franchise(abbr)


def fetch_league_results(season: int) -> list[dict[str, Any]]:
    """Final scores for a season from data.wnba.com (covers 1997+)."""
    data = get_json(LEAGUE_SCHEDULE_URL.format(season=season), timeout=45)
    rows: list[dict[str, Any]] = []
    for month in data.get("lscd", []):
        for game in (month.get("mscd") or {}).get("g", []):
            if game.get("stt") != "Final":
                continue
            home = game.get("h") or {}
            away = game.get("v") or {}
            home_score = _to_int(home.get("s"))
            away_score = _to_int(away.get("s"))
            if home_score is None or away_score is None:
                continue
            gid = str(game.get("gid") or "")
            # gid = {league:2}{type:1}{season:2}{game:5}; type 1=pre, 2=reg,
            # 3=all-star, 4=playoffs, 5=Commissioner's Cup final.
            type_digit = gid[2] if len(gid) >= 3 and gid[:2] == "10" else "2"
            if type_digit in ("1", "3"):
                continue
            season_type = 3 if type_digit == "4" else 2
            rows.append(
                {
                    "date": str(game.get("gdte") or ""),
                    "season": season,
                    "season_type": season_type,
                    "home": canon_franchise(home.get("ta") or ""),
                    "away": canon_franchise(away.get("ta") or ""),
                    "home_tricode": str(home.get("ta") or "").lower(),
                    "away_tricode": str(away.get("ta") or "").lower(),
                    "home_score": home_score,
                    "away_score": away_score,
                    "arena_city": game.get("ac"),
                    "arena_state": game.get("as"),
                }
            )
    rows.sort(key=lambda r: r["date"])
    return rows


def get_json(url: str, *, timeout: int = 30, retries: int = 3) -> Any:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.8 * (attempt + 1))
    raise OSError(f"GET failed after {retries} tries: {url}") from last


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_calendar_date(event_date: str) -> str:
    """America/Toronto calendar day for ESPN kickoffs (not UTC [:10])."""
    from web.season_games import _event_date_iso

    return _event_date_iso(event_date)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("+", "")
    # ESPN encodes even-money juice / pick'em spreads as EVEN or PK.
    if text.upper() in {"EVEN", "PK"}:
        return 0.0
    if text == "":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def fetch_season_events(season: int) -> list[dict[str, Any]]:
    """All WNBA events for a season (regular + playoffs; preseason excluded)."""
    data = get_json(
        SCOREBOARD_URL.format(start=f"{season}0415", end=f"{season}1115"),
        timeout=60,
    )
    events: list[dict[str, Any]] = []
    for event in data.get("events", []):
        season_type = _to_int((event.get("season") or {}).get("type")) or 0
        if season_type not in (2, 3):  # regular / post only
            continue
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        status = (((event.get("status") or {}).get("type")) or {}).get("name") or ""
        home: dict[str, Any] | None = None
        away: dict[str, Any] | None = None
        for side in comp.get("competitors") or []:
            team = side.get("team") or {}
            entry = {
                "team_id": str(team.get("id") or ""),
                "abbr": str(team.get("abbreviation") or "").lower(),
                "name": team.get("displayName"),
                "score": _to_int(side.get("score")),
            }
            if side.get("homeAway") == "home":
                home = entry
            elif side.get("homeAway") == "away":
                away = entry
        if not home or not away:
            continue
        events.append(
            {
                "event_id": str(event.get("id") or ""),
                "date": _event_calendar_date(str(event.get("date") or "")),
                "season": season,
                "season_type": season_type,
                "completed": status == "STATUS_FINAL",
                "neutral_site": bool(comp.get("neutralSite")),
                "home_id": home["team_id"],
                "away_id": away["team_id"],
                "home_abbr": home["abbr"],
                "away_abbr": away["abbr"],
                "home_name": home["name"],
                "away_name": away["name"],
                "home_score": home["score"],
                "away_score": away["score"],
            }
        )
    events.sort(key=lambda e: (e["date"], e["event_id"]))
    return events


def _parse_made_attempted(display: str) -> tuple[float, float] | None:
    parts = str(display).split("-")
    if len(parts) != 2:
        return None
    made = _to_float(parts[0])
    att = _to_float(parts[1])
    if made is None or att is None:
        return None
    return made, att


def _parse_minutes(value: Any) -> float | None:
    minutes = _to_float(str(value).strip())
    if minutes is None or minutes < 0:
        return None
    return minutes


def _player_minutes(side: dict[str, Any]) -> list[list[Any]]:
    """[[athlete_id, minutes], ...] for players who logged minutes (2006+)."""
    groups = side.get("statistics") or []
    if not groups:
        return []
    group = groups[0]
    names = [str(n).upper() for n in (group.get("names") or group.get("labels") or [])]
    try:
        min_idx = names.index("MIN")
    except ValueError:
        return []
    players: list[list[Any]] = []
    for entry in group.get("athletes") or []:
        if entry.get("didNotPlay"):
            continue
        stats = entry.get("stats") or []
        if len(stats) <= min_idx:
            continue
        minutes = _parse_minutes(stats[min_idx])
        athlete_id = str((entry.get("athlete") or {}).get("id") or "")
        if not athlete_id or minutes is None or minutes <= 0:
            continue
        players.append([athlete_id, minutes])
    return players


def fetch_box_score(event_id: str) -> dict[str, dict[str, Any]] | None:
    """Team box stats keyed by ESPN team id. None when unavailable.

    Each team dict carries four-factors inputs plus an optional "players" list
    of [athlete_id, minutes] pairs (available from ~2006) used by the
    availability/continuity features.
    """
    data = get_json(SUMMARY_URL.format(event_id=event_id), timeout=45)
    teams = (data.get("boxscore") or {}).get("teams") or []
    if len(teams) != 2:
        return None
    player_sides = (data.get("boxscore") or {}).get("players") or []
    minutes_by_team: dict[str, list[list[Any]]] = {}
    for side in player_sides:
        team_id = str((side.get("team") or {}).get("id") or "")
        if team_id:
            minutes_by_team[team_id] = _player_minutes(side)

    out: dict[str, dict[str, Any]] = {}
    for side in teams:
        team_id = str((side.get("team") or {}).get("id") or "")
        stats: dict[str, Any] = {}
        for item in side.get("statistics") or []:
            name = item.get("name") or ""
            display = item.get("displayValue") or ""
            if name == "fieldGoalsMade-fieldGoalsAttempted":
                pair = _parse_made_attempted(display)
                if pair:
                    stats["fgm"], stats["fga"] = pair
            elif name == "threePointFieldGoalsMade-threePointFieldGoalsAttempted":
                pair = _parse_made_attempted(display)
                if pair:
                    stats["tpm"], stats["tpa"] = pair
            elif name == "freeThrowsMade-freeThrowsAttempted":
                pair = _parse_made_attempted(display)
                if pair:
                    stats["ftm"], stats["fta"] = pair
            elif name == "offensiveRebounds":
                value = _to_float(display)
                if value is not None:
                    stats["orb"] = value
            elif name == "defensiveRebounds":
                value = _to_float(display)
                if value is not None:
                    stats["drb"] = value
            elif name == "totalTurnovers":
                value = _to_float(display)
                if value is not None:
                    stats["tov"] = value
            elif name == "turnovers" and "tov" not in stats:
                value = _to_float(display)
                if value is not None:
                    stats["tov"] = value
            elif name == "assists":
                value = _to_float(display)
                if value is not None:
                    stats["ast"] = value
        if "fga" not in stats:
            return None
        players = minutes_by_team.get(team_id)
        if players:
            stats["players"] = players
        out[team_id] = stats
    return out


def _american(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    # ESPN EVEN sometimes arrives as numeric 0 — treat as +100.
    if number == 0:
        return 100.0
    if abs(number) < 100:
        return None
    return number


def _signed_spread_from_details(details: str, home_abbr: str, away_abbr: str) -> float | None:
    """Parse 'LV -6.5' style details into a home-relative spread."""
    parts = str(details or "").strip().split()
    if len(parts) != 2:
        return None
    team, number = parts[0].lower(), _to_float(parts[1])
    if number is None:
        return None
    if team == home_abbr.lower():
        return number
    if team == away_abbr.lower():
        return -number
    return None


def _side_odds(item_side: dict[str, Any]) -> dict[str, Any]:
    from web.nba_odds_espn import MAX_NBA_SPREAD, _nested_american, _valid_handicap_line

    side = item_side or {}
    # Prefer ESPN core close.* shape (same as live multi-book); fall back to flat.
    ml = _american(_nested_american(side, "close", "moneyLine"))
    if ml is None:
        ml = _american(side.get("moneyLine"))
    spread_odds = _american(_nested_american(side, "close", "spread"))
    if spread_odds is None:
        spread_odds = _american(side.get("spreadOdds"))
    raw_ps = _nested_american(side, "close", "pointSpread")
    if raw_ps is None:
        current = side.get("current") or {}
        ps = (current.get("pointSpread") or {}).get("american")
        raw_ps = _to_float(str(ps).replace("+", "")) if ps not in (None, "") else None
    out: dict[str, Any] = {
        "ml": ml,
        "spread_odds": spread_odds,
        # Drop ML-sized values ESPN sometimes dumps into pointSpread.
        "point_spread": _valid_handicap_line(raw_ps, max_abs=MAX_NBA_SPREAD),
    }
    open_blob = side.get("open") or {}
    ml_open = _nested_american(side, "open", "moneyLine")
    if ml_open is None:
        ml_open_raw = (open_blob.get("moneyLine") or {}).get("american")
        ml_open = (
            _to_float(str(ml_open_raw).replace("+", ""))
            if ml_open_raw not in (None, "")
            else None
        )
    out["ml_open"] = _american(ml_open)
    raw_open = _nested_american(side, "open", "pointSpread")
    if raw_open is None:
        ps_open = (open_blob.get("pointSpread") or {}).get("american")
        raw_open = (
            _to_float(str(ps_open).replace("+", "")) if ps_open not in (None, "") else None
        )
    out["spread_open"] = _valid_handicap_line(raw_open, max_abs=MAX_NBA_SPREAD)
    return out


def fetch_event_odds(
    event_id: str,
    home_abbr: str,
    away_abbr: str,
) -> list[dict[str, Any]]:
    """Per-book odds rows for one event (moneylines, home spread, total)."""
    from web.nba_odds_espn import MAX_NBA_SPREAD, _valid_handicap_line

    try:
        data = get_json(ODDS_URL.format(event_id=event_id), timeout=45)
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for item in data.get("items") or []:
        provider = ((item.get("provider") or {}).get("name") or "").strip()
        if "live odds" in provider.lower():
            continue
        home_side = _side_odds(item.get("homeTeamOdds") or {})
        away_side = _side_odds(item.get("awayTeamOdds") or {})

        home_spread = home_side.get("point_spread")
        if home_spread is None:
            details_spread = _signed_spread_from_details(
                item.get("details") or "", home_abbr, away_abbr
            )
            if details_spread is not None:
                home_spread = details_spread
            else:
                raw = _to_float(item.get("spread"))
                if raw is not None:
                    home_odds = item.get("homeTeamOdds") or {}
                    away_odds = item.get("awayTeamOdds") or {}
                    if home_odds.get("favorite"):
                        home_spread = -abs(raw)
                    elif away_odds.get("favorite"):
                        home_spread = abs(raw)
                    else:
                        # ESPN often sends a signed home line; do not flip when
                        # favorite flags are missing/false.
                        home_spread = raw

        home_spread = _valid_handicap_line(home_spread, max_abs=MAX_NBA_SPREAD)
        rows.append(
            {
                "provider": provider,
                "home_ml": home_side.get("ml"),
                "away_ml": away_side.get("ml"),
                "home_spread": home_spread,
                "home_spread_odds": home_side.get("spread_odds"),
                "away_spread_odds": away_side.get("spread_odds"),
                "total": _to_float(item.get("overUnder")),
                "home_ml_open": home_side.get("ml_open"),
                "away_ml_open": away_side.get("ml_open"),
                "home_spread_open": home_side.get("spread_open"),
            }
        )
    return rows


def devig_two_way(price_a: float, price_b: float) -> tuple[float, float] | None:
    """Implied no-vig probabilities for a two-way American-odds market."""
    from web.basketball_v2_market import devig_home_prob

    home = devig_home_prob(price_a, price_b)
    if home is None:
        return None
    return home, 1.0 - home
