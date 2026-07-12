"""NHL v2 data layer: MoneyPuck game-by-game xG + NHL Stats API results/goalies.

All fetchers use stdlib urllib (CI has no requests). Offline history is cached
under .build-cache/nhl-history; the live path applies its own daily TTL cache.

Canonical team keys are NHL API abbreviations (TOR, LAK, NJD, ...). Helpers
convert to the ESPN-style keys used by the closing-odds archive (la, nj, ...).
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "nhl-history"

USER_AGENT = "Mozilla/5.0 (Sports-Odds-Algorithms NHL v2)"
STATS_BASE = "https://api.nhle.com/stats/rest/en"
WEB_BASE = "https://api-web.nhle.com/v1"
MONEYPUCK_TEAMS_URL = (
    "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
)

# MoneyPuck legacy spellings -> NHL API abbreviations.
MP_ABBR_FIXES: dict[str, str] = {
    "L.A": "LAK",
    "N.J": "NJD",
    "S.J": "SJS",
    "T.B": "TBL",
    "PHX": "ARI",
}

# NHL API abbreviation -> ESPN closing-odds key (lowercase). Others lowercase as-is.
NHL_TO_ESPN_KEY: dict[str, str] = {
    "LAK": "la",
    "NJD": "nj",
    "SJS": "sj",
    "TBL": "tb",
}

MP_SITUATIONS = ("all", "5on5", "5on4", "4on5")

MP_METRICS = (
    "xGoalsFor",
    "xGoalsAgainst",
    "scoreVenueAdjustedxGoalsFor",
    "scoreVenueAdjustedxGoalsAgainst",
    "flurryScoreVenueAdjustedxGoalsFor",
    "flurryScoreVenueAdjustedxGoalsAgainst",
    "goalsFor",
    "goalsAgainst",
    "shotsOnGoalFor",
    "shotsOnGoalAgainst",
    "highDangerxGoalsFor",
    "highDangerxGoalsAgainst",
    "scoreAdjustedShotsAttemptsFor",
    "scoreAdjustedShotsAttemptsAgainst",
    "shotAttemptsFor",
    "shotAttemptsAgainst",
    "blockedShotAttemptsFor",
    "blockedShotAttemptsAgainst",
    "reboundxGoalsFor",
    "reboundxGoalsAgainst",
    "hitsFor",
    "hitsAgainst",
    "penaltiesFor",
    "penaltiesAgainst",
    "dZoneGiveawaysFor",
    "iceTime",
)


def canon_team(abbr: str) -> str:
    abbr = str(abbr or "").strip().upper()
    return MP_ABBR_FIXES.get(abbr, abbr)


def espn_key(abbr: str) -> str:
    abbr = canon_team(abbr)
    return NHL_TO_ESPN_KEY.get(abbr, abbr.lower())


def get_json(url: str, *, retries: int = 4, timeout: int = 90) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = exc
        time.sleep(1.2 * (attempt + 1))
    raise OSError(f"Failed after {retries} tries: {url}") from last_error


def get_bytes(url: str, *, retries: int = 3, timeout: int = 300) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(2.0 * (attempt + 1))
    raise OSError(f"Failed after {retries} tries: {url}") from last_error


# ---------------------------------------------------------------------------
# MoneyPuck team game-by-game (xG per game, situational splits)
# ---------------------------------------------------------------------------

def download_moneypuck_teams(dest: Path | None = None) -> Path:
    """Download all_teams.csv (~130 MB) to the history cache."""
    dest = dest or (CACHE_ROOT / "mp_all_teams.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = get_bytes(MONEYPUCK_TEAMS_URL)
    dest.write_bytes(payload)
    return dest


def parse_moneypuck_teams(path: Path | None = None) -> dict[int, dict[str, dict[str, float]]]:
    """Map gameId -> team abbr -> flattened situational metrics.

    Keys look like ``all_xGoalsFor``, ``5on4_xGoalsFor`` etc. Only regular +
    playoff NHL games present in the CSV are returned.
    """
    path = path or (CACHE_ROOT / "mp_all_teams.csv")
    games: dict[int, dict[str, dict[str, float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            situation = row.get("situation") or ""
            if situation not in MP_SITUATIONS:
                continue
            try:
                game_id = int(row["gameId"])
            except (TypeError, ValueError):
                continue
            team = canon_team(row.get("team") or "")
            if not team:
                continue
            bucket = games.setdefault(game_id, {}).setdefault(team, {})
            for metric in MP_METRICS:
                raw = row.get(metric)
                if raw in (None, ""):
                    continue
                try:
                    bucket[f"{situation}_{metric}"] = float(raw)
                except ValueError:
                    continue
            if "home" not in bucket:
                bucket["home"] = 1.0 if (row.get("home_or_away") == "HOME") else 0.0
    return games


def parse_moneypuck_team_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Slim per-(game, team, situation) rows for cache serialization."""
    path = path or (CACHE_ROOT / "mp_all_teams.csv")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            situation = row.get("situation") or ""
            if situation not in MP_SITUATIONS:
                continue
            try:
                game_id = int(row["gameId"])
            except (TypeError, ValueError):
                continue
            slim: dict[str, Any] = {
                "gameId": game_id,
                "team": canon_team(row.get("team") or ""),
                "situation": situation,
                "home": 1 if row.get("home_or_away") == "HOME" else 0,
                "season": int(row.get("season") or 0),
                "gameDate": int(row.get("gameDate") or 0),
            }
            for metric in MP_METRICS:
                raw = row.get(metric)
                if raw not in (None, ""):
                    try:
                        slim[metric] = float(raw)
                    except ValueError:
                        pass
            rows.append(slim)
    return rows


# ---------------------------------------------------------------------------
# NHL Stats API: per-game team results + goalie logs
# ---------------------------------------------------------------------------

def _paged_stats(endpoint: str, season: int, game_type: int) -> list[dict[str, Any]]:
    season_id = f"{season}{season + 1}"
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        params = {
            "isAggregate": "false",
            "isGame": "true",
            "start": str(start),
            "limit": "100",
            "sort": json.dumps([
                {"property": "gameDate", "direction": "ASC"},
                {"property": "gameId", "direction": "ASC"},
            ]),
            "cayenneExp": f"seasonId={season_id} and gameTypeId={game_type}",
        }
        url = f"{STATS_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
        payload = get_json(url)
        chunk = payload.get("data") or []
        rows.extend(chunk)
        total = int(payload.get("total") or 0)
        start += len(chunk)
        if not chunk or start >= total:
            break
    return rows


def fetch_team_games(season: int) -> list[dict[str, Any]]:
    """Per-team per-game rows (regular + playoffs) with W/L, PP/PK, faceoffs."""
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for chunk in pool.map(
            lambda gt: _paged_stats("team/summary", season, gt), (2, 3)
        ):
            rows.extend(chunk)
    for row in rows:
        row["teamAbbrev"] = canon_team(_team_abbrev_from_row(row))
        row["opponentTeamAbbrev"] = canon_team(row.get("opponentTeamAbbrev") or "")
    rows.sort(key=lambda r: (str(r.get("gameDate")), int(r.get("gameId") or 0)))
    return rows


_TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Phoenix Coyotes": "ARI",
    "Atlanta Thrashers": "ATL", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Carolina Hurricanes": "CAR", "Columbus Blue Jackets": "CBJ",
    "Calgary Flames": "CGY", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
    "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL", "Montreal Canadiens": "MTL",
    "New Jersey Devils": "NJD", "Nashville Predators": "NSH",
    "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT",
    "Seattle Kraken": "SEA", "San Jose Sharks": "SJS", "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA", "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK", "Winnipeg Jets": "WPG", "Washington Capitals": "WSH",
}


def _team_abbrev_from_row(row: dict[str, Any]) -> str:
    abbr = row.get("teamAbbrev")
    if abbr:
        return str(abbr)
    return _TEAM_NAME_TO_ABBR.get(str(row.get("teamFullName") or ""), "")


def fetch_goalie_games(season: int) -> list[dict[str, Any]]:
    """Per-goalie per-game rows (regular + playoffs) with starts and save data."""
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for chunk in pool.map(
            lambda gt: _paged_stats("goalie/summary", season, gt), (2, 3)
        ):
            rows.extend(chunk)
    for row in rows:
        row["teamAbbrev"] = canon_team(row.get("teamAbbrev") or "")
    rows.sort(key=lambda r: (str(r.get("gameDate")), int(r.get("gameId") or 0)))
    return rows


# ---------------------------------------------------------------------------
# NHL Web API: daily schedule (for live inference)
# ---------------------------------------------------------------------------

def fetch_schedule_day(day_iso: str) -> list[dict[str, Any]]:
    """Games scheduled on `day_iso` with team abbrevs and game state."""
    payload = get_json(f"{WEB_BASE}/schedule/{day_iso}")
    games: list[dict[str, Any]] = []
    for week_day in payload.get("gameWeek") or []:
        if str(week_day.get("date")) != day_iso:
            continue
        for game in week_day.get("games") or []:
            games.append(
                {
                    "gameId": int(game.get("id") or 0),
                    "date": day_iso,
                    "game_type": int(game.get("gameType") or 2),
                    "start_utc": game.get("startTimeUTC"),
                    "state": game.get("gameState"),
                    "home": canon_team(((game.get("homeTeam") or {}).get("abbrev")) or ""),
                    "away": canon_team(((game.get("awayTeam") or {}).get("abbrev")) or ""),
                }
            )
    return games


def build_game_index(
    team_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Collapse per-team rows into one record per game (home perspective)."""
    games: dict[int, dict[str, Any]] = {}
    for row in team_rows:
        game_id = int(row.get("gameId") or 0)
        if not game_id:
            continue
        record = games.setdefault(
            game_id,
            {
                "gameId": game_id,
                "date": str(row.get("gameDate") or "")[:10],
                "game_type": int(str(game_id)[4:6] or 2),
            },
        )
        side = "home" if row.get("homeRoad") == "H" else "away"
        record[side] = row.get("teamAbbrev")
        record[f"{side}_goals"] = int(row.get("goalsFor") or 0)
        record[f"{side}_shots"] = round(float(row.get("shotsForPerGame") or 0.0))
        record[f"{side}_win"] = int(row.get("wins") or 0)
        record[f"{side}_ot_loss"] = int(row.get("otLosses") or 0)
        record[f"{side}_pp_pct"] = float(row.get("powerPlayPct") or 0.0)
        record[f"{side}_pk_pct"] = float(row.get("penaltyKillPct") or 0.0)
        # Preserve missing as None so the feature engine can skip (not treat as 0.0).
        raw_fo = row.get("faceoffWinPct")
        record[f"{side}_faceoff_pct"] = (
            float(raw_fo) if raw_fo is not None and raw_fo != "" else None
        )
        record[f"{side}_reg_win"] = int(row.get("winsInRegulation") or 0)
    complete = {
        gid: g
        for gid, g in games.items()
        if g.get("home") and g.get("away")
    }
    return complete


def serialize_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = sorted({key for row in rows for key in row})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
