"""NBA v2 data layer: ESPN scores (1996+), box scores (2003+), odds join (2016+).

Sources (stdlib urllib, CI-safe):
  - ESPN site.api scoreboard ranges: event ids, final scores, season type,
    neutral-site flags. Season year = ending calendar year (2010-11 -> 2011).
  - ESPN site.api summary per event: team box scores (four-factor inputs).
  - data/supplemental/closing-odds/nba.csv: multi-book median open/close lines
    (built by scripts/fetch_nba_odds.py) joined by date + franchise keys.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "nba-history"
CLOSING_ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "nba.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    "?dates={start}-{end}&limit=1000"
)
SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{event_id}/competitions/{event_id}/odds"
)

FIRST_SEASON = 1997  # ending year of 1996-97
BOX_FIRST_SEASON = 2003
ODDS_FIRST_SEASON = 2016

# ESPN abbreviations / historical tricodes -> canonical franchise key.
# Franchise keys keep Elo continuous across relocations/rebrands.
FRANCHISE: dict[str, str] = {
    "atl": "atl", "bos": "bos", "bkn": "bkn", "bro": "bkn", "nj": "bkn", "njn": "bkn",
    "cha": "cha", "cho": "cha", "chh": "cha",
    "chi": "chi", "cle": "cle", "dal": "dal", "den": "den", "det": "det",
    "gs": "gs", "gsw": "gs", "hou": "hou", "ind": "ind",
    "lac": "lac", "lal": "lal",
    "mem": "mem", "van": "mem",  # Grizzlies
    "mia": "mia", "mil": "mil", "min": "min",
    "no": "no", "nop": "no", "noh": "no", "nok": "no",
    "ny": "ny", "nyk": "ny",
    "okc": "okc", "sea": "okc",  # SuperSonics -> Thunder
    "orl": "orl", "phi": "phi", "phx": "phx", "pho": "phx",
    "por": "por", "sac": "sac", "sa": "sa", "sas": "sa",
    "tor": "tor", "uta": "uta", "uth": "uta",
    "wsh": "wsh", "was": "wsh", "wshington": "wsh",
}

# ESPN team ids stay stable across rebrands (SEA id 25 became OKC).
ESPN_ID_TO_FRANCHISE: dict[str, str] = {
    "1": "atl", "2": "bos", "17": "bkn", "30": "cha", "4": "chi", "5": "cle",
    "6": "dal", "7": "den", "8": "det", "9": "gs", "10": "hou", "11": "ind",
    "12": "lac", "13": "lal", "29": "mem", "14": "mia", "15": "mil", "16": "min",
    "3": "no", "18": "ny", "25": "okc", "19": "orl", "20": "phi", "21": "phx",
    "22": "por", "23": "sac", "24": "sa", "28": "tor", "26": "uta", "27": "wsh",
}


def canon_franchise(tricode_or_abbr: str) -> str:
    key = str(tricode_or_abbr or "").lower().strip()
    return FRANCHISE.get(key, key)


def franchise_for_espn_id(team_id: str, abbr: str = "") -> str:
    mapped = ESPN_ID_TO_FRANCHISE.get(str(team_id))
    if mapped:
        return mapped
    return canon_franchise(abbr)


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


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _season_date_chunks(season: int) -> list[tuple[str, str]]:
    """Split a season into ESPN-safe date windows (API caps at ~1000 events)."""
    start_year = season - 1
    # Oct–Jan, Feb–Apr, May–Jul (playoffs / bubble)
    return [
        (f"{start_year}1001", f"{season}0131"),
        (f"{season}0201", f"{season}0430"),
        (f"{season}0501", f"{season}0731"),
        # 2019-20 bubble / late restart also spilled into Aug–Oct 2020
        (f"{season}0801", f"{season}1015"),
    ]


def fetch_season_events(season: int) -> list[dict[str, Any]]:
    """All NBA events for a season (regular + playoffs; preseason excluded)."""
    by_id: dict[str, dict[str, Any]] = {}
    for start, end in _season_date_chunks(season):
        try:
            data = get_json(SCOREBOARD_URL.format(start=start, end=end), timeout=90)
        except OSError:
            continue
        for event in data.get("events", []):
            season_blob = event.get("season") or {}
            season_type = _to_int(season_blob.get("type")) or 0
            season_year = _to_int(season_blob.get("year")) or 0
            if season_year and season_year != season:
                continue
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
            event_id = str(event.get("id") or "")
            if not event_id:
                continue
            by_id[event_id] = {
                "event_id": event_id,
                "date": str(event.get("date") or "")[:10],
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
    events = list(by_id.values())
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


def _player_minutes(data: Any) -> dict[str, list[list[Any]]]:
    """Per-team ``[[athlete_id, minutes], ...]`` rows from boxscore.players.

    Only athletes who logged positive minutes are kept; used downstream for
    rotation-continuity / star-availability features.
    """
    out: dict[str, list[list[Any]]] = {}
    for group in (data.get("boxscore") or {}).get("players") or []:
        team_id = str((group.get("team") or {}).get("id") or "")
        rows: list[list[Any]] = []
        for block in group.get("statistics") or []:
            names = [str(n).upper() for n in block.get("names") or []]
            try:
                min_idx = names.index("MIN")
            except ValueError:
                continue
            for athlete in block.get("athletes") or []:
                if athlete.get("didNotPlay"):
                    continue
                row = athlete.get("stats") or []
                if min_idx >= len(row):
                    continue
                raw = str(row[min_idx]).strip()
                minutes = _to_float(raw.split(":")[0]) if raw else None
                if minutes is None or minutes <= 0:
                    continue
                athlete_id = str((athlete.get("athlete") or {}).get("id") or "")
                if athlete_id:
                    rows.append([athlete_id, minutes])
        if team_id and rows:
            out[team_id] = rows
    return out


def fetch_box_score(event_id: str) -> dict[str, dict[str, Any]] | None:
    """Team box stats keyed by ESPN team id. None when unavailable.

    Each team dict may also carry ``players``: ``[[athlete_id, minutes], ...]``
    for athletes who logged minutes (availability features).
    """
    data = get_json(SUMMARY_URL.format(event_id=event_id), timeout=45)
    teams = (data.get("boxscore") or {}).get("teams") or []
    if len(teams) != 2:
        return None
    players_by_team = _player_minutes(data)
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
        players = players_by_team.get(team_id)
        if players:
            stats["players"] = players
        out[team_id] = stats
    return out


def _american(value: Any) -> float | None:
    number = _to_float(value)
    if number is None or abs(number) < 100:
        return None
    return number


def _signed_spread_from_details(details: str, home_abbr: str, away_abbr: str) -> float | None:
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
    out: dict[str, Any] = {
        "ml": _american((item_side or {}).get("moneyLine")),
        "spread_odds": _to_float((item_side or {}).get("spreadOdds")),
    }
    current = (item_side or {}).get("current") or {}
    ps = (current.get("pointSpread") or {}).get("american")
    out["point_spread"] = _to_float(str(ps).replace("+", "")) if ps not in (None, "") else None
    open_blob = (item_side or {}).get("open") or {}
    ml_open = (open_blob.get("moneyLine") or {}).get("american")
    out["ml_open"] = _american(str(ml_open).replace("+", "")) if ml_open not in (None, "") else None
    ps_open = (open_blob.get("pointSpread") or {}).get("american")
    out["spread_open"] = (
        _to_float(str(ps_open).replace("+", "")) if ps_open not in (None, "") else None
    )
    return out


def fetch_event_odds(
    event_id: str,
    home_abbr: str,
    away_abbr: str,
) -> list[dict[str, Any]]:
    """Per-book odds rows for one event (moneylines, home spread, total)."""
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
                    favored_home = bool((item.get("homeTeamOdds") or {}).get("favorite"))
                    home_spread = -abs(raw) if favored_home else abs(raw)
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


def load_closing_odds_index() -> dict[tuple[str, str, str], dict[str, Any]]:
    """(date, home_franchise, away_franchise) -> closing-odds CSV row."""
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not CLOSING_ODDS_CSV.is_file():
        return index
    with CLOSING_ODDS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            date = str(row.get("date") or "")[:10]
            home = canon_franchise(row.get("home_key") or "")
            away = canon_franchise(row.get("away_key") or "")
            if not (date and home and away):
                continue
            index[(date, home, away)] = {
                "home_ml": _to_float(row.get("home_close_ml")),
                "away_ml": _to_float(row.get("away_close_ml")),
                "home_spread": _to_float(row.get("home_close_spread")),
                "home_spread_odds": _to_float(row.get("home_spread_odds")),
                "away_spread_odds": _to_float(row.get("away_spread_odds")),
                "home_spread_open": _to_float(row.get("home_open_spread")),
                "total": _to_float(row.get("close_total")),
                "n_books": _to_int(row.get("n_books")) or 0,
            }
    return index


def devig_two_way(price_a: float, price_b: float) -> tuple[float, float] | None:
    """Implied no-vig probabilities for a two-way American-odds market."""

    def implied(american: float) -> float | None:
        if american >= 100:
            return 100.0 / (american + 100.0)
        if american <= -100:
            return -american / (-american + 100.0)
        return None

    pa = implied(price_a)
    pb = implied(price_b)
    if pa is None or pb is None:
        return None
    total = pa + pb
    if total <= 0:
        return None
    return pa / total, pb / total
