"""CBB v2 data layer: ESPN daily scoreboards (scores + conference flags).

College basketball scoreboard date-range queries 404 on ESPN's site API, so
history is assembled from per-day scoreboards cached under
``.build-cache/cbb-history/``. Team keys are stable ESPN team ids; abbreviations
are retained for live slate joins. No Torvik ratings are loaded here.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "cbb-history"
DAY_CACHE = CACHE_ROOT / "days"
CLOSING_ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "cbb.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard?dates={date}&groups=50&limit=300"
)

# Sparse abbr aliases seen on the public slate / ensemble joins.
ABBR_ALIASES: dict[str, str] = {
    "uconn": "conn",
    "connecticut": "conn",
}

FIRST_SEASON = 2018  # ending year of 2017-18
SEASON_START = (11, 1)  # Nov 1 of start year
SEASON_END = (4, 15)  # Apr 15 of ending year


def canon_abbr(abbr: str) -> str:
    key = str(abbr or "").lower().strip()
    return ABBR_ALIASES.get(key, key)


def force_postseason_neutral_site(game: dict[str, Any]) -> None:
    """March/April postseason games are usually neutral; ESPN often omits the flag."""
    date = str(game.get("date") or "")
    month = date[5:7] if len(date) >= 7 else ""
    if int(game.get("season_type") or 2) == 3 and month in {"03", "04"}:
        game["neutral_site"] = True


def team_key(espn_id: str, abbr: str = "") -> str:
    """Prefer stable ESPN id; fall back to canonical abbreviation."""
    eid = str(espn_id or "").strip()
    if eid:
        return eid
    return canon_abbr(abbr)


def get_json(url: str, *, timeout: int = 30, retries: int = 3) -> Any:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.6 * (attempt + 1))
    raise OSError(f"GET failed after {retries} tries: {url}") from last


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def cbb_season_for_date(target: date | str) -> int:
    """CBB season = ending calendar year (Nov Y-1 .. Apr Y -> season Y)."""
    if isinstance(target, str):
        target = date.fromisoformat(target[:10])
    if target.month >= 8:
        return target.year + 1
    return target.year


def _season_days(season: int) -> Iterator[date]:
    day = date(season - 1, *SEASON_START)
    end = date(season, *SEASON_END)
    while day <= end:
        yield day
        day += timedelta(days=1)


def _parse_event(event: dict[str, Any], *, fallback_date: str) -> dict[str, Any] | None:
    season_blob = event.get("season") or {}
    season_type = _to_int(season_blob.get("type")) or 0
    if season_type not in (2, 3):  # regular / postseason only
        return None
    comps = event.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]
    status = (((event.get("status") or {}).get("type")) or {}).get("name") or ""
    completed = status == "STATUS_FINAL" or bool(
        (((comp.get("status") or {}).get("type")) or {}).get("completed")
    )
    home: dict[str, Any] | None = None
    away: dict[str, Any] | None = None
    for side in comp.get("competitors") or []:
        team = side.get("team") or {}
        entry = {
            "team_id": str(team.get("id") or ""),
            "abbr": canon_abbr(str(team.get("abbreviation") or "")),
            "name": team.get("displayName"),
            "score": _to_int(side.get("score")),
            "conference_id": str(team.get("conferenceId") or "")
            or str((comp.get("groups") or {}).get("id") or ""),
        }
        if side.get("homeAway") == "home":
            home = entry
        elif side.get("homeAway") == "away":
            away = entry
    if not home or not away:
        return None
    event_id = str(event.get("id") or "")
    if not event_id:
        return None
    raw_date = str(event.get("date") or fallback_date)
    # Always store America/Toronto YYYY-MM-DD (never raw ISO / UTC[:10]).
    from web.season_games import _event_date_iso

    day_iso = _event_date_iso(raw_date) if "T" in raw_date else raw_date[:10]
    if not day_iso:
        day_iso = str(fallback_date)[:10]
    return {
        "event_id": event_id,
        "date": day_iso,
        "season": _to_int(season_blob.get("year")) or 0,
        "season_type": season_type,
        "completed": completed,
        "neutral_site": bool(comp.get("neutralSite")),
        "conference_game": bool(comp.get("conferenceCompetition")),
        "home_id": home["team_id"],
        "away_id": away["team_id"],
        "home_abbr": home["abbr"],
        "away_abbr": away["abbr"],
        "home_name": home["name"],
        "away_name": away["name"],
        "home_score": home["score"],
        "away_score": away["score"],
        "home_conference_id": home["conference_id"],
        "away_conference_id": away["conference_id"],
    }


def fetch_day_events(day: date, *, use_cache: bool = True) -> list[dict[str, Any]]:
    """Completed + scheduled events for one calendar day (D1 via groups=50)."""
    DAY_CACHE.mkdir(parents=True, exist_ok=True)
    datestr = day.strftime("%Y%m%d")
    cache_path = DAY_CACHE / f"{datestr}.json"
    if use_cache and cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(payload.get("events") or [])
        except (json.JSONDecodeError, OSError):
            pass
    try:
        data = get_json(SCOREBOARD_URL.format(date=datestr), timeout=45)
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for event in data.get("events") or []:
        parsed = _parse_event(event, fallback_date=day.isoformat())
        if parsed is not None:
            # Normalize UTC evening tipoffs to the requested local slate day
            # when ESPN's ISO date rolled past midnight UTC.
            if str(parsed.get("date") or "")[:10] != day.isoformat():
                parsed = dict(parsed)
                parsed["date"] = day.isoformat()
            events.append(parsed)
    cache_path.write_text(
        json.dumps({"events": events}, separators=(",", ":")),
        encoding="utf-8",
    )
    return events


def fetch_season_events(season: int, *, use_cache: bool = True) -> list[dict[str, Any]]:
    """All regular + postseason CBB events for an ending-year season."""
    season_dir = CACHE_ROOT / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    events_path = season_dir / "events.json"
    if use_cache and events_path.is_file():
        try:
            return list(json.loads(events_path.read_text(encoding="utf-8")).get("events") or [])
        except (json.JSONDecodeError, OSError):
            pass

    from concurrent.futures import ThreadPoolExecutor

    days = list(_season_days(season))
    by_id: dict[str, dict[str, Any]] = {}

    def _one(day: date) -> list[dict[str, Any]]:
        return fetch_day_events(day, use_cache=use_cache)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for events in pool.map(_one, days):
            for event in events:
                if event.get("season") and int(event["season"]) not in (0, season):
                    # Compare calendar days only — ISO suffixes break string bounds.
                    event_date = str(event.get("date") or "")[:10]
                    if not (f"{season - 1}-11-01" <= event_date <= f"{season}-04-15"):
                        continue
                    event = dict(event)
                    event["season"] = season
                elif not event.get("season"):
                    event = dict(event)
                    event["season"] = season
                by_id[str(event["event_id"])] = event

    events = list(by_id.values())
    events.sort(key=lambda e: (str(e.get("date") or ""), str(e.get("event_id") or "")))
    events_path.write_text(
        json.dumps({"events": events}, separators=(",", ":")),
        encoding="utf-8",
    )
    return events


def load_closing_odds_index() -> dict[tuple[str, str, str], dict[str, Any]]:
    """date + home_key + away_key -> odds row (keys are abbreviations)."""
    if not CLOSING_ODDS_CSV.is_file():
        return {}
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    with CLOSING_ODDS_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("date") or "")[:10],
                canon_abbr(str(row.get("home_key") or "")),
                canon_abbr(str(row.get("away_key") or "")),
            )
            if key[0] and key[1] and key[2]:
                index[key] = row
    return index


def build_abbr_id_map(events: list[dict[str, Any]]) -> dict[str, str]:
    """Most recent ESPN id observed for each abbreviation."""
    mapping: dict[str, str] = {}
    for event in events:
        for abbr_key, id_key in (
            ("home_abbr", "home_id"),
            ("away_abbr", "away_id"),
        ):
            abbr = canon_abbr(str(event.get(abbr_key) or ""))
            eid = str(event.get(id_key) or "")
            if abbr and eid:
                mapping[abbr] = eid
    return mapping
