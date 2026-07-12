"""Build WNBA closing spreads/ML from ESPN's core odds API (median consensus)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

from web.nba_odds_espn import _consensus, _get_json, _valid_american
from web.season_games import _event_date_iso, _normalize_abbr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".build-cache" / "wnba-odds"
OUTPUT_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "wnba.csv"

# WNBA regular season is roughly May–October.
SKIP_MONTHS = frozenset({11, 12, 1, 2, 3, 4})

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={date}"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba/"
    "events/{event}/competitions/{comp}/odds"
)

CLOSING_FIELDS = (
    "date",
    "home_key",
    "away_key",
    "home_close_ml",
    "away_close_ml",
    "home_close_spread",
    "away_close_spread",
    "home_spread_odds",
    "away_spread_odds",
    "source",
)


def _iter_completed_events(start: date, end: date) -> Iterator[dict[str, Any]]:
    day = start
    while day <= end:
        datestr = day.strftime("%Y%m%d")
        try:
            payload = _get_json(SCOREBOARD_URL.format(date=datestr))
        except OSError:
            day += timedelta(days=1)
            continue
        for event in payload.get("events") or []:
            competition = (event.get("competitions") or [{}])[0]
            status = ((competition.get("status") or {}).get("type") or {})
            if not status.get("completed"):
                continue
            competitors = competition.get("competitors") or []
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            home_abbr = _normalize_abbr("wnba", (home.get("team") or {}).get("abbreviation", ""))
            away_abbr = _normalize_abbr("wnba", (away.get("team") or {}).get("abbreviation", ""))
            if not home_abbr or not away_abbr:
                continue
            try:
                int(home.get("score"))
                int(away.get("score"))
            except (TypeError, ValueError):
                continue
            tip_date = _event_date_iso(
                str(event.get("date") or competition.get("date") or "")
            ) or day.isoformat()
            yield {
                "date": tip_date,
                "event": str(event.get("id") or ""),
                "comp": str(competition.get("id") or ""),
                "home_key": home_abbr,
                "away_key": away_abbr,
            }
        day += timedelta(days=1)


def _cache_path(datestr: str) -> Path:
    return CACHE_DIR / f"{datestr}.json"


def _empty_odds_row(event: dict[str, Any]) -> dict[str, Any]:
    """Stub row when odds lookup fails — preserves the completed game."""
    return {
        "date": event["date"],
        "home_key": event["home_key"],
        "away_key": event["away_key"],
        "home_close_ml": None,
        "away_close_ml": None,
        "home_close_spread": None,
        "away_close_spread": None,
        "home_spread_odds": None,
        "away_spread_odds": None,
        "n_books": 0,
        "source": "espn-core",
    }


def collect_day_rows(day: date, *, use_cache: bool = True) -> list[dict[str, Any]]:
    datestr = day.strftime("%Y%m%d")
    cache_file = _cache_path(datestr)
    if use_cache and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            # Empty / partial-day / all-failed odds caches must not stick: one
            # filled game must not freeze siblings still at n_books=0.
            if cached and all(int(r.get("n_books") or 0) > 0 for r in cached):
                return cached
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    def _odds_row(event: dict[str, Any]) -> dict[str, Any]:
        row = _empty_odds_row(event)
        try:
            odds_payload = _get_json(
                ODDS_URL.format(event=event["event"], comp=event["comp"])
            )
        except OSError:
            return row
        items = [
            item
            for item in (odds_payload.get("items") or [])
            if "live" not in ((item.get("provider") or {}).get("name", "").lower())
        ]
        if not items:
            return row
        consensus = _consensus(items)
        home_ml = _valid_american(consensus.get("home_close_ml"))
        away_ml = _valid_american(consensus.get("away_close_ml"))
        home_spread = consensus.get("home_close_spread")
        if home_ml is None and away_ml is None and home_spread is None:
            return row
        row.update(
            {
                "home_close_ml": int(round(home_ml)) if home_ml is not None else None,
                "away_close_ml": int(round(away_ml)) if away_ml is not None else None,
                "home_close_spread": home_spread,
                "away_close_spread": consensus.get("away_close_spread"),
                # Missing juice stays None — do not invent -110 for training/CLV rows.
                "home_spread_odds": (
                    int(round(consensus["home_spread_odds"]))
                    if consensus.get("home_spread_odds") is not None
                    else None
                ),
                "away_spread_odds": (
                    int(round(consensus["away_spread_odds"]))
                    if consensus.get("away_spread_odds") is not None
                    else None
                ),
                # Prefer consensus count (parsed books only); never fall back to
                # raw provider shells which inflate thin ESPN payloads.
                "n_books": int(consensus.get("n_books") or 0),
            }
        )
        return row

    events = list(_iter_completed_events(day, day))
    rows: list[dict[str, Any]] = []
    if events:
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(_odds_row, events))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def fetch_wnba_odds_rows(
    start: date,
    end: date,
    *,
    use_cache: bool = True,
    day_workers: int = 6,
) -> list[dict[str, Any]]:
    days: list[date] = []
    day = start
    while day <= end:
        if day.month not in SKIP_MONTHS:
            days.append(day)
        day += timedelta(days=1)

    all_rows: list[dict[str, Any]] = []
    done = 0
    total = len(days)

    def _work(target: date) -> list[dict[str, Any]]:
        return collect_day_rows(target, use_cache=use_cache)

    with ThreadPoolExecutor(max_workers=day_workers) as pool:
        futures = {pool.submit(_work, target): target for target in days}
        for future in as_completed(futures):
            all_rows.extend(future.result())
            done += 1
            if done % 60 == 0:
                print(f"  {done}/{total} days, {len(all_rows)} games", flush=True)
    return all_rows
