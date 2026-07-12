"""Build a real CFB closing/opening odds table from ESPN's core odds API.

Adapts web/nba_odds_espn.py to college football (FBS, groups=80). Walks daily
scoreboards from Aug 20 to Jan 20 for each requested season, pulls per-event
odds from the ESPN core API behind a global request throttle, takes a median
consensus across books, and writes data/supplemental/closing-odds/cfb.csv in
the same schema as nba.csv.

Completed games whose odds lookup fails are still written (empty odds fields,
n_books=0) so the walk-forward Elo backtest sees every FBS result. Per-day
responses are cached under .build-cache/cfb-odds/ when at least one row has
odds (n_books>0); empty or all-failed days are refetched on the next run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_odds_espn import OUTPUT_FIELDS, _consensus, _get_json  # noqa: E402
from web.season_games import _event_date_iso, _normalize_abbr  # noqa: E402

# CFB blowouts routinely exceed NBA's 40-pt handicap cap (archive max ~115).
MAX_CFB_SPREAD = 120.0

CACHE_DIR = PROJECT_ROOT / ".build-cache" / "cfb-odds"
OUTPUT_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "cfb.csv"

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/"
    "scoreboard?dates={date}&groups=80&limit=300"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
    "events/{event}/competitions/{comp}/odds"
)

# FBS season window: late August kickoffs through the CFP title game in January.
SEASON_START = (8, 20)
SEASON_END = (1, 20)

THROTTLE_SECONDS = 0.18
ODDS_WORKERS = 4

_throttle_lock = threading.Lock()
_next_request_at = 0.0


def _throttled_get(url: str) -> dict[str, Any]:
    """Global rate limiter: at most one ESPN request per THROTTLE_SECONDS."""
    global _next_request_at
    with _throttle_lock:
        wait = _next_request_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _next_request_at = time.monotonic() + THROTTLE_SECONDS
    return _get_json(url)


def _season_days(season: int) -> Iterator[date]:
    day = date(season, *SEASON_START)
    end = date(season + 1, *SEASON_END)
    while day <= end:
        yield day
        day += timedelta(days=1)


def _iter_completed_events(day: date) -> Iterator[dict[str, Any]]:
    datestr = day.strftime("%Y%m%d")
    try:
        payload = _throttled_get(SCOREBOARD_URL.format(date=datestr))
    except OSError:
        return
    for event in payload.get("events") or []:
        competition = (event.get("competitions") or [{}])[0]
        status = (competition.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
        competitors = competition.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_abbr = _normalize_abbr("cfb", (home.get("team") or {}).get("abbreviation", ""))
        away_abbr = _normalize_abbr("cfb", (away.get("team") or {}).get("abbreviation", ""))
        if not home_abbr or not away_abbr:
            continue
        try:
            home_score = int(home.get("score"))
            away_score = int(away.get("score"))
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
            "home_final": home_score,
            "away_final": away_score,
        }


def _odds_row(event: dict[str, Any]) -> dict[str, Any]:
    """One output row per completed game; odds fields empty when unavailable."""
    row: dict[str, Any] = {field: None for field in OUTPUT_FIELDS}
    row.update(
        {
            "date": event["date"],
            "home_key": event["home_key"],
            "away_key": event["away_key"],
            "home_final": event["home_final"],
            "away_final": event["away_final"],
            "n_books": 0,
            "source": "espn-core",
        }
    )
    try:
        payload = _throttled_get(ODDS_URL.format(event=event["event"], comp=event["comp"]))
    except OSError:
        return row
    items = [
        item
        for item in (payload.get("items") or [])
        if "live" not in ((item.get("provider") or {}).get("name", "").lower())
    ]
    if not items:
        return row
    consensus = _consensus(items, max_handicap_abs=MAX_CFB_SPREAD)
    # Keep away-only / totals-only closes — do not require a home ML or spread.
    if (
        consensus.get("home_close_spread") is None
        and consensus.get("away_close_spread") is None
        and consensus.get("home_close_ml") is None
        and consensus.get("away_close_ml") is None
        and consensus.get("close_total") is None
    ):
        return row
    row.update(
        {
            "home_close_ml": consensus["home_close_ml"],
            "away_close_ml": consensus["away_close_ml"],
            "home_close_spread": consensus["home_close_spread"],
            "away_close_spread": consensus["away_close_spread"],
            # Fail closed: do not invent -110 juice when books omit spread odds.
            "home_spread_odds": consensus["home_spread_odds"],
            "away_spread_odds": consensus["away_spread_odds"],
            "home_open_spread": consensus["home_open_spread"],
            "away_open_spread": consensus["away_open_spread"],
            "close_total": consensus["close_total"],
            "open_total": consensus["open_total"],
            "n_books": consensus["n_books"],
        }
    )
    return row


def collect_day_rows(day: date, *, use_cache: bool = True) -> list[dict[str, Any]]:
    cache_file = CACHE_DIR / f"{day.strftime('%Y%m%d')}.json"
    if use_cache and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            # Empty-odds / partial-day caches must not stick: one filled game
            # used to freeze siblings still at n_books=0. Sticky [] (scoreboard
            # outage / mid-day empty slate) must also refetch.
            if cached and all(int(r.get("n_books") or 0) > 0 for r in cached):
                return cached
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    events = list(_iter_completed_events(day))
    rows: list[dict[str, Any]] = []
    if events:
        with ThreadPoolExecutor(max_workers=ODDS_WORKERS) as pool:
            rows = [row for row in pool.map(_odds_row, events) if row is not None]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CFB (FBS) odds from ESPN core API")
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    seasons = [int(year) for year in args.seasons.split(",") if year.strip()]

    started = time.monotonic()
    rows_by_event_key: dict[tuple, dict[str, Any]] = {}
    for season in seasons:
        days = list(_season_days(season))
        season_rows = 0
        for index, day in enumerate(days, 1):
            for row in collect_day_rows(day, use_cache=not args.no_cache):
                rows_by_event_key[(row["date"], row["home_key"], row["away_key"])] = row
                season_rows += 1
            if index % 14 == 0 or index == len(days):
                elapsed = time.monotonic() - started
                print(
                    f"  season {season}: {index}/{len(days)} days, "
                    f"{season_rows} games this season, "
                    f"{len(rows_by_event_key)} total ({elapsed:.0f}s elapsed)",
                    flush=True,
                )

    rows = sorted(rows_by_event_key.values(), key=lambda r: (r["date"], r["home_key"]))
    with_odds = sum(1 for r in rows if r.get("home_close_spread") is not None)
    with_open = sum(1 for r in rows if r.get("home_open_spread") is not None)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in OUTPUT_FIELDS})

    print(
        f"wrote {OUTPUT_CSV} ({len(rows)} games, {with_odds} with closing spread, "
        f"{with_open} with opening spread)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
