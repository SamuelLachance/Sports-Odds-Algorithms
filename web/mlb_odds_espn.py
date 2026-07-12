"""Build MLB open/close moneylines + totals from ESPN's core odds API (median consensus)."""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

from web.nba_odds_espn import (
    _get_json,
    _nested_american,
    _to_float,
    _valid_american,
    _valid_handicap_line,
)
from web.sbr_odds import _repair_same_sign_spreads
from web.season_games import _event_date_iso, _normalize_abbr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".build-cache" / "mlb-odds"
OUTPUT_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "mlb.csv"

# Minimal off-season (no regular-season games).
SKIP_MONTHS = frozenset({1, 2})
# Real MLB run lines are ±1.5 (±2.5 rare). ESPN sometimes dumps moneylines here.
MAX_MLB_RUN_LINE = 7.0

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date}"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/"
    "events/{event}/competitions/{comp}/odds"
)

CLOSING_FIELDS = (
    "date",
    "home_key",
    "away_key",
    "home_close_ml",
    "away_close_ml",
    "home_open_ml",
    "away_open_ml",
    "home_close_spread",
    "away_close_spread",
    "home_spread_odds",
    "away_spread_odds",
    "close_total",
    "open_total",
    "n_books",
    "source",
)


def _median(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _provider_line_mlb(item: dict[str, Any]) -> dict[str, float | None]:
    """One book's home-oriented MLB lines: open/close ML, run line, totals."""
    home = item.get("homeTeamOdds") or {}
    away = item.get("awayTeamOdds") or {}

    home_close_ml = _valid_american(_nested_american(home, "close", "moneyLine"))
    if home_close_ml is None:
        home_close_ml = _valid_american(_to_float(home.get("moneyLine")))
    away_close_ml = _valid_american(_nested_american(away, "close", "moneyLine"))
    if away_close_ml is None:
        away_close_ml = _valid_american(_to_float(away.get("moneyLine")))

    home_open_ml = _valid_american(_nested_american(home, "open", "moneyLine"))
    away_open_ml = _valid_american(_nested_american(away, "open", "moneyLine"))

    home_close_spread = _valid_handicap_line(
        _nested_american(home, "close", "pointSpread"),
        max_abs=MAX_MLB_RUN_LINE,
    )
    if home_close_spread is None:
        raw_spread = _to_float(item.get("spread"))
        if raw_spread is not None:
            magnitude = abs(raw_spread)
            if home.get("favorite"):
                home_close_spread = -magnitude
            elif away.get("favorite"):
                home_close_spread = magnitude
            else:
                home_close_spread = raw_spread
        home_close_spread = _valid_handicap_line(
            home_close_spread, max_abs=MAX_MLB_RUN_LINE
        )
    away_close_spread = _valid_handicap_line(
        _nested_american(away, "close", "pointSpread"),
        max_abs=MAX_MLB_RUN_LINE,
    )
    if away_close_spread is None and home_close_spread is not None:
        away_close_spread = -home_close_spread

    home_close_spread, away_close_spread = _repair_same_sign_spreads(
        home_close_spread,
        away_close_spread,
        home_ml=home_close_ml,
        away_ml=away_close_ml,
    )

    close_total = _to_float(((item.get("close") or {}) or {}).get("total"))
    if close_total is None:
        close_total = _to_float(item.get("overUnder"))
    open_total = _to_float(((item.get("open") or {}) or {}).get("total"))

    return {
        "home_close_ml": home_close_ml,
        "away_close_ml": away_close_ml,
        "home_open_ml": home_open_ml,
        "away_open_ml": away_open_ml,
        "home_close_spread": home_close_spread,
        "away_close_spread": away_close_spread,
        "home_spread_odds": _valid_american(_nested_american(home, "close", "spread")),
        "away_spread_odds": _valid_american(_nested_american(away, "close", "spread")),
        "close_total": close_total,
        "open_total": open_total,
    }


def _consensus_mlb(items: list[dict[str, Any]]) -> dict[str, Any]:
    lines = [_provider_line_mlb(item) for item in items]
    keys = (
        "home_close_ml",
        "away_close_ml",
        "home_open_ml",
        "away_open_ml",
        "home_close_spread",
        "away_close_spread",
        "home_spread_odds",
        "away_spread_odds",
        "close_total",
        "open_total",
    )
    # Count only providers that yielded at least one parsed market number —
    # empty shells must not inflate n_books (fail-open on consensus quality).
    parsed = [line for line in lines if any(line.get(key) is not None for key in keys)]
    consensus: dict[str, Any] = {
        key: _median([line[key] for line in parsed]) for key in keys
    }
    consensus["n_books"] = len(parsed)
    return consensus


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
            home_abbr = _normalize_abbr("mlb", (home.get("team") or {}).get("abbreviation", ""))
            away_abbr = _normalize_abbr("mlb", (away.get("team") or {}).get("abbreviation", ""))
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
        "home_open_ml": None,
        "away_open_ml": None,
        "home_close_spread": None,
        "away_close_spread": None,
        "home_spread_odds": None,
        "away_spread_odds": None,
        "close_total": None,
        "open_total": None,
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
            if (
                cached
                and "home_open_ml" in cached[0]
                and all(int(r.get("n_books") or 0) > 0 for r in cached)
            ):
                return cached
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    def _odds_row(event: dict[str, Any]) -> dict[str, Any]:
        row = _empty_odds_row(event)

        def _int_or_none(value: Any) -> int | None:
            return int(round(value)) if value is not None else None

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
        consensus = _consensus_mlb(items)
        home_ml = _valid_american(consensus.get("home_close_ml"))
        away_ml = _valid_american(consensus.get("away_close_ml"))
        home_spread = consensus.get("home_close_spread")
        # Keep run-line-only rows (match NBA/WNBA) — do not require a ML.
        if home_ml is None and away_ml is None and home_spread is None:
            return row
        row.update(
            {
                "home_close_ml": _int_or_none(home_ml),
                "away_close_ml": _int_or_none(away_ml),
                "home_open_ml": _int_or_none(consensus.get("home_open_ml")),
                "away_open_ml": _int_or_none(consensus.get("away_open_ml")),
                "home_close_spread": home_spread,
                "away_close_spread": consensus.get("away_close_spread"),
                # Missing juice stays None — do not invent -110 for training/CLV rows.
                "home_spread_odds": _int_or_none(consensus.get("home_spread_odds")),
                "away_spread_odds": _int_or_none(consensus.get("away_spread_odds")),
                "close_total": consensus.get("close_total"),
                "open_total": consensus.get("open_total"),
                "n_books": consensus.get("n_books"),
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
    time.sleep(0.05)
    return rows


def fetch_mlb_odds_rows(
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
