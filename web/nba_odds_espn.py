"""Build a real NBA closing/opening odds table from ESPN's core odds API.

ESPN exposes per-event odds (opening + closing point spread, moneyline, and
total) across multiple sportsbooks. We take a median consensus across books so
the table reflects a robust market line rather than a single book. Team keys
match ``web/season_games.py`` (ESPN abbreviations) so the odds join cleanly to
the ML feature store.

Coverage notes (empirically):
- 2022-present: books expose explicit open/close objects.
- ~2016-2021: only a flat (effectively closing) line per book is available.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

# NBA has no regular games in July/August; skip to avoid wasted requests.
SKIP_MONTHS = frozenset({7, 8})
# Real NBA spreads are rarely beyond ~30; larger values are usually ML/juice dumps.
MAX_NBA_SPREAD = 40.0

from web.sbr_odds import _repair_same_sign_spreads
from web.season_games import _event_date_iso, _normalize_abbr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".build-cache" / "nba-odds"
OUTPUT_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "nba.csv"

USER_AGENT = "Sports-Odds-Algorithms/2.0"
SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{event}/competitions/{comp}/odds"
)

OUTPUT_FIELDS = (
    "date",
    "home_key",
    "away_key",
    "home_final",
    "away_final",
    "home_close_ml",
    "away_close_ml",
    "home_close_spread",
    "away_close_spread",
    "home_spread_odds",
    "away_spread_odds",
    "home_open_spread",
    "away_open_spread",
    "close_total",
    "open_total",
    "n_books",
    "source",
)


def _get_json(url: str, *, timeout: int = 30, retries: int = 3) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.4 * (attempt + 1))
    raise OSError(f"GET failed after {retries} tries: {url} ({last_exc})")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("+", "")
    if not text:
        return None
    # Case-insensitive PK/EVEN (HTML often emits ``Pk`` / ``Even``); match SBR.
    if text.upper() in {"EVEN", "PK"}:
        return 0.0
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _nested_american(node: Any, *keys: str) -> float | None:
    cur = node
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, dict):
        # Use the american field only; the decimal "value" would corrupt odds.
        cur = cur.get("american")
    return _to_float(cur)


def _valid_american(value: float | None) -> float | None:
    """American odds always have magnitude >= 100; reject decimal/garbage.

    ESPN EVEN sometimes arrives as 0 — treat as +100 (even money).
    Reject bools: ``False == 0`` would otherwise become +100.
    """
    if value is None or isinstance(value, bool):
        return None
    if value == 0:
        return 100.0
    if 100.0 <= abs(value) <= 100000.0:
        return value
    return None


def _valid_handicap_line(value: float | None, *, max_abs: float) -> float | None:
    """Keep real spread/run/puck lines; drop ML/juice dumps into handicap fields.

    American juice/ML magnitudes start at 100. Even when ``max_abs`` is raised for
    college blowouts (CFB ≤120), |line| ≥ 100 is never a real handicap.
    Reject bools: ``False == 0`` would otherwise become a pick'em line.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if abs(number) >= 100.0:
        return None
    if abs(number) > max_abs:
        return None
    return number


def _valid_total(value: float | None, *, max_total: float = 500.0) -> float | None:
    """Keep real over/under totals; drop EVEN/pk→0 and non-positive garbage.

    ``_to_float`` maps EVEN/pk to 0.0 for ML/spread cells — that must not become
    a closing total of 0.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number <= 0.0 or number > max_total:
        return None
    return number


def _provider_line(
    item: dict[str, Any],
    *,
    max_handicap_abs: float = MAX_NBA_SPREAD,
) -> dict[str, float | None]:
    """Extract one book's home-oriented lines, preferring close over flat values."""
    home = item.get("homeTeamOdds") or {}
    away = item.get("awayTeamOdds") or {}

    home_close_spread = _valid_handicap_line(
        _nested_american(home, "close", "pointSpread"),
        max_abs=max_handicap_abs,
    )
    away_close_spread = _valid_handicap_line(
        _nested_american(away, "close", "pointSpread"),
        max_abs=max_handicap_abs,
    )
    raw_spread = _to_float(item.get("spread"))
    # Juice/ML dumps in nested pointSpread are already dropped by
    # ``_valid_handicap_line`` (|x| ≥ 100). Flat ``spread`` goes through the
    # same helper below so CFB/CBB raised caps cannot reintroduce −110.
    if home_close_spread is None:
        # Flat ESPN ``spread`` is usually a signed home line. Only apply
        # favorite+magnitude when home is chalk and the flat value is still a
        # positive magnitude (NHL/MLB often send ``spread: 1.5`` + home favorite).
        # Never invert an already-signed line via a wrong away.favorite flag.
        if raw_spread is not None:
            if home.get("favorite") and not away.get("favorite") and raw_spread > 0:
                home_close_spread = -raw_spread
            else:
                home_close_spread = raw_spread
        home_close_spread = _valid_handicap_line(
            home_close_spread, max_abs=max_handicap_abs
        )
    if away_close_spread is None and home_close_spread is not None:
        away_close_spread = -home_close_spread

    home_ml = _nested_american(home, "close", "moneyLine")
    if home_ml is None:
        # Flat dict ``{american: ...}`` needs unwrap; bare ``_to_float(dict)`` drops ML.
        home_ml = _nested_american(home, "moneyLine")
    away_ml = _nested_american(away, "close", "moneyLine")
    if away_ml is None:
        away_ml = _nested_american(away, "moneyLine")
    home_ml = _valid_american(home_ml)
    away_ml = _valid_american(away_ml)

    home_close_spread, away_close_spread = _repair_same_sign_spreads(
        home_close_spread,
        away_close_spread,
        home_ml=home_ml,
        away_ml=away_ml,
    )

    home_open_spread = _valid_handicap_line(
        _nested_american(home, "open", "pointSpread"),
        max_abs=max_handicap_abs,
    )

    total = _valid_total(_to_float(item.get("overUnder")))
    open_total = _valid_total(_to_float(((item.get("open") or {}) or {}).get("total")))

    home_spread_odds = _valid_american(_nested_american(home, "close", "spread"))
    if home_spread_odds is None:
        # Flat dict ``{american: ...}`` needs unwrap (same as moneyLine).
        home_spread_odds = _valid_american(_nested_american(home, "spread"))
    away_spread_odds = _valid_american(_nested_american(away, "close", "spread"))
    if away_spread_odds is None:
        away_spread_odds = _valid_american(_nested_american(away, "spread"))

    return {
        "home_close_spread": home_close_spread,
        "away_close_spread": away_close_spread,
        "home_close_ml": home_ml,
        "away_close_ml": away_ml,
        "home_open_spread": home_open_spread,
        "close_total": total,
        "open_total": open_total,
        "home_spread_odds": home_spread_odds,
        "away_spread_odds": away_spread_odds,
    }


def _median(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(clean) if clean else None


def _median_handicap(values: list[float | None]) -> float | None:
    """Median handicap; fail closed when books disagree on favorite side.

    Opposite-sign lines of equal magnitude median to 0.0 and invent a fake
    pick'em — reject mixed-sign pools instead.
    """
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    nonzero = [v for v in clean if v != 0.0]
    if nonzero and len({v > 0 for v in nonzero}) > 1:
        return None
    return statistics.median(clean)


def _consensus(
    items: list[dict[str, Any]],
    *,
    max_handicap_abs: float = MAX_NBA_SPREAD,
) -> dict[str, Any]:
    lines = [_provider_line(item, max_handicap_abs=max_handicap_abs) for item in items]
    close_keys = (
        "home_close_spread",
        "away_close_spread",
        "home_close_ml",
        "away_close_ml",
        "close_total",
        "home_spread_odds",
        "away_spread_odds",
    )
    keys = close_keys + ("home_open_spread", "open_total")
    american_keys = frozenset(
        {"home_close_ml", "away_close_ml", "home_spread_odds", "away_spread_odds"}
    )
    handicap_keys = frozenset(
        {"home_close_spread", "away_close_spread", "home_open_spread"}
    )
    total_keys = frozenset({"close_total", "open_total"})
    # Medians may still use open fields; empty shells stay out of both pools.
    parsed = [line for line in lines if any(line.get(key) is not None for key in keys)]
    consensus: dict[str, Any] = {}
    for key in keys:
        raw_values = [line[key] for line in parsed]
        if key in handicap_keys:
            value = _median_handicap(raw_values)
        else:
            value = _median(raw_values)
            # Even-book medians can average into the invalid |x| < 100 band.
            if key in american_keys:
                value = _valid_american(value)
            elif key in total_keys:
                value = _valid_total(value)
        consensus[key] = value
    if consensus.get("away_open_spread") is None and consensus.get("home_open_spread") is not None:
        consensus["away_open_spread"] = -consensus["home_open_spread"]
    else:
        consensus["away_open_spread"] = None
    # Count only providers with at least one parsed *close* market — open-only
    # shells still feed open medians but must not inflate n_books.
    consensus["n_books"] = sum(
        1 for line in lines if any(line.get(key) is not None for key in close_keys)
    )
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
            home_abbr = _normalize_abbr("nba", (home.get("team") or {}).get("abbreviation", ""))
            away_abbr = _normalize_abbr("nba", (away.get("team") or {}).get("abbreviation", ""))
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
        day += timedelta(days=1)


def _cache_path(datestr: str) -> Path:
    return CACHE_DIR / f"{datestr}.json"


def _empty_odds_row(event: dict[str, Any]) -> dict[str, Any]:
    """Stub row when odds lookup fails — preserves the completed game."""
    return {
        "date": event["date"],
        "home_key": event["home_key"],
        "away_key": event["away_key"],
        "home_final": event["home_final"],
        "away_final": event["away_final"],
        "home_close_ml": None,
        "away_close_ml": None,
        "home_close_spread": None,
        "away_close_spread": None,
        "home_spread_odds": None,
        "away_spread_odds": None,
        "home_open_spread": None,
        "away_open_spread": None,
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
                # Missing juice stays None — do not invent -110 for training/CLV rows.
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

    events = list(_iter_completed_events(day, day))
    rows: list[dict[str, Any]] = []
    if events:
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(_odds_row, events))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def fetch_nba_odds_rows(
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
