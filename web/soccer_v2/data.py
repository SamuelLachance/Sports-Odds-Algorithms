"""Soccer v2 data layer: football-data.co.uk full-history fetch + parse.

Covers the top-5 European leagues plus their second divisions (promoted-team
rating continuity). One CSV per division-season; columns vary by era:

  results:  Date, HomeTeam, AwayTeam, FTHG, FTAG (always)
  stats:    HS, AS, HST, AST, HC, AC, HY, AY, HR, AR (top-5 since 2000-01)
  odds:     B365H/D/A (since ~2002), PSH/D/A (Pinnacle pre-close, since
            2012-14), PSCH/D/A (Pinnacle closing, since 2015-16),
            AvgH/AvgCH etc. (market average open/close, since 2019-20)
"""

from __future__ import annotations

import csv
import io
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_DIR = PROJECT_ROOT / "data" / "soccer_history"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

USER_AGENT = "Sports-Odds-Algorithms/2.0 (research)"

# internal league id -> football-data division code (top flight)
LEAGUE_DIVISIONS: dict[str, str] = {
    "epl": "E0",
    "bundesliga": "D1",
    "laliga": "SP1",
    "seriea": "I1",
    "ligue1": "F1",
}

# second divisions feed the same country Elo pool (promotion continuity)
SECOND_DIVISIONS: dict[str, str] = {
    "E0": "E1",
    "D1": "D2",
    "SP1": "SP2",
    "I1": "I2",
    "F1": "F2",
}

DIVISION_TO_LEAGUE: dict[str, str] = {code: lg for lg, code in LEAGUE_DIVISIONS.items()}

# division code -> country pool key (shared Elo across tiers)
DIVISION_COUNTRY: dict[str, str] = {
    "E0": "E",
    "E1": "E",
    "D1": "D",
    "D2": "D",
    "SP1": "SP",
    "SP2": "SP",
    "I1": "I",
    "I2": "I",
    "F1": "F",
    "F2": "F",
}

# tier per division
DIVISION_TIER: dict[str, int] = {
    "E0": 1,
    "E1": 2,
    "D1": 1,
    "D2": 2,
    "SP1": 1,
    "SP2": 2,
    "I1": 1,
    "I2": 2,
    "F1": 1,
    "F2": 2,
}

FIRST_SEASON = 2000  # stats era for top divisions
FIRST_SEASON_TIER2 = 2004  # second divisions: odds coverage solid from here


def season_tag(season: int) -> str:
    """2015 -> '1516' (football-data folder tag for the 2015-16 season)."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def current_season(today: datetime | None = None) -> int:
    now = today or datetime.utcnow()
    return now.year if now.month >= 7 else now.year - 1


def csv_url(division: str, season: int) -> str:
    return f"{BASE_URL}/{season_tag(season)}/{division}.csv"


def fetch_csv_text(
    division: str,
    season: int,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    retries: int = 3,
) -> str | None:
    """Download (or reuse cached) division-season CSV text."""
    cache = (cache_dir or HISTORY_DIR) / f"{division}_{season}.csv"
    if cache.is_file() and not force:
        return cache.read_text(encoding="utf-8", errors="replace")

    url = csv_url(division, season)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                text = response.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    if not text.strip() or ("HomeTeam" not in text and "Home Team" not in text):
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else None


def _parse_date(raw: str) -> str | None:
    raw = (raw or "").strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _first_odds(record: dict[str, str], keys: tuple[str, ...]) -> float | None:
    """First valid decimal-odds value among candidate columns."""
    for key in keys:
        value = _to_float(record.get(key))
        if value is not None and value > 1.0:
            return value
    return None


def parse_season_csv(
    text: str,
    division: str,
    season: int,
) -> list[dict[str, Any]]:
    """Rows: results + stats + open/close decimal odds, chronologically sorted."""
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    for record in reader:
        home = (record.get("HomeTeam") or "").strip()
        away = (record.get("AwayTeam") or "").strip()
        if not home or not away:
            continue
        date_iso = _parse_date(record.get("Date") or "")
        if not date_iso:
            continue
        home_goals = _to_int(record.get("FTHG"))
        away_goals = _to_int(record.get("FTAG"))
        if home_goals is None or away_goals is None:
            continue

        row: dict[str, Any] = {
            "division": division,
            "season": season,
            "date": date_iso,
            "home": home,
            "away": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_shots": _to_int(record.get("HS")),
            "away_shots": _to_int(record.get("AS")),
            "home_sot": _to_int(record.get("HST")),
            "away_sot": _to_int(record.get("AST")),
            "home_corners": _to_int(record.get("HC")),
            "away_corners": _to_int(record.get("AC")),
            "home_reds": _to_int(record.get("HR")),
            "away_reds": _to_int(record.get("AR")),
            # earliest available price (Pinnacle pre-close, then B365, then avg)
            "open_home": _first_odds(
                record, ("PSH", "B365H", "AvgH", "BbAvH", "WHH", "IWH")
            ),
            "open_draw": _first_odds(
                record, ("PSD", "B365D", "AvgD", "BbAvD", "WHD", "IWD")
            ),
            "open_away": _first_odds(
                record, ("PSA", "B365A", "AvgA", "BbAvA", "WHA", "IWA")
            ),
            # closing price (Pinnacle close, then market-average close, B365 close)
            "close_home": _first_odds(record, ("PSCH", "AvgCH", "B365CH")),
            "close_draw": _first_odds(record, ("PSCD", "AvgCD", "B365CD")),
            "close_away": _first_odds(record, ("PSCA", "AvgCA", "B365CA")),
            # soft-book reference prices (Bet365 pre-close and close)
            "b365_home": _first_odds(record, ("B365H",)),
            "b365_draw": _first_odds(record, ("B365D",)),
            "b365_away": _first_odds(record, ("B365A",)),
            "b365c_home": _first_odds(record, ("B365CH",)),
            "b365c_draw": _first_odds(record, ("B365CD",)),
            "b365c_away": _first_odds(record, ("B365CA",)),
            # market best price (open era): grade line shopping headroom
            "max_home": _first_odds(record, ("MaxH", "BbMxH")),
            "max_draw": _first_odds(record, ("MaxD", "BbMxD")),
            "max_away": _first_odds(record, ("MaxA", "BbMxA")),
        }
        rows.append(row)
    rows.sort(key=lambda item: item["date"])
    return rows


def load_division_season(
    division: str,
    season: int,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    text = fetch_csv_text(division, season, cache_dir=cache_dir, force=force)
    if not text:
        return []
    return parse_season_csv(text, division, season)


def all_division_codes() -> list[str]:
    return sorted(set(LEAGUE_DIVISIONS.values()) | set(SECOND_DIVISIONS.values()))


def seasons_for_division(division: str, *, through: int) -> list[int]:
    first = FIRST_SEASON if DIVISION_TIER.get(division, 1) == 1 else FIRST_SEASON_TIER2
    return list(range(first, through + 1))


def devig_decimal(
    home: float | None,
    draw: float | None,
    away: float | None,
) -> tuple[float, float, float] | None:
    """Decimal 1X2 odds -> vig-free probabilities (0-1)."""
    if not home or not draw or not away:
        return None
    if home <= 1.0 or draw <= 1.0 or away <= 1.0:
        return None
    inv = (1.0 / home, 1.0 / draw, 1.0 / away)
    total = sum(inv)
    if total <= 0:
        return None
    return inv[0] / total, inv[1] / total, inv[2] / total
