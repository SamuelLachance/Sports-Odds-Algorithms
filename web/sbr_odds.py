"""Fetch historical closing odds from sportsbookreviewsonline.com archives."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import date as date_cls
from pathlib import Path
from typing import Any

from web.closing_odds_db import ODDS_DIR, normalize_team_key

USER_AGENT = "Sports-Odds-Algorithms/2.0"
BLACKLIST = frozenset({"pk", "PK", "NL", "nl", "a100", "a105", "a110", ".5+03", ".5ev", "-"})

TRANSLATED_URL = (
    "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/"
    "master/config/translated.json"
)
SBR_ARCHIVE_URLS = {
    "nba": (
        "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/"
        "main/data/nba_archive_10Y.json"
    ),
    "nhl": (
        "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/"
        "main/data/nhl_archive_10Y.json"
    ),
    "mlb": (
        "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/"
        "main/data/mlb_archive_10Y.json"
    ),
    "nfl": (
        "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/"
        "main/data/nfl_archive_10Y.json"
    ),
}

# SBR nickname / city labels → nflverse abbreviations used by the training table.
NFL_NICK_TO_ABBR: dict[str, str] = {
    "cardinals": "ari",
    "arizona": "ari",
    "falcons": "atl",
    "atlanta": "atl",
    "ravens": "bal",
    "baltimore": "bal",
    "bills": "buf",
    "buffalo": "buf",
    "panthers": "car",
    "carolina": "car",
    "bears": "chi",
    "chicago": "chi",
    "bengals": "cin",
    "cincinnati": "cin",
    "browns": "cle",
    "cleveland": "cle",
    "cowboys": "dal",
    "dallas": "dal",
    "broncos": "den",
    "denver": "den",
    "lions": "det",
    "detroit": "det",
    "packers": "gb",
    "greenbay": "gb",
    "texans": "hou",
    "houston": "hou",
    "colts": "ind",
    "indianapolis": "ind",
    "jaguars": "jax",
    "jacksonville": "jax",
    "chiefs": "kc",
    "kansascity": "kc",
    "raiders": "lv",
    "lasvegas": "lv",
    "oakland": "oak",
    "chargers": "lac",
    "lachargers": "lac",
    "sandiego": "sd",
    "rams": "la",
    "larams": "la",
    "stlouis": "stl",
    "dolphins": "mia",
    "miami": "mia",
    "vikings": "min",
    "minnesota": "min",
    "patriots": "ne",
    "newengland": "ne",
    "saints": "no",
    "neworleans": "no",
    "giants": "nyg",
    "nygiants": "nyg",
    "jets": "nyj",
    "nyjets": "nyj",
    "eagles": "phi",
    "philadelphia": "phi",
    "steelers": "pit",
    "pittsburgh": "pit",
    "49ers": "sf",
    "fortyniners": "sf",
    "sanfrancisco": "sf",
    "seahawks": "sea",
    "seattle": "sea",
    "buccaneers": "tb",
    "tampabay": "tb",
    "titans": "ten",
    "tennessee": "ten",
    "commanders": "was",
    "washington": "was",
    "redskins": "was",
    "footballteam": "was",
}


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _load_translated() -> dict[str, dict[str, str]]:
    try:
        payload = json.loads(_fetch_text(TRANSLATED_URL))
    except (OSError, json.JSONDecodeError):
        return {}
    return {key: value for key, value in payload.items() if isinstance(value, dict)}


def _translate_name(sport: str, raw: str, translated: dict[str, dict[str, str]]) -> str:
    sport_map = translated.get(sport, {})
    if raw in sport_map:
        return sport_map[raw]
    for prefix, suffix in sport_map.items():
        if raw.startswith(prefix):
            return suffix
    return raw


def _parse_table_rows(html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S):
        cells = [
            re.sub(r"<[^>]+>", "", cell).strip()
            for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S)
        ]
        if cells:
            rows.append(cells)
    return rows


def _make_datestr(
    raw: str, season: int, *, start: int = 8, yr_end: int = 12
) -> str | None:
    """Map MMDD + season-start year to YYYY-MM-DD.

    ``season`` is the year the season *starts* (NBA 2023 → 2023-24). Months
    before ``start`` roll into ``season + 1``. Calendar-year sports (MLB) pass
    ``start=1`` so Jan–Dec stay on ``season``. ``yr_end`` is retained for
    callers but no longer flips the year (NBA ``yr_end=12`` never matched a
    valid month; MLB callers should use ``start=1`` instead of ``start=3``).

    Empty / short / non-numeric / impossible calendar cells return ``None`` so
    callers can skip the pair instead of aborting the whole season scrape.
    """
    del yr_end  # kept for call-site compatibility
    text = str(raw).strip()
    if len(text) == 3:
        text = f"0{text}"
    if len(text) < 4 or not text[:4].isdigit():
        return None
    month = int(text[:2])
    day = int(text[2:4])
    year = season if month >= start else season + 1
    try:
        return date_cls(year, month, day).isoformat()
    except ValueError:
        return None


def _clean_number(value: str) -> float:
    value = str(value).strip()
    if value in BLACKLIST or not value:
        return 0.0
    try:
        parsed = float(value)
    except ValueError:
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return parsed


def _optional_number(value: str) -> float | None:
    """Parse a numeric cell; return None for blank/blacklist (not 0.0)."""
    text = str(value).strip()
    if not text:
        return None
    # Case-insensitive PK/EVEN (HTML often emits ``Pk``); match archive parsers.
    upper = text.upper()
    if upper in {"PK", "EVEN"}:
        return 0.0
    if text in BLACKLIST:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _coerce_american(number: float | None, *, default: int | None = None) -> int | None:
    """Map EVEN/0 → +100; reject invalid |odds| < 100; missing → default."""
    if number is None:
        return default
    if not math.isfinite(number):
        return default
    if number == 0:
        return 100
    if abs(number) < 100:
        return None
    return int(number)


def _american_ml_cell(value: str) -> int | None:
    """Closing moneyline; ESPN/SBR EVEN (0) → +100. Missing → None."""
    return _coerce_american(_optional_number(value))


def _spread_line_cell(value: str) -> float | None:
    """Closing spread; pick'em (0 / PK) kept as 0.0.

    Reject |x| ≥ 100 (American ML/juice dumps misaligned into handicap cells).
    """
    number = _optional_number(value)
    if number is None:
        return None
    if abs(number) >= 100.0:
        return None
    return number


def _spread_juice_cell(value: str, default: int | None = None) -> int | None:
    """Spread juice; EVEN (0) → +100; missing → default (None = fail closed)."""
    return _coerce_american(_optional_number(value), default=default)


def _repair_same_sign_spreads(
    home_spread: float | None,
    away_spread: float | None,
    *,
    home_ml: float | int | None = None,
    away_ml: float | int | None = None,
) -> tuple[float | None, float | None]:
    """Mirror sparse lines; on same-sign dumps, assign −line to the ML favorite.

    SBR often duplicates the favorite handicap on both rows. Always trusting the
    home cell flips the board when the away side is chalk.

    Also reconcile opposite-sign lines that disagree with a clear ML favorite
    (e.g. flat ESPN ``spread: 1.5`` without favorite flags → home dog while
    home is ML chalk).
    """
    if home_spread is not None and away_spread is None:
        return home_spread, -home_spread
    if away_spread is not None and home_spread is None:
        return -away_spread, away_spread
    if home_spread is None or away_spread is None:
        return home_spread, away_spread
    home_f = float(home_spread)
    away_f = float(away_spread)
    # bool is a subclass of int; False→0.0 must not invent EVEN ML chalk.
    if isinstance(home_ml, bool):
        home_ml = None
    if isinstance(away_ml, bool):
        away_ml = None
    # Pick'em (0) on one side with a real line on the other is inconsistent.
    pk_mismatch = (home_f == 0.0) ^ (away_f == 0.0)
    same_sign = (
        home_f != 0.0
        and away_f != 0.0
        and (home_f > 0) == (away_f > 0)
    )
    if not pk_mismatch and not same_sign:
        # Already opposite (or both PK). If ML names a different chalk, flip.
        if (
            home_f != 0.0
            and away_f != 0.0
            and home_ml is not None
            and away_ml is not None
            and float(home_ml) != float(away_ml)
        ):
            ml_home_favorite = float(home_ml) < float(away_ml)
            spread_home_favorite = home_f < 0.0
            if ml_home_favorite != spread_home_favorite:
                magnitude = max(abs(home_f), abs(away_f))
                if ml_home_favorite:
                    return -magnitude, magnitude
                return magnitude, -magnitude
        return home_spread, away_spread
    magnitude = max(abs(home_f), abs(away_f))
    away_is_favorite = (
        away_ml is not None
        and home_ml is not None
        and float(away_ml) < float(home_ml)
    )
    if not away_is_favorite and pk_mismatch and (
        home_ml is None or away_ml is None or float(away_ml) == float(home_ml)
    ):
        # Prefer the sign of the non-zero dump when MLs do not break the tie.
        away_is_favorite = away_f < 0 or home_f > 0
    if away_is_favorite:
        return magnitude, -magnitude
    return -magnitude, magnitude


def _pairwise(rows: list[list[str]]) -> list[tuple[list[str], list[str]]]:
    pairs: list[tuple[list[str], list[str]]] = []
    for index in range(1, len(rows) - 1, 2):
        pairs.append((rows[index], rows[index + 1]))
    return pairs


def _rows_from_html_table(sport: str, season: int, html: str, translated: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    table_rows = _parse_table_rows(html)
    if len(table_rows) < 3:
        return []
    body = table_rows[1:]
    output: list[dict[str, Any]] = []
    for away_row, home_row in _pairwise(body):
        if len(away_row) < 12 or len(home_row) < 12:
            continue
        away_name = _translate_name(sport, away_row[3], translated)
        home_name = _translate_name(sport, home_row[3], translated)
        away_key = normalize_team_key(sport, away_name)
        home_key = normalize_team_key(sport, home_name)
        if not away_key or not home_key:
            continue
        away_close_ml = _american_ml_cell(away_row[11])
        home_close_ml = _american_ml_cell(home_row[11])
        open_a = _spread_line_cell(away_row[9])
        open_h = _spread_line_cell(home_row[9])
        close_a = _spread_line_cell(away_row[10])
        close_h = _spread_line_cell(home_row[10])
        if open_a is not None and open_h is not None:
            # Away favorite (lower/more-negative open) → home gets +line.
            # Equal opens (SBR often duplicates the favorite line) → break ties
            # with moneyline when available; else treat home as chalk.
            # Close ML also wins when the favorite flipped after open: mirroring
            # from the open chalk invents opposite-sign closes that skip
            # ``_repair_same_sign_spreads`` and keep the wrong side as favorite.
            away_is_favorite = open_a < open_h
            if away_close_ml is not None and home_close_ml is not None:
                ml_away_favorite = float(away_close_ml) < float(home_close_ml)
                if open_a == open_h or ml_away_favorite != away_is_favorite:
                    away_is_favorite = ml_away_favorite
            if away_is_favorite:
                home_spread = -close_a if close_a is not None else None
                away_spread = close_a if close_a is not None else (
                    -close_h if close_h is not None else None
                )
                home_open_spread = -open_a
                away_open_spread = open_a
            else:
                home_spread = close_h
                away_spread = -close_h if close_h is not None else (
                    close_a if close_a is not None else None
                )
                home_open_spread = open_h
                away_open_spread = -open_h if open_h is not None else (
                    open_a if open_a is not None else None
                )
        else:
            home_spread = close_h
            away_spread = close_a
            home_open_spread = None
            away_open_spread = None
        # Always emit opposite-sign home/away closes (open and no-open paths).
        home_spread, away_spread = _repair_same_sign_spreads(
            home_spread,
            away_spread,
            home_ml=home_close_ml,
            away_ml=away_close_ml,
        )
        home_open_spread, away_open_spread = _repair_same_sign_spreads(
            home_open_spread,
            away_open_spread,
            home_ml=home_close_ml,
            away_ml=away_close_ml,
        )
        game_date = _make_datestr(away_row[0], season)
        if not game_date:
            continue
        output.append(
            {
                "date": game_date,
                "home_key": home_key,
                "away_key": away_key,
                "home_close_ml": home_close_ml,
                "away_close_ml": away_close_ml,
                "home_close_spread": home_spread,
                "away_close_spread": away_spread,
                "home_open_ml": None,  # HTML archive has no open ML
                "away_open_ml": None,
                "home_open_spread": home_open_spread,
                "away_open_spread": away_open_spread,
                # NBA/NFL HTML archives do not expose spread juice — fail closed.
                "home_spread_odds": None,
                "away_spread_odds": None,
                "source": "sbr-online",
            }
        )
    return output


def _rows_from_nhl_html(sport: str, season: int, html: str, translated: dict[str, dict[str, str]], *, covid: bool = False) -> list[dict[str, Any]]:
    table_rows = _parse_table_rows(html)
    body = table_rows[1:]
    output: list[dict[str, Any]] = []
    for away_row, home_row in _pairwise(body):
        # Need ML [9] + spread [10]; juice [11] is optional (sparse HTML).
        if len(away_row) < 10 or len(home_row) < 10:
            continue
        away_name = _translate_name(sport, away_row[3], translated)
        home_name = _translate_name(sport, home_row[3], translated)
        away_key = normalize_team_key(sport, away_name)
        home_key = normalize_team_key(sport, home_name)
        if not away_key or not home_key:
            continue
        home_spread = _spread_line_cell(home_row[10])
        away_spread = _spread_line_cell(away_row[10])
        home_close_ml = _american_ml_cell(home_row[9])
        away_close_ml = _american_ml_cell(away_row[9])
        # Mirror sparse / same-sign puck lines; ML picks the favorite side.
        home_spread, away_spread = _repair_same_sign_spreads(
            home_spread,
            away_spread,
            home_ml=home_close_ml,
            away_ml=away_close_ml,
        )
        # COVID 2020-21 NHL slate was played in calendar 2021; SBR
        # season key is still 2020. Pass calendar year when start=1.
        game_date = _make_datestr(
            away_row[0],
            season + 1 if covid else season,
            start=1 if covid else 8,
            yr_end=12,
        )
        if not game_date:
            continue
        output.append(
            {
                "date": game_date,
                "home_key": home_key,
                "away_key": away_key,
                "home_close_ml": home_close_ml,
                "away_close_ml": away_close_ml,
                "home_close_spread": home_spread,
                "away_close_spread": away_spread,
                "home_spread_odds": (
                    _spread_juice_cell(home_row[11]) if len(home_row) > 11 else None
                ),
                "away_spread_odds": (
                    _spread_juice_cell(away_row[11]) if len(away_row) > 11 else None
                ),
                "source": "sbr-online",
            }
        )
    return output


def _xlsx_col_index(cell_ref: str) -> int | None:
    """Parse OpenXML cell ref (``A1``, ``Q12``) to 0-based column index."""
    letters = "".join(ch for ch in str(cell_ref) if ch.isalpha())
    if not letters:
        return None
    index = 0
    for ch in letters.upper():
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return index - 1


def _xlsx_rows(path_bytes: bytes) -> list[list[str]]:
    rows: list[list[str]] = []
    with zipfile.ZipFile(io.BytesIO(path_bytes)) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("m:si", ns):
                text = "".join(node.text or "" for node in si.findall(".//m:t", ns))
                shared_strings.append(text)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for row in sheet.findall("m:sheetData/m:row", ns):
            values: list[str] = []
            next_col = 0
            for cell in row.findall("m:c", ns):
                cell_type = cell.get("t")
                value_node = cell.find("m:v", ns)
                if value_node is None:
                    raw_text = ""
                else:
                    raw = value_node.text or ""
                    if cell_type == "s":
                        if raw.isdigit():
                            idx = int(raw)
                            raw_text = (
                                shared_strings[idx]
                                if 0 <= idx < len(shared_strings)
                                else ""
                            )
                        else:
                            raw_text = raw
                    else:
                        raw_text = raw
                col = _xlsx_col_index(cell.get("r") or "")
                if col is None:
                    col = next_col
                while len(values) <= col:
                    values.append("")
                values[col] = raw_text
                next_col = col + 1
            rows.append(values)
    return rows


def _archive_date(raw: Any) -> str:
    text = str(raw or "").strip().split(".")[0]
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text[:10]


def _parse_optional_int(value: Any) -> int | None:
    """Archive moneyline; EVEN (0) → +100. Missing → None.

    Must match ``_american_ml_cell`` used by HTML/xlsx scrapers — do not treat
    numeric 0 as missing (that silently drops even-money closers).
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.upper() in {"EVEN", "PK"}:
        return 100
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return _coerce_american(number)


def _parse_optional_float(value: Any) -> float | None:
    """Archive spread line; pick'em (0 / PK / EVEN) kept as 0.0. Missing → None.

    Must match ``_spread_line_cell`` used by HTML scrapers (reject |x| ≥ 100).
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.upper() in {"PK", "EVEN"}:
        return 0.0
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if abs(number) >= 100.0:
        return None
    return number


def nfl_team_key(label: Any) -> str | None:
    """Map city/nickname/full name/abbr → nflverse lowercase abbr (``was``, ``la``)."""
    raw = str(label or "").strip()
    if not raw:
        return None
    compact = (
        raw.lower()
        .replace(".", "")
        .replace("-", "")
        .replace("'", "")
        .replace(" ", "")
    )
    # Common Odds API / full-name forms.
    full_name_aliases = {
        "cincinnatibengals": "bengals",
        "tampabaybuccaneers": "buccaneers",
        "washingtoncommanders": "commanders",
        "washingtonfootballteam": "footballteam",
        "losangelesrams": "larams",
        "losangeleschargers": "lachargers",
        "newyorkgiants": "nygiants",
        "newyorkjets": "nyjets",
        "newenglandpatriots": "newengland",
        "neworleanssaints": "neworleans",
        "kansascitychiefs": "kansascity",
        "greenbaypackers": "greenbay",
        "sanfrancisco49ers": "sanfrancisco",
        "lasvegasraiders": "lasvegas",
        "oaklandraiders": "oakland",
        "sandiegochargers": "sandiego",
        "stlouisrams": "stlouis",
        "arizonacardinals": "cardinals",
        "atlantafalcons": "falcons",
        "baltimoreravens": "ravens",
        "buffalobills": "bills",
        "carolinapanthers": "panthers",
        "chicagobears": "bears",
        "clevelandbrowns": "browns",
        "dallascowboys": "cowboys",
        "denverbroncos": "broncos",
        "detroitlions": "lions",
        "houstontexans": "texans",
        "indianapoliscolts": "colts",
        "jacksonvillejaguars": "jaguars",
        "miamidolphins": "dolphins",
        "minnesotavikings": "vikings",
        "philadelphiaeagles": "eagles",
        "pittsburghsteelers": "steelers",
        "seattleseahawks": "seahawks",
        "tennesseetitans": "titans",
    }
    compact = full_name_aliases.get(compact, compact)
    if compact in NFL_NICK_TO_ABBR:
        abbr = NFL_NICK_TO_ABBR[compact]
    else:
        # Try last token as nickname ("Buffalo Bills" → bills).
        parts = raw.lower().replace(".", "").split()
        nick = parts[-1] if parts else compact
        if nick in NFL_NICK_TO_ABBR:
            abbr = NFL_NICK_TO_ABBR[nick]
        else:
            mapped = normalize_team_key("nfl", raw)
            if mapped and mapped in NFL_NICK_TO_ABBR:
                abbr = NFL_NICK_TO_ABBR[mapped]
            elif mapped and len(mapped) <= 3:
                abbr = mapped
            else:
                abbr = NFL_NICK_TO_ABBR.get(mapped or "", mapped or "")
    if not abbr:
        return None
    # Align with nflverse / training table keys.
    if abbr in {"wsh", "was"}:
        return "was"
    if abbr == "lar":
        return "la"
    if abbr == "jac":
        return "jax"
    return abbr


def _archive_team_key(sport: str, label: Any) -> str | None:
    """Normalize archive team labels; NFL nicknames → nflverse abbrs."""
    sport = sport.lower()
    if sport == "nfl":
        return nfl_team_key(label)
    return normalize_team_key(sport, str(label or "").strip())


def fetch_sbr_archive_rows(sport: str) -> list[dict[str, Any]]:
    """Pre-scraped SBR archive JSON (2011-2022) when live HTML pages are empty."""
    sport = sport.lower()
    url = SBR_ARCHIVE_URLS.get(sport)
    if not url:
        return []
    try:
        payload = json.loads(_fetch_text(url))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  skipped {sport} archive: {exc}", flush=True)
        return []
    if not isinstance(payload, list):
        return []

    output: list[dict[str, Any]] = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        home_key = _archive_team_key(sport, record.get("home_team", ""))
        away_key = _archive_team_key(sport, record.get("away_team", ""))
        game_date = _archive_date(record.get("date"))
        if not home_key or not away_key or not game_date:
            continue
        home_close_ml = _parse_optional_int(record.get("home_close_ml"))
        away_close_ml = _parse_optional_int(record.get("away_close_ml"))
        home_spread = _parse_optional_float(record.get("home_close_spread"))
        away_spread = _parse_optional_float(record.get("away_close_spread"))
        home_spread, away_spread = _repair_same_sign_spreads(
            home_spread,
            away_spread,
            home_ml=home_close_ml,
            away_ml=away_close_ml,
        )
        home_open_spread = _parse_optional_float(record.get("home_open_spread"))
        away_open_spread = _parse_optional_float(record.get("away_open_spread"))
        home_open_spread, away_open_spread = _repair_same_sign_spreads(
            home_open_spread,
            away_open_spread,
            home_ml=home_close_ml,
            away_ml=away_close_ml,
        )
        open_total = _parse_optional_float(
            record.get("open_over_under") or record.get("open_total")
        )
        close_total = _parse_optional_float(
            record.get("close_over_under") or record.get("close_total")
        )
        # Totals are magnitudes; reject pick'em 0 that slipped through.
        if open_total is not None and open_total <= 0:
            open_total = None
        if close_total is not None and close_total <= 0:
            close_total = None
        output.append(
            {
                "date": game_date,
                "home_key": home_key,
                "away_key": away_key,
                "home_close_ml": home_close_ml,
                "away_close_ml": away_close_ml,
                "home_close_spread": home_spread,
                "away_close_spread": away_spread,
                "home_open_ml": _parse_optional_int(record.get("home_open_ml")),
                "away_open_ml": _parse_optional_int(record.get("away_open_ml")),
                "home_open_spread": home_open_spread,
                "away_open_spread": away_open_spread,
                "open_total": open_total,
                "close_total": close_total,
                # Archive JSON has no juice fields — do not invent -110.
                "home_spread_odds": None,
                "away_spread_odds": None,
                "source": "sbr-archive",
            }
        )
    return output


def _odds_row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    """Comparable close fields only (ignore source) for DH / duplicate detection."""
    return (
        row.get("home_close_ml"),
        row.get("away_close_ml"),
        row.get("home_close_spread"),
        row.get("away_close_spread"),
        row.get("home_spread_odds"),
        row.get("away_spread_odds"),
    )


def _row_has_close_fields(row: dict[str, Any]) -> bool:
    """True when any close ML/spread/juice field is present (not an empty stub)."""
    return any(v is not None for v in _odds_row_signature(row))


def _dedupe_odds_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse identical rows; keep conflicting same-key lines (MLB DH fail-closed).

    Prefer online over archive when online has real close fields. Empty online
    stubs must not wipe filled archive rows for the same key. Distinct odds under
    the same (date, home, away) must both survive so ``closing_odds_db`` can mark
    the key ambiguous instead of silently picking one DH leg.
    """
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["date"], row["home_key"], row["away_key"])
        by_key.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for group in by_key.values():
        online = [r for r in group if r.get("source") != "sbr-archive"]
        online_rich = [r for r in online if _row_has_close_fields(r)]
        if online_rich:
            pool = online_rich
        else:
            archive = [r for r in group if r.get("source") == "sbr-archive"]
            archive_rich = [r for r in archive if _row_has_close_fields(r)]
            pool = archive_rich if archive_rich else (online if online else group)
        seen: set[tuple[Any, ...]] = set()
        for row in pool:
            sig = _odds_row_signature(row)
            if sig in seen:
                continue
            seen.add(sig)
            output.append(row)
    return output


def fetch_sbr_season_rows(sport: str, season: int, translated: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    sport = sport.lower()
    if sport in {"nba", "nfl"}:
        season_label = f"{season}-{season + 1}"
        url = f"https://www.sportsbookreviewsonline.com/scoresoddsarchives/{sport}-odds-{season_label}"
        html = _fetch_text(url)
        return _rows_from_html_table(sport, season, html, translated)
    if sport == "nhl":
        season_label = "2021" if season == 2020 else f"{season}-{season + 1}"
        covid = season == 2020
        url = f"https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl-odds-{season_label}"
        html = _fetch_text(url)
        return _rows_from_nhl_html(sport, season, html, translated, covid=covid)
    if sport == "mlb":
        url = (
            "https://www.sportsbookreviewsonline.com/wp-content/uploads/"
            f"sportsbookreviewsonline_com_737/mlb-odds-{season}.xlsx"
        )
        sheet_rows = _xlsx_rows(_fetch_bytes(url))
        body = sheet_rows[1:]
        output: list[dict[str, Any]] = []
        for index in range(1, len(body) - 1, 2):
            away_row, home_row = body[index], body[index + 1]
            # Need ML [16] + spread [17]; juice [18] is optional (sparse xlsx).
            if len(away_row) < 17 or len(home_row) < 17:
                continue
            away_name = _translate_name("mlb", away_row[3], translated)
            home_name = _translate_name("mlb", home_row[3], translated)
            away_key = normalize_team_key("mlb", away_name)
            home_key = normalize_team_key("mlb", home_name)
            if not away_key or not home_key:
                continue
            home_spread = _spread_line_cell(home_row[17])
            away_spread = _spread_line_cell(away_row[17])
            home_close_ml = _american_ml_cell(home_row[16])
            away_close_ml = _american_ml_cell(away_row[16])
            # Mirror sparse / same-sign run lines; ML picks the favorite side.
            home_spread, away_spread = _repair_same_sign_spreads(
                home_spread,
                away_spread,
                home_ml=home_close_ml,
                away_ml=away_close_ml,
            )
            # MLB season year == calendar year (Mar–Nov games).
            game_date = _make_datestr(away_row[0], season, start=1, yr_end=12)
            if not game_date:
                continue
            output.append(
                {
                    "date": game_date,
                    "home_key": home_key,
                    "away_key": away_key,
                    "home_close_ml": home_close_ml,
                    "away_close_ml": away_close_ml,
                    "home_close_spread": home_spread,
                    "away_close_spread": away_spread,
                    "home_spread_odds": (
                        _spread_juice_cell(home_row[18]) if len(home_row) > 18 else None
                    ),
                    "away_spread_odds": (
                        _spread_juice_cell(away_row[18]) if len(away_row) > 18 else None
                    ),
                    "source": "sbr-online",
                }
            )
        return output
    return []


def write_sbr_cache(sport: str, seasons: list[int], path: Path | None = None) -> int:
    translated = _load_translated()
    rows: list[dict[str, Any]] = list(fetch_sbr_archive_rows(sport))
    for season in seasons:
        try:
            rows.extend(fetch_sbr_season_rows(sport, season, translated))
        except OSError as exc:
            print(f"  skipped {sport} {season}: {exc}", flush=True)
    rows = _dedupe_odds_rows(rows)
    out = path or (ODDS_DIR / f"{sport.lower()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "home_key",
        "away_key",
        "home_close_ml",
        "away_close_ml",
        "home_close_spread",
        "away_close_spread",
        "home_open_ml",
        "away_open_ml",
        "home_open_spread",
        "away_open_spread",
        "home_spread_odds",
        "away_spread_odds",
        "open_total",
        "close_total",
        "source",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
