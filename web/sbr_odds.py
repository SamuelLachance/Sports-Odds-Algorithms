"""Fetch historical closing odds from sportsbookreviewsonline.com archives."""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
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


def _make_datestr(raw: str, season: int, *, start: int = 8, yr_end: int = 12) -> str:
    """Map MMDD + season-start year to YYYY-MM-DD.

    ``season`` is the year the season *starts* (NBA 2023 → 2023-24). Months
    before ``start`` roll into ``season + 1``. For mid-calendar seasons that
    pass ``start=1`` (NHL COVID 2020-21), pass the calendar year of the games
    (``season + 1``) as ``season`` — otherwise Jan–Jun stamp as the prior year.
    """
    raw = str(raw).strip()
    if len(raw) == 3:
        raw = f"0{raw}"
    month = int(raw[:2])
    day = int(raw[2:4])
    year = season if month >= start else season + 1
    if month > yr_end and start == 8:
        year = season + 1 if month < start else season
    return f"{year:04d}-{month:02d}-{day:02d}"


def _clean_number(value: str) -> float:
    value = str(value).strip()
    if value in BLACKLIST or not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _optional_number(value: str) -> float | None:
    """Parse a numeric cell; return None for blank/blacklist (not 0.0)."""
    text = str(value).strip()
    if not text or text in BLACKLIST:
        # Pick'em spreads are encoded as PK — keep as 0.0.
        # EVEN sometimes appears in juice/ML cells — treat as 0 for American map.
        lowered = text.lower()
        if lowered in {"pk", "even"}:
            return 0.0
        return None
    if text.upper() == "EVEN":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_american(number: float | None, *, default: int | None = None) -> int | None:
    """Map EVEN/0 → +100; reject invalid |odds| < 100; missing → default."""
    if number is None:
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
    """Closing spread; pick'em (0 / PK) kept as 0.0."""
    return _optional_number(value)


def _spread_juice_cell(value: str, default: int | None = None) -> int | None:
    """Spread juice; EVEN (0) → +100; missing → default (None = fail closed)."""
    return _coerce_american(_optional_number(value), default=default)


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
        open_a = _optional_number(away_row[9])
        open_h = _optional_number(home_row[9])
        close_a = _optional_number(away_row[10])
        close_h = _optional_number(home_row[10])
        if open_a is not None and open_h is not None:
            # Away favorite (lower/more-negative open) → home gets +line.
            # Always emit opposite-sign home/away closes.
            if open_a < open_h:
                home_spread = -close_a if close_a is not None else None
                away_spread = close_a if close_a is not None else (
                    -close_h if close_h is not None else None
                )
            else:
                home_spread = close_h
                away_spread = -close_h if close_h is not None else (
                    close_a if close_a is not None else None
                )
        else:
            home_spread = close_h
            away_spread = close_a
            # If only one side is present, mirror the other.
            if home_spread is not None and away_spread is None:
                away_spread = -home_spread
            elif away_spread is not None and home_spread is None:
                home_spread = -away_spread
            # Same-sign closes are inconsistent — prefer home and mirror away.
            elif (
                home_spread is not None
                and away_spread is not None
                and home_spread != 0
                and away_spread != 0
                and (home_spread > 0) == (away_spread > 0)
            ):
                away_spread = -home_spread
        output.append(
            {
                "date": _make_datestr(away_row[0], season),
                "home_key": home_key,
                "away_key": away_key,
                "home_close_ml": home_close_ml,
                "away_close_ml": away_close_ml,
                "home_close_spread": home_spread,
                "away_close_spread": away_spread,
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
        # Mirror sparse / same-sign puck lines (same repair as NBA/NFL HTML).
        if home_spread is not None and away_spread is None:
            away_spread = -home_spread
        elif away_spread is not None and home_spread is None:
            home_spread = -away_spread
        elif (
            home_spread is not None
            and away_spread is not None
            and home_spread != 0
            and away_spread != 0
            and (home_spread > 0) == (away_spread > 0)
        ):
            away_spread = -home_spread
        output.append(
            {
                # COVID 2020-21 NHL slate was played in calendar 2021; SBR
                # season key is still 2020. Pass calendar year when start=1.
                "date": _make_datestr(
                    away_row[0],
                    season + 1 if covid else season,
                    start=1 if covid else 8,
                    yr_end=12,
                ),
                "home_key": home_key,
                "away_key": away_key,
                "home_close_ml": _american_ml_cell(home_row[9]),
                "away_close_ml": _american_ml_cell(away_row[9]),
                "home_close_spread": home_spread,
                "away_close_spread": away_spread,
                "home_spread_odds": _spread_juice_cell(home_row[11]),
                "away_spread_odds": _spread_juice_cell(away_row[11]),
                "source": "sbr-online",
            }
        )
    return output


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
            for cell in row.findall("m:c", ns):
                cell_type = cell.get("t")
                value_node = cell.find("m:v", ns)
                if value_node is None:
                    values.append("")
                    continue
                raw = value_node.text or ""
                if cell_type == "s":
                    values.append(shared_strings[int(raw)] if raw.isdigit() else raw)
                else:
                    values.append(raw)
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
    """Archive spread line; pick'em (0) kept as 0.0. Missing → None."""
    if value in (None, ""):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number


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
        home_key = normalize_team_key(sport, record.get("home_team", ""))
        away_key = normalize_team_key(sport, record.get("away_team", ""))
        game_date = _archive_date(record.get("date"))
        if not home_key or not away_key or not game_date:
            continue
        output.append(
            {
                "date": game_date,
                "home_key": home_key,
                "away_key": away_key,
                "home_close_ml": _parse_optional_int(record.get("home_close_ml")),
                "away_close_ml": _parse_optional_int(record.get("away_close_ml")),
                "home_close_spread": _parse_optional_float(record.get("home_close_spread")),
                "away_close_spread": _parse_optional_float(record.get("away_close_spread")),
                # Archive JSON has no juice fields — do not invent -110.
                "home_spread_odds": None,
                "away_spread_odds": None,
                "source": "sbr-archive",
            }
        )
    return output


def _dedupe_odds_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["date"], row["home_key"], row["away_key"])
        existing = merged.get(key)
        if existing is None or existing.get("source") == "sbr-archive":
            merged[key] = row
    return list(merged.values())


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
            if len(away_row) < 17 or len(home_row) < 17:
                continue
            away_name = _translate_name("mlb", away_row[3], translated)
            home_name = _translate_name("mlb", home_row[3], translated)
            away_key = normalize_team_key("mlb", away_name)
            home_key = normalize_team_key("mlb", home_name)
            if not away_key or not home_key:
                continue
            output.append(
                {
                    "date": _make_datestr(away_row[0], season, start=3, yr_end=10),
                    "home_key": home_key,
                    "away_key": away_key,
                    "home_close_ml": _american_ml_cell(home_row[16]),
                    "away_close_ml": _american_ml_cell(away_row[16]),
                    "home_close_spread": _spread_line_cell(home_row[17]),
                    "away_close_spread": _spread_line_cell(away_row[17]),
                    "home_spread_odds": _spread_juice_cell(home_row[18]),
                    "away_spread_odds": _spread_juice_cell(away_row[18]),
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
        "home_spread_odds",
        "away_spread_odds",
        "source",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
