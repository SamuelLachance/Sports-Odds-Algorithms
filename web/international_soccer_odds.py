"""International soccer closing odds from football-data.co.uk rolling exports."""

from __future__ import annotations

import csv
import io
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.football_data_uk import _parse_american_odds
from web.league_profiles import INTERNATIONAL_SOCCER_LEAGUES
from web.season_games import _load_espn_team_ids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "supplemental" / "international-closing-odds"
BASE_URL = "https://www.football-data.co.uk"

# Rolling international results + average closing 1X2 odds (misnamed "WorldCup*").
SOURCE_FILES: tuple[str, ...] = ("WorldCup2026.csv", "WorldCup2022.csv")

# football-data country labels that differ from ESPN display names.
MANUAL_COUNTRY_ALIASES: dict[str, str] = {
    "antigua and barbuda": "atg",
    "bosnia & herzegovina": "bih",
    "bosnia and herzegovina": "bih",
    "brunei": "bru",
    "central africa": "cta",
    "central african republic": "cta",
    "chad": "cha",
    "chinese taipei": "tpe",
    "congo": "cgo",
    "cook islands": "cok",
    "curacao": "cuw",
    "czech republic": "cze",
    "czechia": "cze",
    "d.r. congo": "cod",
    "dr congo": "cod",
    "congo dr": "cod",
    "democratic republic of congo": "cod",
    "djibouti": "dji",
    "eswatini": "swz",
    "guinea bissau": "gnb",
    "guinea-bissau": "gnb",
    "ireland": "irl",
    "republic of ireland": "irl",
    "ivory coast": "civ",
    "cote d'ivoire": "civ",
    "côte d'ivoire": "civ",
    "kyrgyzstan": "kgz",
    "laos": "lao",
    "latvia": "lva",
    "malaysia": "mas",
    "mauritius": "mri",  # FIFA/ESPN; Mauritania is mtn — do not conflate
    "mauritania": "mtn",
    "montserrat": "msr",
    "nepal": "nep",
    "new caledonia": "ncl",
    "north korea": "prk",
    "korea dpr": "prk",
    "papua new guinea": "png",
    "saint kitts and nevis": "skn",
    "st kitts and nevis": "skn",
    "saint lucia": "lca",
    "st lucia": "lca",
    "saint vincent and the grenadines": "vin",
    "st vincent and the grenadines": "vin",
    "samoa": "sam",
    "sao tome and principe": "stp",
    "são tomé and príncipe": "stp",
    "seychelles": "sey",
    "somalia": "som",
    "south korea": "kor",
    "korea republic": "kor",
    "south sudan": "ssd",
    "sri lanka": "sri",
    "suriname": "sur",
    "tahiti": "tah",
    "tajikistan": "tjk",
    "timor-leste": "tls",
    "east timor": "tls",
    "tonga": "tga",
    "trinidad & tobago": "tri",
    "trinidad and tobago": "tri",
    "turkey": "tur",
    "türkiye": "tur",
    "usa": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "uae": "uae",
    "united arab emirates": "uae",
    "ukraine": "ukr",
    "wales": "wal",
    "northern ireland": "nir",
    "scotland": "sco",
    "england": "eng",
    "holland": "ned",
    "netherlands": "ned",
    "republic of ireland": "irl",
}


def _normalize_country_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("&", "and").split())


@lru_cache(maxsize=1)
def _espn_country_to_abbr() -> dict[str, str]:
    mapping: dict[str, str] = {}
    leagues = set(INTERNATIONAL_SOCCER_LEAGUES) | {"fifa_friendlies"}
    for league in leagues:
        for _team_id, abbr, name in _load_espn_team_ids(league):
            if not abbr or not name:
                continue
            mapping[_normalize_country_label(name)] = abbr.lower()
            mapping[abbr.lower()] = abbr.lower()
    for label, abbr in MANUAL_COUNTRY_ALIASES.items():
        mapping[_normalize_country_label(label)] = abbr.lower()
    return mapping


def country_to_espn_abbr(label: str) -> str | None:
    mapping = _espn_country_to_abbr()
    normalized = _normalize_country_label(label)
    if normalized in mapping:
        return mapping[normalized]
    for key, abbr in mapping.items():
        if normalized.startswith(key) or key.startswith(normalized):
            return abbr
    return None


def _parse_fd_date(raw: str) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            parsed = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _fetch_source_file(filename: str) -> str | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / filename
    if cache_path.is_file():
        return cache_path.read_text(encoding="utf-8-sig", errors="replace")

    url = f"{BASE_URL}/{filename}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8-sig", errors="replace")
    except OSError:
        return None
    if "Home" not in text and "HomeTeam" not in text:
        return None
    cache_path.write_text(text, encoding="utf-8")
    return text


@lru_cache(maxsize=1)
def load_international_fd_games() -> list[dict[str, Any]]:
    """Completed international matches with average closing 1X2 odds."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    for filename in SOURCE_FILES:
        raw = _fetch_source_file(filename)
        if not raw:
            continue
        reader = csv.DictReader(io.StringIO(raw))
        for record in reader:
            home_label = (record.get("Home") or record.get("HomeTeam") or "").strip()
            away_label = (record.get("Away") or record.get("AwayTeam") or "").strip()
            if not home_label or not away_label:
                continue
            home_key = country_to_espn_abbr(home_label)
            away_key = country_to_espn_abbr(away_label)
            if not home_key or not away_key:
                continue
            game_date = _parse_fd_date(record.get("Date") or "")
            if not game_date:
                continue
            try:
                home_goals = int(record.get("HG") or record.get("FTHG") or "")
                away_goals = int(record.get("AG") or record.get("FTAG") or "")
            except ValueError:
                continue
            home_odds = _parse_american_odds(
                record.get("H_Avg") or record.get("AvgH") or record.get("H_Max")
            )
            draw_odds = _parse_american_odds(
                record.get("D_Avg") or record.get("AvgD") or record.get("D_Max")
            )
            away_odds = _parse_american_odds(
                record.get("A_Avg") or record.get("AvgA") or record.get("A_Max")
            )
            if home_odds is None or draw_odds is None or away_odds is None:
                continue
            signature = (game_date, home_key, away_key, home_goals, away_goals)
            if signature in seen:
                continue
            seen.add(signature)
            rows.append(
                {
                    "date": game_date,
                    "home_key": home_key,
                    "away_key": away_key,
                    "home_name": home_label,
                    "away_name": away_label,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                    "source": "football-data.co.uk/international",
                }
            )
    rows.sort(key=lambda item: item["date"])
    return rows


@lru_cache(maxsize=1)
def load_international_odds_index() -> dict[tuple[str, str, str], dict[str, int | None]]:
    """Map (iso_date, home_key, away_key) -> American 1X2 closing odds."""
    index: dict[tuple[str, str, str], dict[str, int | None]] = {}
    for record in load_international_fd_games():
        index[(record["date"], record["home_key"], record["away_key"])] = {
            "home_odds": record["home_odds"],
            "draw_odds": record["draw_odds"],
            "away_odds": record["away_odds"],
        }
    return index


def international_odds_lookup(
    game_date: str,
    home_key: str,
    away_key: str,
) -> dict[str, int | None] | None:
  index = load_international_odds_index()
  iso = game_date[:10] if game_date else ""
  return index.get((iso, home_key.lower(), away_key.lower()))


def international_odds_metadata() -> dict[str, Any]:
    index = load_international_odds_index()
    return {
        "source": "football-data.co.uk",
        "files": list(SOURCE_FILES),
        "rows": len(index),
    }
