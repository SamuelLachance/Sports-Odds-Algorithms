"""Load historical soccer results (+ closing odds) from football-data.co.uk."""

from __future__ import annotations

import csv
import io
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.league_profiles import BACKTEST_PRO_SEASONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "supplemental" / "football-data-uk"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# football-data league code per internal league id
LEAGUE_CODES: dict[str, str] = {
    "epl": "E0",
    "bundesliga": "D1",
    "laliga": "SP1",
    "seriea": "I1",
    "ligue1": "F1",
}

# football-data HomeTeam/AwayTeam labels -> ESPN abbr (lowercase)
TEAM_TO_ABBR: dict[str, dict[str, str]] = {
    "epl": {
        "Arsenal": "ars",
        "Aston Villa": "avl",
        "Bournemouth": "bou",
        "AFC Bournemouth": "bou",
        "Brentford": "bre",
        "Brighton": "bha",
        "Brighton & Hove Albion": "bha",
        "Burnley": "bur",
        "Chelsea": "che",
        "Crystal Palace": "cry",
        "Everton": "eve",
        "Fulham": "ful",
        "Ipswich": "ips",
        "Ipswich Town": "ips",
        "Leeds": "lee",
        "Leeds United": "lee",
        "Leicester": "lei",
        "Leicester City": "lei",
        "Liverpool": "liv",
        "Luton": "lut",
        "Luton Town": "lut",
        "Man City": "mnc",
        "Manchester City": "mnc",
        "Man United": "man",
        "Manchester United": "man",
        "Newcastle": "new",
        "Newcastle United": "new",
        "Norwich": "nor",
        "Norwich City": "nor",
        "Nott'm Forest": "nfo",
        "Nottingham Forest": "nfo",
        "Sheffield United": "shu",
        "Southampton": "sou",
        "Sunderland": "sun",
        "Tottenham": "tot",
        "Tottenham Hotspur": "tot",
        "Watford": "wat",
        "West Brom": "wba",
        "West Bromwich Albion": "wba",
        "West Ham": "whu",
        "West Ham United": "whu",
        "Wolves": "wol",
        "Wolverhampton Wanderers": "wol",
    },
    "bundesliga": {
        "Bayern Munich": "bay",
        "Dortmund": "dor",
        "RB Leipzig": "rbl",
        "Leverkusen": "b04",
        "Bayer Leverkusen": "b04",
        "Ein Frankfurt": "sge",
        "Eintracht Frankfurt": "sge",
        "Freiburg": "scf",
        "Hoffenheim": "tsg",
        "Wolfsburg": "wob",
        "Mainz": "mai",
        "Augsburg": "fca",
        "Union Berlin": "fcu",
        "Werder Bremen": "svw",
        "Bochum": "boc",
        "Heidenheim": "fch",
        "Darmstadt": "dam",
        "Koln": "koe",
        "FC Koln": "koe",
        "Gladbach": "bmg",
        "M'gladbach": "bmg",
        "Stuttgart": "vfb",
        "Hertha": "ber",
        "Schalke 04": "s04",
    },
    "laliga": {
        "Barcelona": "bar",
        "Real Madrid": "rma",
        "Ath Madrid": "atm",
        "Atletico Madrid": "atm",
        "Sevilla": "sev",
        "Valencia": "val",
        "Villarreal": "vil",
        "Betis": "bet",
        "Real Betis": "bet",
        "Sociedad": "rso",
        "Real Sociedad": "rso",
        "Athletic Club": "ath",
        "Ath Bilbao": "ath",
        "Celta": "cel",
        "Celta Vigo": "cel",
        "Getafe": "get",
        "Girona": "gir",
        "Las Palmas": "lpa",
        "Mallorca": "mal",
        "Osasuna": "osa",
        "Alaves": "ala",
        "Cadiz": "cad",
        "Granada": "gra",
        "Leganes": "leg",
        "Levante": "lev",
        "Espanyol": "esp",
        "Valladolid": "vll",
        "Almeria": "alm",
        "Vallecano": "ray",
        "Rayo Vallecano": "ray",
    },
    "seriea": {
        "Inter": "int",
        "Internazionale": "int",
        "Milan": "mil",
        "AC Milan": "mil",
        "Juventus": "juv",
        "Napoli": "nap",
        "Roma": "rom",
        "Lazio": "laz",
        "Atalanta": "ata",
        "Fiorentina": "fio",
        "Torino": "tor",
        "Bologna": "bol",
        "Udinese": "udi",
        "Sassuolo": "sas",
        "Empoli": "emp",
        "Monza": "mon",
        "Lecce": "lec",
        "Verona": "ver",
        "Hellas Verona": "ver",
        "Genoa": "gen",
        "Cagliari": "cag",
        "Frosinone": "fro",
        "Salernitana": "sal",
        "Spezia": "spe",
        "Sampdoria": "sam",
        "Venezia": "ven",
        "Como": "com",
        "Parma": "par",
    },
    "ligue1": {
        "Paris SG": "psg",
        "Paris Saint-Germain": "psg",
        "Marseille": "olm",
        "Lyon": "ol",
        "Lille": "lil",
        "Monaco": "asm",
        "Rennes": "ren",
        "Nice": "ogc",
        "Lens": "rcl",
        "Reims": "sdr",
        "Strasbourg": "rcs",
        "Montpellier": "mhsc",
        "Nantes": "fcn",
        "Brest": "sb29",
        "Toulouse": "tou",
        "Lorient": "fcl",
        "Clermont": "cfe",
        "Metz": "fcm",
        "Le Havre": "hac",
        "Angers": "sco",
        "Auxerre": "aja",
        "St Etienne": "asse",
        "Troyes": "est",
    },
}


def _season_tags(count: int = BACKTEST_PRO_SEASONS) -> list[str]:
    """Recent football-data season folder tags (e.g. 2425)."""
    now = datetime.now(timezone.utc)
    # European seasons: start year of season
    start_year = now.year if now.month >= 7 else now.year - 1
    tags: list[str] = []
    for offset in range(count):
        y1 = start_year - offset
        y2 = y1 + 1
        tags.append(f"{y1 % 100:02d}{y2 % 100:02d}")
    return tags


def _parse_american_odds(value: str | None) -> int | None:
    if not value or value.strip() == "":
        return None
    try:
        decimal = float(value)
    except ValueError:
        return None
    if decimal <= 1.0:
        return None
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    return int(round(-100 / (decimal - 1.0)))


def _team_abbr(league: str, label: str) -> str | None:
    mapping = TEAM_TO_ABBR.get(league, {})
    if label in mapping:
        return mapping[label]
    # loose match on prefix
    for key, abbr in mapping.items():
        if label.startswith(key) or key.startswith(label):
            return abbr
    return None


def _fetch_csv(league_code: str, season_tag: str) -> str | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{league_code}_{season_tag}.csv"
    if cache_path.is_file():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = f"{BASE_URL}/{season_tag}/{league_code}.csv"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip() or "HomeTeam" not in text:
        return None
    cache_path.write_text(text, encoding="utf-8")
    return text


@lru_cache(maxsize=16)
def load_football_data_uk_games(league: str) -> list[dict[str, Any]]:
    """
    Completed games from football-data.co.uk for major European leagues.

    Returns dicts with home/away keys, scores, ISO date, and optional closing odds.
    """
    league = league.lower()
    code = LEAGUE_CODES.get(league)
    if not code:
        return []

    rows: list[dict[str, Any]] = []
    for season_tag in _season_tags():
        raw = _fetch_csv(code, season_tag)
        if not raw:
            continue
        reader = csv.DictReader(io.StringIO(raw))
        for record in reader:
            home_label = (record.get("HomeTeam") or "").strip()
            away_label = (record.get("AwayTeam") or "").strip()
            if not home_label or not away_label:
                continue
            home_key = _team_abbr(league, home_label)
            away_key = _team_abbr(league, away_label)
            if not home_key or not away_key:
                continue
            try:
                home_goals = int(record.get("FTHG") or "")
                away_goals = int(record.get("FTAG") or "")
            except ValueError:
                continue
            date_raw = (record.get("Date") or "").strip()
            if not date_raw:
                continue
            # football-data dates: DD/MM/YY or DD/MM/YYYY
            for fmt in ("%d/%m/%y", "%d/%m/%Y"):
                try:
                    parsed = datetime.strptime(date_raw, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    parsed = None
            if parsed is None:
                continue
            odds_home = _parse_american_odds(
                record.get("PSH") or record.get("B365H") or record.get("AvgH")
            )
            odds_draw = _parse_american_odds(
                record.get("PSD") or record.get("B365D") or record.get("AvgD")
            )
            odds_away = _parse_american_odds(
                record.get("PSA") or record.get("B365A") or record.get("AvgA")
            )
            rows.append(
                {
                    "date": parsed.strftime("%Y-%m-%d"),
                    "home_key": home_key,
                    "away_key": away_key,
                    "home_name": home_label,
                    "away_name": away_label,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_odds": odds_home,
                    "draw_odds": odds_draw,
                    "away_odds": odds_away,
                    "source": "football-data.co.uk",
                }
            )
    rows.sort(key=lambda item: item["date"])
    return rows
