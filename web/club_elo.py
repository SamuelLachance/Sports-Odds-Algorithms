"""ClubElo public ratings API for club soccer leagues."""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from typing import Any

USER_AGENT = "Sports-Odds-Algorithms/2.0"
CLUB_ELO_API = "http://api.clubelo.com"

# ESPN abbr -> ClubElo team slug (http://clubelo.com/{slug})
CLUB_ELO_SLUGS: dict[str, dict[str, str]] = {
    "epl": {
        "ars": "Arsenal",
        "avl": "AstonVilla",
        "bou": "Bournemouth",
        "bre": "Brentford",
        "bha": "Brighton",
        "bur": "Burnley",
        "che": "Chelsea",
        "cry": "CrystalPalace",
        "eve": "Everton",
        "ful": "Fulham",
        "ips": "Ipswich",
        "lee": "Leeds",
        "lei": "Leicester",
        "liv": "Liverpool",
        "lut": "Luton",
        "mnc": "ManCity",
        "man": "ManUnited",
        "new": "Newcastle",
        "nfo": "Forest",
        "shu": "SheffieldUnited",
        "sou": "Southampton",
        "sun": "Sunderland",
        "tot": "Tottenham",
        "whu": "WestHam",
        "wol": "Wolves",
    },
    "bundesliga": {
        "bay": "Bayern",
        "dor": "Dortmund",
        "rbl": "RBLeipzig",
        "b04": "Leverkusen",
        "sge": "Frankfurt",
        "scf": "Freiburg",
        "tsg": "Hoffenheim",
        "wob": "Wolfsburg",
        "mai": "Mainz",
        "fca": "Augsburg",
        "fcu": "UnionBerlin",
        "svw": "Werder",
        "boc": "Bochum",
        "fch": "Heidenheim",
        "koe": "Koeln",
        "bmg": "Gladbach",
        "vfb": "Stuttgart",
        "s04": "Schalke",
    },
    "laliga": {
        "bar": "Barcelona",
        "rma": "RealMadrid",
        "atm": "Athletic",
        "atl": "Atletico",
        "sev": "Sevilla",
        "vil": "Villarreal",
        "bet": "Betis",
        "soc": "Sociedad",
        "val": "Valencia",
        "cel": "Celta",
        "get": "Getafe",
        "gir": "Girona",
        "mal": "Mallorca",
        "osa": "Osasuna",
        "ray": "Rayo",
        "ala": "Alaves",
        "cad": "Cadiz",
        "gra": "Granada",
        "las": "LasPalmas",
        "leg": "Leganes",
        "esp": "Espanyol",
        "vll": "Valladolid",
    },
    "seriea": {
        "juv": "Juventus",
        "int": "Inter",
        "mil": "Milan",
        "nap": "Napoli",
        "rom": "Roma",
        "laz": "Lazio",
        "ata": "Atalanta",
        "fio": "Fiorentina",
        "bol": "Bologna",
        "tor": "Torino",
        "udi": "Udinese",
        "mon": "Monza",
        "gen": "Genoa",
        "cag": "Cagliari",
        "emp": "Empoli",
        "ver": "Verona",
        "lec": "Lecce",
        "par": "Parma",
        "com": "Como",
        "ven": "Venezia",
    },
    "ligue1": {
        "psg": "PSG",
        "mar": "Marseille",
        "lyo": "Lyon",
        "mon": "Monaco",
        "lil": "Lille",
        "len": "Lens",
        "ren": "Rennes",
        "nic": "Nice",
        "str": "Strasbourg",
        "nan": "Nantes",
        "tou": "Toulouse",
        "bre": "Brest",
        "rei": "Reims",
        "hav": "LeHavre",
        "lor": "Lorient",
        "met": "Metz",
        "aux": "Auxerre",
        "ang": "Angers",
        "sti": "StEtienne",
    },
    "mls": {
        "atl": "Atlanta",
        "aus": "Austin",
        "cha": "Charlotte",
        "chi": "Chicago",
        "cin": "Cincinnati",
        "clb": "Columbus",
        "col": "Colorado",
        "dal": "Dallas",
        "dc": "DCUnited",
        "hou": "Houston",
        "la": "LAGalaxy",
        "laf": "LAFC",
        "mia": "Miami",
        "min": "Minnesota",
        "mtl": "Montreal",
        "nsh": "Nashville",
        "nyc": "NewYork",
        "nyr": "NewYork",
        "orl": "Orlando",
        "phi": "Philadelphia",
        "por": "Portland",
        "rsl": "SaltLake",
        "sj": "SanJose",
        "sea": "Seattle",
        "skc": "KansasCity",
        "stl": "StLouis",
        "tor": "Toronto",
        "van": "Vancouver",
    },
}

CLUB_ELO_LEAGUES = frozenset(CLUB_ELO_SLUGS.keys())


def supports_club_elo(league: str) -> bool:
    return league.lower() in CLUB_ELO_LEAGUES


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8")
            if not raw.strip():
                return None
            return json.loads(raw)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


@lru_cache(maxsize=256)
def fetch_team_elo(club_slug: str) -> float | None:
    """Latest ClubElo rating for a team slug (e.g. Arsenal)."""
    payload = _fetch_json(f"{CLUB_ELO_API}/{club_slug}")
    if not payload or not isinstance(payload, list):
        return None
    for row in reversed(payload):
        if not isinstance(row, dict):
            continue
        elo = row.get("Elo")
        if elo is not None:
            try:
                return float(elo)
            except (TypeError, ValueError):
                continue
    return None


def resolve_club_elo_slug(league: str, abbr: str) -> str | None:
    league = league.lower()
    abbr = abbr.lower()
    return (CLUB_ELO_SLUGS.get(league) or {}).get(abbr)


def get_team_elo_rating(league: str, abbr: str) -> float | None:
    slug = resolve_club_elo_slug(league, abbr)
    if not slug:
        return None
    return fetch_team_elo(slug)


def elo_binary_home_win_pct(home_elo: float, away_elo: float, *, home_adv: float = 65.0) -> float:
    diff = (home_elo + home_adv) - away_elo
    return 100.0 / (1.0 + 10.0 ** (-diff / 400.0))
