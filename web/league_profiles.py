"""League metadata, ESPN paths, and algorithm profile mapping."""

from __future__ import annotations

from typing import TypedDict

# Algo V2 curves exist for nba, nhl, mlb only — map new sports to closest profile.
ALGO_PROFILE: dict[str, str] = {
    "nba": "nba",
    "wnba": "nba",
    "cbb": "nba",
    "nfl": "nba",
    "cfb": "nba",
    "nhl": "nhl",
    "mlb": "mlb",
    "mls": "nhl",
    "epl": "nhl",
    "laliga": "nhl",
    "bundesliga": "nhl",
    "seriea": "nhl",
    "ligue1": "nhl",
    "worldcup": "nhl",
    "fifa_friendlies": "nhl",
    "concacaf_wcq": "nhl",
    "concacaf_gold": "nhl",
    "concacaf_nations": "nhl",
    "uefa_euro": "nhl",
    "uefa_nations": "nhl",
    "copa_america": "nhl",
}

NUM_PERIODS: dict[str, int] = {
    "nba": 4,
    "wnba": 4,
    "cbb": 2,
    "nfl": 4,
    "cfb": 4,
    "nhl": 3,
    "mlb": 9,
    "mls": 2,
    "epl": 2,
    "laliga": 2,
    "bundesliga": 2,
    "seriea": 2,
    "ligue1": 2,
    "worldcup": 2,
    "fifa_friendlies": 2,
    "concacaf_wcq": 2,
    "concacaf_gold": 2,
    "concacaf_nations": 2,
    "uefa_euro": 2,
    "uefa_nations": 2,
    "copa_america": 2,
}

DEFAULT_DATES: dict[str, str] = {
    "nba": "4-16-2017",
    "nhl": "4-12-2017",
    "mlb": "10-25-2016",
    "nfl": "1-15-2025",
    "wnba": "9-15-2024",
    "cbb": "3-15-2025",
    "cfb": "12-15-2024",
    "mls": "10-25-2025",
    "epl": "4-15-2025",
    "laliga": "4-15-2025",
    "bundesliga": "4-15-2025",
    "seriea": "4-15-2025",
    "ligue1": "4-15-2025",
    "worldcup": "6-12-2026",
    "fifa_friendlies": "6-12-2026",
    "concacaf_wcq": "6-12-2026",
    "concacaf_gold": "6-12-2026",
    "concacaf_nations": "6-12-2026",
    "uefa_euro": "6-12-2026",
    "uefa_nations": "6-12-2026",
    "copa_america": "6-12-2026",
}

DEMO_SEASONS: dict[str, str] = {
    "nba": "2017",
    "nhl": "2017",
    "mlb": "2016",
}

# Minimum completed games in the current season before the model runs without prior-season data.
# Matches the algo's last-10-games window; prior season is prepended only when below this threshold.
MIN_GAMES_FOR_MODEL = 10

# Minimum league-wide completed games before power ratings + logistic blend can run.
MIN_GAMES_FOR_POWER = 5

# Leagues with very large ESPN rosters — use scoreboard walks instead of per-team schedules.
LARGE_ROSTER_LEAGUES: tuple[str, ...] = ("cbb", "cfb")

# Days of scoreboard history to scan when collecting completed games.
POWER_SCOREBOARD_LOOKBACK: dict[str, int] = {
    "cbb": 150,
    "cfb": 120,
    "fifa_friendlies": 120,
    "default": 365,
}

# Minimum American-odds edge vs model before a bet is recommended or tracked.
MIN_RECOMMENDED_EDGE = 50

# Basketball and American football — recommend spread instead of moneyline.
SPREAD_BET_CATEGORIES: tuple[str, ...] = ("basketball", "football")

DEFAULT_SPREAD_JUICE = -110


class LeagueProfile(TypedDict):
    id: str
    name: str
    sport_path: str
    category: str
    coach_code: str | None


LEAGUE_PROFILES: dict[str, LeagueProfile] = {
    "nba": {
        "id": "nba",
        "name": "NBA",
        "sport_path": "basketball/nba",
        "category": "basketball",
        "coach_code": "NBA",
    },
    "wnba": {
        "id": "wnba",
        "name": "WNBA",
        "sport_path": "basketball/wnba",
        "category": "basketball",
        "coach_code": "WNBA",
    },
    "cbb": {
        "id": "cbb",
        "name": "NCAA Men's Basketball",
        "sport_path": "basketball/mens-college-basketball",
        "category": "basketball",
        "coach_code": "CBB",
    },
    "nhl": {
        "id": "nhl",
        "name": "NHL",
        "sport_path": "hockey/nhl",
        "category": "hockey",
        "coach_code": "NHL",
    },
    "mlb": {
        "id": "mlb",
        "name": "MLB",
        "sport_path": "baseball/mlb",
        "category": "baseball",
        "coach_code": "MLB",
    },
    "nfl": {
        "id": "nfl",
        "name": "NFL",
        "sport_path": "football/nfl",
        "category": "football",
        "coach_code": "NFL",
    },
    "cfb": {
        "id": "cfb",
        "name": "NCAA Football",
        "sport_path": "football/college-football",
        "category": "football",
        "coach_code": "CFB",
    },
    "mls": {
        "id": "mls",
        "name": "MLS",
        "sport_path": "soccer/usa.1",
        "category": "soccer",
        "coach_code": None,
    },
    "epl": {
        "id": "epl",
        "name": "Premier League",
        "sport_path": "soccer/eng.1",
        "category": "soccer",
        "coach_code": None,
    },
    "laliga": {
        "id": "laliga",
        "name": "La Liga",
        "sport_path": "soccer/esp.1",
        "category": "soccer",
        "coach_code": None,
    },
    "bundesliga": {
        "id": "bundesliga",
        "name": "Bundesliga",
        "sport_path": "soccer/ger.1",
        "category": "soccer",
        "coach_code": None,
    },
    "seriea": {
        "id": "seriea",
        "name": "Serie A",
        "sport_path": "soccer/ita.1",
        "category": "soccer",
        "coach_code": None,
    },
    "ligue1": {
        "id": "ligue1",
        "name": "Ligue 1",
        "sport_path": "soccer/fra.1",
        "category": "soccer",
        "coach_code": None,
    },
    "worldcup": {
        "id": "worldcup",
        "name": "FIFA World Cup",
        "sport_path": "soccer/fifa.world",
        "category": "soccer",
        "coach_code": None,
    },
    "fifa_friendlies": {
        "id": "fifa_friendlies",
        "name": "FIFA Friendlies",
        "sport_path": "soccer/fifa.friendly",
        "category": "soccer",
        "coach_code": None,
    },
    "concacaf_wcq": {
        "id": "concacaf_wcq",
        "name": "CONCACAF WC Qualifying",
        "sport_path": "soccer/fifa.worldq.concacaf",
        "category": "soccer",
        "coach_code": None,
    },
    "concacaf_gold": {
        "id": "concacaf_gold",
        "name": "CONCACAF Gold Cup",
        "sport_path": "soccer/concacaf.gold",
        "category": "soccer",
        "coach_code": None,
    },
    "concacaf_nations": {
        "id": "concacaf_nations",
        "name": "CONCACAF Nations League",
        "sport_path": "soccer/concacaf.nations.league",
        "category": "soccer",
        "coach_code": None,
    },
    "uefa_euro": {
        "id": "uefa_euro",
        "name": "UEFA European Championship",
        "sport_path": "soccer/uefa.euro",
        "category": "soccer",
        "coach_code": None,
    },
    "uefa_nations": {
        "id": "uefa_nations",
        "name": "UEFA Nations League",
        "sport_path": "soccer/uefa.nations",
        "category": "soccer",
        "coach_code": None,
    },
    "copa_america": {
        "id": "copa_america",
        "name": "Copa América",
        "sport_path": "soccer/conmebol.america",
        "category": "soccer",
        "coach_code": None,
    },
}

INTERNATIONAL_SOCCER_LEAGUES: tuple[str, ...] = (
    "worldcup",
    "fifa_friendlies",
    "concacaf_wcq",
    "concacaf_gold",
    "concacaf_nations",
    "uefa_euro",
    "uefa_nations",
    "copa_america",
)

# Sparse tournaments can backfill recent form from FIFA friendlies on the same ESPN team id.
INTERNATIONAL_TOURNAMENT_LEAGUES: tuple[str, ...] = tuple(
    league for league in INTERNATIONAL_SOCCER_LEAGUES if league != "fifa_friendlies"
)
FRIENDLIES_SUPPLEMENT_LEAGUE = "fifa_friendlies"

SUPPORTED_LEAGUES: tuple[str, ...] = tuple(LEAGUE_PROFILES.keys())
SOCCER_LEAGUES: tuple[str, ...] = tuple(
    key for key, profile in LEAGUE_PROFILES.items() if profile["category"] == "soccer"
)

# Historical draw rate baseline (%), adjusted by model closeness in bet_advisor.
SOCCER_DRAW_BASE: dict[str, float] = {
    "default": 26.0,
    "seriea": 28.5,
    "ligue1": 27.0,
    "bundesliga": 24.5,
    "epl": 25.5,
    "laliga": 26.5,
    "mls": 23.0,
    "worldcup": 25.0,
    "fifa_friendlies": 27.0,
    "uefa_euro": 24.0,
    "copa_america": 25.0,
}


def is_soccer_league(league: str) -> bool:
    return league.lower() in SOCCER_LEAGUES


def uses_spread_bets(league: str) -> bool:
    league = league.lower()
    profile = LEAGUE_PROFILES.get(league)
    if not profile:
        return False
    return profile["category"] in SPREAD_BET_CATEGORIES


def get_algo_league(league: str) -> str:
    league = league.lower()
    return ALGO_PROFILE.get(league, league)


def get_league_profile(league: str) -> LeagueProfile:
    league = league.lower()
    if league not in LEAGUE_PROFILES:
        raise ValueError(f"Unsupported league: {league}")
    return LEAGUE_PROFILES[league]


def list_leagues_metadata() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for league_id, profile in LEAGUE_PROFILES.items():
        algo = get_algo_league(league_id)
        rows.append(
            {
                "id": league_id,
                "name": profile["name"],
                "category": profile["category"],
                "description": (
                    f"Live ESPN data · Unified model (Algo V2 + power ratings, {algo.upper()} profile)"
                ),
            }
        )
    return rows
