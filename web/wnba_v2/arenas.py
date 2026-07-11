"""WNBA home-market coordinates + altitudes for travel/fatigue features.

Keyed by canonical franchise key (see web/wnba_v2/data.py). Relocations use
the market for the era; since Elo state is franchise-continuous but travel is
computed from the current home market, we key coordinates by (franchise, era)
via the season-aware lookup below.
"""

from __future__ import annotations

# franchise -> list of (first_season, lat, lon, altitude_m) sorted ascending
FRANCHISE_MARKETS: dict[str, list[tuple[int, float, float, float]]] = {
    "atl": [(2008, 33.757, -84.396, 320.0)],
    "chi": [(2006, 41.882, -87.674, 182.0)],
    # Orlando -> Uncasville
    "con": [(1999, 28.539, -81.384, 25.0), (2003, 41.491, -72.091, 38.0)],
    # Detroit -> Tulsa -> Arlington
    "dal": [
        (1998, 42.331, -83.046, 183.0),
        (2010, 36.154, -95.993, 194.0),
        (2016, 32.731, -97.108, 184.0),
    ],
    "gsv": [(2025, 37.768, -122.388, 5.0)],
    "ind": [(2000, 39.764, -86.156, 218.0)],
    # Salt Lake City -> San Antonio -> Las Vegas
    "lva": [
        (1997, 40.769, -111.901, 1288.0),
        (2003, 29.427, -98.437, 198.0),
        (2018, 36.091, -115.176, 610.0),
    ],
    "las": [(1997, 34.043, -118.267, 89.0)],
    "min": [(1999, 44.979, -93.276, 253.0)],
    "nyl": [(1997, 40.751, -73.994, 10.0), (2021, 40.683, -73.976, 10.0)],
    "phx": [(1997, 33.446, -112.071, 331.0)],
    "sea": [(2000, 47.622, -122.354, 54.0)],
    "was": [(1998, 38.898, -77.021, 8.0)],
    "tor": [(2026, 43.643, -79.379, 76.0)],
    "por": [(2026, 45.532, -122.667, 15.0)],
    # defunct franchises
    "hou": [(1997, 29.751, -95.362, 12.0)],
    "sac": [(1997, 38.649, -121.518, 8.0)],
    "cle": [(1997, 41.496, -81.688, 199.0)],
    "cha": [(1997, 35.225, -80.839, 229.0)],
    "mia": [(2000, 25.781, -80.187, 2.0)],
    "prt": [(2000, 45.532, -122.667, 15.0)],
    "orl": [(1999, 28.539, -81.384, 25.0)],
    "utah": [(1997, 40.769, -111.901, 1288.0)],
    "det": [(1998, 42.331, -83.046, 183.0)],
    "tul": [(2010, 36.154, -95.993, 194.0)],
    "sas": [(2003, 29.427, -98.437, 198.0)],
}


def _market_row(franchise: str, season: int) -> tuple[int, float, float, float] | None:
    markets = FRANCHISE_MARKETS.get(franchise)
    if not markets:
        return None
    best: tuple[int, float, float, float] | None = None
    for row in markets:
        if season >= row[0]:
            best = row
    return best or markets[0]


def market_coords(franchise: str, season: int) -> tuple[float, float] | None:
    row = _market_row(franchise, season)
    if row is None:
        return None
    return row[1], row[2]


def market_altitude_m(franchise: str, season: int) -> float | None:
    row = _market_row(franchise, season)
    if row is None:
        return None
    return row[3]
