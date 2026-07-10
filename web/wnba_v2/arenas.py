"""WNBA home-market coordinates for travel-distance features.

Keyed by canonical franchise key (see web/wnba_v2/data.py). Relocations use
the market for the era; since Elo state is franchise-continuous but travel is
computed from the current home market, we key coordinates by (franchise, era)
via the season-aware lookup below.
"""

from __future__ import annotations

# franchise -> list of (first_season, lat, lon) sorted ascending
FRANCHISE_MARKETS: dict[str, list[tuple[int, float, float]]] = {
    "atl": [(2008, 33.757, -84.396)],
    "chi": [(2006, 41.882, -87.674)],
    "con": [(1999, 28.539, -81.384), (2003, 41.491, -72.091)],  # Orlando -> Uncasville
    "dal": [(1998, 42.331, -83.046), (2010, 36.154, -95.993), (2016, 32.731, -97.108)],
    "gsv": [(2025, 37.768, -122.388)],
    "ind": [(2000, 39.764, -86.156)],
    "lva": [(1997, 40.769, -111.901), (2003, 29.427, -98.437), (2018, 36.091, -115.176)],
    "las": [(1997, 34.043, -118.267)],
    "min": [(1999, 44.979, -93.276)],
    "nyl": [(1997, 40.751, -73.994), (2021, 40.683, -73.976)],
    "phx": [(1997, 33.446, -112.071)],
    "sea": [(2000, 47.622, -122.354)],
    "was": [(1998, 38.898, -77.021)],
    "tor": [(2026, 43.643, -79.379)],
    "por": [(2026, 45.532, -122.667)],
    # defunct franchises
    "hou": [(1997, 29.751, -95.362)],
    "sac": [(1997, 38.649, -121.518)],
    "cle": [(1997, 41.496, -81.688)],
    "cha": [(1997, 35.225, -80.839)],
    "mia": [(2000, 25.781, -80.187)],
    "prt": [(2000, 45.532, -122.667)],
    "orl": [(1999, 28.539, -81.384)],
    "utah": [(1997, 40.769, -111.901)],
    "det": [(1998, 42.331, -83.046)],
    "tul": [(2010, 36.154, -95.993)],
    "sas": [(2003, 29.427, -98.437)],
}


def market_coords(franchise: str, season: int) -> tuple[float, float] | None:
    markets = FRANCHISE_MARKETS.get(franchise)
    if not markets:
        return None
    best: tuple[float, float] | None = None
    for first_season, lat, lon in markets:
        if season >= first_season:
            best = (lat, lon)
    return best or (markets[0][1], markets[0][2])
