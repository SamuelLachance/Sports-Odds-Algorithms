"""NBA home-market coordinates for travel-distance features.

Keyed by canonical franchise key (see web/nba_v2/data.py). Relocations use
the market for the era via season-aware lookup.
"""

from __future__ import annotations

# franchise -> list of (first_season_ending_year, lat, lon) sorted ascending
FRANCHISE_MARKETS: dict[str, list[tuple[int, float, float]]] = {
    "atl": [(1949, 33.757, -84.396)],
    "bos": [(1946, 42.366, -71.062)],
    "bkn": [(1976, 40.728, -73.958), (2012, 40.683, -73.975)],  # NJ -> Brooklyn
    "cha": [(2005, 35.225, -80.839)],
    "chi": [(1966, 41.881, -87.674)],
    "cle": [(1970, 41.496, -81.688)],
    "dal": [(1980, 32.790, -96.810)],
    "den": [(1976, 39.749, -105.008)],
    "det": [(1957, 42.341, -83.055)],
    "gs": [(1962, 37.768, -122.388)],
    "hou": [(1971, 29.751, -95.362)],
    "ind": [(1976, 39.764, -86.155)],
    "lac": [(1984, 34.043, -118.267)],
    "lal": [(1960, 34.043, -118.267)],
    "mem": [(1995, 49.280, -123.110), (2001, 35.138, -90.051)],  # Vancouver -> Memphis
    "mia": [(1988, 25.781, -80.187)],
    "mil": [(1968, 43.045, -87.917)],
    "min": [(1989, 44.979, -93.276)],
    "no": [(2002, 29.949, -90.082), (2005, 35.225, -80.839), (2007, 29.949, -90.082)],
    "ny": [(1946, 40.751, -73.993)],
    "okc": [(1967, 47.622, -122.354), (2008, 35.463, -97.515)],  # Seattle -> OKC
    "orl": [(1989, 28.539, -81.384)],
    "phi": [(1963, 39.901, -75.172)],
    "phx": [(1968, 33.446, -112.071)],
    "por": [(1970, 45.532, -122.667)],
    "sac": [(1985, 38.580, -121.500)],
    "sa": [(1976, 29.427, -98.437)],
    "tor": [(1995, 43.643, -79.379)],
    "uta": [(1979, 40.768, -111.901)],
    "wsh": [(1973, 38.898, -77.021)],
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
