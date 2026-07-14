"""NFL team timezone / travel / altitude helpers."""

from __future__ import annotations

import math

# Approximate UTC offsets in standard time (ignore DST — relative diffs matter).
TEAM_TZ_OFFSET: dict[str, float] = {
    "ari": -7.0,
    "atl": -5.0,
    "bal": -5.0,
    "buf": -5.0,
    "car": -5.0,
    "chi": -6.0,
    "cin": -5.0,
    "cle": -5.0,
    "dal": -6.0,
    "den": -7.0,
    "det": -5.0,
    "gb": -6.0,
    "hou": -6.0,
    "ind": -5.0,
    "jax": -5.0,
    "jac": -5.0,
    "kc": -6.0,
    "la": -8.0,
    "lac": -8.0,
    "lar": -8.0,
    "lv": -8.0,
    "mia": -5.0,
    "min": -6.0,
    "ne": -5.0,
    "no": -6.0,
    "nyg": -5.0,
    "nyj": -5.0,
    "oak": -8.0,
    "phi": -5.0,
    "pit": -5.0,
    "sd": -8.0,
    "sea": -8.0,
    "sf": -8.0,
    "stl": -6.0,
    "tb": -5.0,
    "ten": -6.0,
    "was": -5.0,
    "wsh": -5.0,
}

# Stadium approximate lat/lon (public geography).
TEAM_COORDS: dict[str, tuple[float, float]] = {
    "ari": (33.53, -112.26),
    "atl": (33.76, -84.40),
    "bal": (39.28, -76.62),
    "buf": (42.77, -78.79),
    "car": (35.23, -80.85),
    "chi": (41.86, -87.62),
    "cin": (39.10, -84.51),
    "cle": (41.51, -81.70),
    "dal": (32.75, -97.09),
    "den": (39.74, -105.02),
    "det": (42.34, -83.05),
    "gb": (44.50, -88.06),
    "hou": (29.68, -95.41),
    "ind": (39.76, -86.16),
    "jax": (30.32, -81.64),
    "jac": (30.32, -81.64),
    "kc": (39.05, -94.48),
    "la": (33.95, -118.34),
    "lar": (33.95, -118.34),
    "lac": (33.95, -118.34),
    "lv": (36.09, -115.18),
    "oak": (37.75, -122.20),
    "mia": (25.96, -80.24),
    "min": (44.97, -93.26),
    "ne": (42.09, -71.26),
    "no": (29.95, -90.08),
    "nyg": (40.81, -74.07),
    "nyj": (40.81, -74.07),
    "phi": (39.90, -75.17),
    "pit": (40.45, -80.02),
    "sd": (32.78, -117.12),
    "sea": (47.60, -122.33),
    "sf": (37.40, -121.97),
    "stl": (38.63, -90.19),
    "tb": (27.98, -82.50),
    "ten": (36.17, -86.77),
    "was": (38.91, -76.86),
    "wsh": (38.91, -76.86),
}

# Fraction of ~1 mile elevation (DEN ~5280 ft).
TEAM_ALTITUDE: dict[str, float] = {
    "den": 1.0,
}


def team_tz(team: str) -> float:
    key = str(team or "").lower().strip()
    return float(TEAM_TZ_OFFSET.get(key, -5.0))


def timezone_diff(home: str, away: str) -> float:
    """Away team hours west of home stadium (positive => westbound travel)."""
    return team_tz(home) - team_tz(away)


def travel_km(from_team: str, to_team: str) -> float:
    """Great-circle km between stadiums (0 when unknown)."""
    a = TEAM_COORDS.get(str(from_team or "").lower().strip())
    b = TEAM_COORDS.get(str(to_team or "").lower().strip())
    if not a or not b:
        return 0.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    h = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(h)))


def altitude(team: str) -> float:
    return float(TEAM_ALTITUDE.get(str(team or "").lower().strip(), 0.0))
