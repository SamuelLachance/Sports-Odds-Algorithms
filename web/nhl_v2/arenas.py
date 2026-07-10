"""NHL arena coordinates (lat, lon) by canonical team abbreviation.

Used for travel-distance features. Historical franchises included so replays
of older seasons resolve (ATL Thrashers, ARI/PHX Coyotes eras).
"""

from __future__ import annotations

ARENA_COORDS: dict[str, tuple[float, float]] = {
    "ANA": (33.8078, -117.8766),   # Honda Center
    "ARI": (33.5319, -112.2611),   # Gila River / Mullett era approximated (Phoenix)
    "ATL": (33.7573, -84.3963),    # Philips Arena (Thrashers)
    "BOS": (42.3662, -71.0621),    # TD Garden
    "BUF": (42.8750, -78.8764),    # KeyBank Center
    "CAR": (35.8033, -78.7219),    # Lenovo Center
    "CBJ": (39.9695, -83.0060),    # Nationwide Arena
    "CGY": (51.0374, -114.0519),   # Scotiabank Saddledome
    "CHI": (41.8807, -87.6742),    # United Center
    "COL": (39.7486, -105.0076),   # Ball Arena
    "DAL": (32.7905, -96.8103),    # American Airlines Center
    "DET": (42.3411, -83.0553),    # Little Caesars Arena
    "EDM": (53.5469, -113.4979),   # Rogers Place
    "FLA": (26.1585, -80.3255),    # Amerant Bank Arena
    "LAK": (34.0430, -118.2673),   # Crypto.com Arena
    "MIN": (44.9447, -93.1011),    # Xcel Energy Center
    "MTL": (45.4960, -73.5693),    # Bell Centre
    "NJD": (40.7336, -74.1711),    # Prudential Center
    "NSH": (36.1593, -86.7785),    # Bridgestone Arena
    "NYI": (40.7126, -73.5904),    # UBS Arena (Nassau/Barclays eras nearby)
    "NYR": (40.7505, -73.9934),    # Madison Square Garden
    "OTT": (45.2969, -75.9273),    # Canadian Tire Centre
    "PHI": (39.9012, -75.1720),    # Wells Fargo Center
    "PIT": (40.4395, -79.9896),    # PPG Paints Arena
    "SEA": (47.6221, -122.3541),   # Climate Pledge Arena
    "SJS": (37.3328, -121.9012),   # SAP Center
    "STL": (38.6268, -90.2027),    # Enterprise Center
    "TBL": (27.9427, -82.4518),    # Amalie Arena
    "TOR": (43.6435, -79.3791),    # Scotiabank Arena
    "UTA": (40.7683, -111.9011),   # Delta Center
    "VAN": (49.2778, -123.1089),   # Rogers Arena
    "VGK": (36.1029, -115.1785),   # T-Mobile Arena
    "WPG": (49.8927, -97.1435),    # Canada Life Centre
    "WSH": (38.8981, -77.0209),    # Capital One Arena
}
