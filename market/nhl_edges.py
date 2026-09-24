"""Attach the EV-threshold value edge (market/__init__.py) to the NHL schedule in site/data/nhl.json.

Same POST-PROCESS contract as the MLB/NFL edge layers: compares the live market
to the market-blind model's probability, never changes a model number. Only games
inside the live window (puck drop within 7 days, unplayed, not yet started — the
payload serves start_utc) are eligible. Opening consensus is frozen on first
sighting; the badge shows while EV vs that opening clears market.EV_THRESHOLD
(0.08 since 2026-08-11) and is marked available while the current price still
clears. Offseason: no NHL h2h events in range -> clean no-op.

FRESHNESS GUARD (market/sched_edges.py): no badge from a build older than 36 hours
(refresh.yml re-serves every 4), from a forecast made more than 7 days before puck
drop, or for a team whose last game has started but whose result the model build
has not absorbed yet. `mkt` keeps flowing either way (the edge ledger's closing
feed), and payload.value_status / schedule[].value_blocked say why a badge is held.

EARLY GATE (sched_edges.GATE_EARLY, on): the shipped NHL model has no goalie or
lineup input, so every NHL forecast is EARLY — and an EARLY forecast is not a pick
(documents/pick_policy.md rule 1), let alone an EV badge, which the policy calls a
stronger claim than a pick. Every NHL game in the window is therefore quoted but
not badged (value_blocked "EARLY tier: ...", counted in value_status.n_early), and
the edge ledger stakes nothing on it. The gate lifts per game once a goalie input
ships and the serve stamps a row PROJECTED/CONFIRMED. The deficit behind it is
measured, not assumed: starting goalies are confirmed on game-day morning, and our
own measurement (data/market_projection.json) says that is the information the
market has and we lack. Setting GATE_EARLY = False restores the 2026-07-31
disclosure policy: badges show with `value.tier` = "EARLY" and DISCLOSURE below.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from market import odds, sched_edges   # noqa: F401  (odds: the fetch, patched in tests)

PROJECT = Path(__file__).resolve().parents[1]
NHL_JSON = PROJECT / "site" / "data" / "nhl.json"
OPENING = PROJECT / "data" / "nhl_opening_odds_2026.json"
from market import EV_THRESHOLD   # noqa: E402,F401  single source (market/__init__.py)
LIVE_WINDOW_DAYS = sched_edges.LIVE_WINDOW_DAYS
TIER = sched_edges.DEFAULT_TIER["nhl"]   # EARLY: no goalie, no lineup (pick policy)
# Rides on a badge only if GATE_EARLY is switched off. A badge is staked in units by
# the edge ledger, so it must not call itself "not a pick".
DISCLOSURE = ("EARLY tier: team ratings only. Priced before starting goalies are "
              "confirmed (game-day morning), the information our own measurement "
              "says the market has and this model lacks. A disagreement indicator, "
              "staked in units as a badge.")

NHL_ODDS_URL = ("https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
                "?regions=us&markets=h2h&oddsFormat=decimal&apiKey={key}")

ODDS_TEAM_TO_NHL = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL", "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
    "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA", "St Louis Blues": "STL", "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR", "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
    "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA",
}


def attach_and_save(payload_path: Path = NHL_JSON, opening_path: Path = OPENING,
                    api_key: str | None = None, now: datetime | None = None) -> int:
    """Rewrite the NHL payload's badges; returns how many were attached."""
    return sched_edges.attach("nhl", payload_path, opening_path, api_key,
                              url=NHL_ODDS_URL, team_map=ODDS_TEAM_TO_NHL,
                              p_key="hp", tier=TIER, note=DISCLOSURE, now=now)
