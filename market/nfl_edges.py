"""Attach the EV-threshold value edge (market/__init__.py) to the NFL schedule in site/data/nfl.json.

Same contract as market/edges.py (MLB): a POST-PROCESS on the market-blind model's
output — never changes a model probability, only compares the live market to it.
Only games inside the LIVE-MODEL window (kickoff within the next 7 days, unplayed,
not yet kicked off) are eligible, matching the serving policy: model probs under a
week out, Monte Carlo beyond. Opening consensus is frozen on first sighting; the
badge shows while EV vs that opening clears market.EV_THRESHOLD (0.08 since
2026-08-11) and is marked available while the current price still clears
(ev_cur > 0). Preseason: The Odds API has no NFL h2h events in range -> no-op.

FRESHNESS GUARD (market/sched_edges.py): the NFL model is rebuilt by a LOCAL weekly
chain, so the payload can be days or weeks older than the price it is compared
with. No badge is attached from a build older than 8 days, from a forecast made
more than 7 days before kickoff (EARLY by the pick policy's own tier line), from a
row the serve stamped EARLY (sched_edges.GATE_EARLY), or for a team whose last game
has kicked off but whose result the model build has not absorbed — a final that
phase0/nfl_results_attach.py wrote into the payload after the build does not
count. `mkt` keeps flowing either way (the edge ledger's closing-line feed), and
payload.value_status / schedule[].value_blocked say why a badge is held.

Within the PROJECTED tier the deficit is disclosed (documents/pick_policy.md,
2026-07-31): every NFL badge comes from one weekly build; news after that build —
including Friday's final injury report when the chain ran earlier in the week — is
not in it. That deficit rides on the badge as `value.note`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from market import odds, sched_edges   # noqa: F401  (odds: the fetch, patched in tests)

PROJECT = Path(__file__).resolve().parents[1]
NFL_JSON = PROJECT / "site" / "data" / "nfl.json"
OPENING = PROJECT / "data" / "nfl_opening_odds_2026.json"
from market import EV_THRESHOLD   # noqa: E402,F401  single source (market/__init__.py)
LIVE_WINDOW_DAYS = sched_edges.LIVE_WINDOW_DAYS
TIER = sched_edges.DEFAULT_TIER["nfl"]   # PROJECTED: live model inside 7 days (policy)
DISCLOSURE = ("PROJECTED tier: priced from the weekly model build stamped on the "
              "badge; injury and depth-chart news after that build is not in it.")

NFL_ODDS_URL = ("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
                "?regions=us&markets=h2h&oddsFormat=decimal&apiKey={key}")

ODDS_TEAM_TO_NFL = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def attach_and_save(payload_path: Path = NFL_JSON, opening_path: Path = OPENING,
                    api_key: str | None = None, now: datetime | None = None) -> int:
    """Rewrite the NFL payload's badges; returns how many were attached."""
    return sched_edges.attach("nfl", payload_path, opening_path, api_key,
                              url=NFL_ODDS_URL, team_map=ODDS_TEAM_TO_NFL,
                              p_key="ph", tier=TIER, note=DISCLOSURE, now=now)
