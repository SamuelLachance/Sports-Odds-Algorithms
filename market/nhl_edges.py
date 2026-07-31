"""Attach the >=20%-EV value edge to the NHL schedule in site/data/nhl.json.

Same POST-PROCESS contract as the MLB/NFL edge layers: compares the live market
to the market-blind model's probability, never changes a model number. Only games
inside the live window (puck-drop within 7 days, unplayed) are eligible. Opening
consensus frozen on first sighting; badge shows while EV vs opening >= 20% and the
current price still clears. Offseason: no NHL h2h events in range -> clean no-op.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market import odds

PROJECT = Path(__file__).resolve().parents[1]
NHL_JSON = PROJECT / "site" / "data" / "nhl.json"
OPENING = PROJECT / "data" / "nhl_opening_odds_2026.json"
EV_THRESHOLD = 0.20
LIVE_WINDOW_DAYS = 7

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
                    api_key: str | None = None) -> int:
    if not payload_path.is_file():
        return 0
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    sched = payload.get("schedule") or []
    if not sched:
        return 0
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    today = now_dt.astimezone(odds.ET).date()
    horizon = (today + timedelta(days=LIVE_WINDOW_DAYS)).isoformat()
    eligible = [g for g in sched
                if g.get("hs") is None and today.isoformat() <= g["d"] <= horizon]
    for g in sched:
        g.pop("value", None)
        g.pop("mkt", None)
    if not eligible:
        payload["value_updated"] = now
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    live = odds.fetch_consensus(api_key, url=NHL_ODDS_URL, team_map=ODDS_TEAM_TO_NHL)
    cache = json.loads(opening_path.read_text()) if opening_path.is_file() else {}
    idx = defaultdict(list)
    for g in live:
        idx[(g["date"], g["home"], g["away"])].append(g)
        cache.setdefault(g["id"], {"home_dec": g["home_dec"], "away_dec": g["away_dec"],
                                   "date": g["date"], "captured": now})
    n = 0
    for c in eligible:
        cand = idx.get((c["d"], c["home"], c["away"]))
        if not cand:
            continue
        g = cand[0]
        # Unconditional price feed for market/edge_ledger.py (see market/edges.py):
        # written BEFORE the EV gate, so a badge that vanishes when the model prob
        # moves cannot truncate a recorded bet's price path short of the close.
        c["mkt"] = {"home_dec": round(g["home_dec"], 3),
                    "away_dec": round(g["away_dec"], 3),
                    "books": g["n_books"], "ts": now}
        op = cache.get(g["id"])
        if not op:
            continue
        p = c["hp"]                                # live-model home prob
        oe = odds.value_side(p, op["home_dec"], op["away_dec"])
        side, ev_open = oe["side"], oe["ev"]
        if ev_open < EV_THRESHOLD:
            continue
        cur_dec = g["home_dec"] if side == "home" else g["away_dec"]
        ev_cur = (p if side == "home" else 1 - p) * cur_dec - 1.0
        c["value"] = {
            "side": side, "team": c["home"] if side == "home" else c["away"],
            "ev_open": round(ev_open, 4), "ev_cur": round(ev_cur, 4),
            "open_dec": round(op["home_dec"] if side == "home" else op["away_dec"], 3),
            "cur_dec": round(cur_dec, 3), "available": ev_cur > 0, "books": g["n_books"],
        }
        n += 1
    payload["value_updated"] = now
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    cutoff = (now_dt.date() - timedelta(days=2)).isoformat()
    cache = {k: v for k, v in cache.items() if v.get("date", "9999") >= cutoff}
    opening_path.write_text(json.dumps(cache), encoding="utf-8")
    return n
