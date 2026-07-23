"""Attach the >=15%-EV value edge to the NFL schedule in site/data/nfl.json.

Same contract as market/edges.py (MLB): a POST-PROCESS on the market-blind model's
output — never changes a model probability, only compares the live market to it.
Only games inside the LIVE-MODEL window (kickoff within the next 7 days, unplayed)
are eligible, matching the serving policy: model probs under a week out, Monte
Carlo beyond. Opening consensus is frozen on first sighting; the badge shows only
while EV vs that opening >= 15% and the current price still clears (ev_cur > 0).
Preseason: The Odds API has no NFL h2h events in range -> no-op, degrades clean.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market import odds

PROJECT = Path(__file__).resolve().parents[1]
NFL_JSON = PROJECT / "site" / "data" / "nfl.json"
OPENING = PROJECT / "data" / "nfl_opening_odds_2026.json"
EV_THRESHOLD = 0.15
LIVE_WINDOW_DAYS = 7

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
        g.pop("value", None)                       # stale badges cleared every run
    if not eligible:
        payload["value_updated"] = now
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    live = odds.fetch_consensus(api_key, url=NFL_ODDS_URL, team_map=ODDS_TEAM_TO_NFL)
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
        op = cache.get(g["id"])
        if not op:
            continue
        p = c["ph"]                                # live-model prob (<=7 days out)
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
