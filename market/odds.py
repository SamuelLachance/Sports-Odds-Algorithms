"""Live MLB moneyline odds from The Odds API -> per-game consensus, for the EV edge.

SERVER-SIDE ONLY. The API key is read from the ODDS_API_KEY env var (a GitHub Actions
secret in CI); it must never reach the committed repo or the public client bundle.
Odds are used purely to compare the market to the model's OUTPUT (the EV edge badge);
the model's probability stays market-blind, so market-independence is preserved.

We take a MEDIAN consensus across US books, and only ever consider PRE-GAME games
(commence_time in the future) -- The Odds API serves live in-game prices for games
already underway, which must not enter the pre-game EV.
"""

from __future__ import annotations

import json
import os
import statistics
import urllib.request
from datetime import datetime, timezone

from mlbwp.live import ET     # importing mlbwp is fine; the guard only forbids the reverse

ODDS_URL = ("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
            "?regions=us&markets=h2h&oddsFormat=decimal&apiKey={key}")

# The Odds API full team name -> Retrosheet code (matches the board's team keys).
ODDS_TEAM_TO_RETRO = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHN",
    "Chicago White Sox": "CHA", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCA", "Los Angeles Angels": "ANA", "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYN", "New York Yankees": "NYA", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDN", "San Francisco Giants": "SFN",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "SLN", "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS",
    "Oakland Athletics": "ATH",
}


def _et_date(iso_utc: str) -> str:
    return datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(ET).date().isoformat()


def fetch_consensus(api_key: str | None = None, now: datetime | None = None) -> list[dict]:
    """Pre-game MLB games with median-consensus decimal prices across US books.

    Returns list of {id, commence, date(ET), home, away (retro), home_dec, away_dec,
    n_books}. Only games whose commence_time is in the future are included. Returns []
    on any failure (missing key, network, quota) so the build degrades gracefully.
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        return []
    now = now or datetime.now(timezone.utc)
    try:
        req = urllib.request.Request(ODDS_URL.format(key=key), headers={"User-Agent": "mlbwp"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception:  # noqa: BLE001 — never let odds break the board build
        return []
    out = []
    for g in data:
        home = ODDS_TEAM_TO_RETRO.get(g.get("home_team"))
        away = ODDS_TEAM_TO_RETRO.get(g.get("away_team"))
        commence = g.get("commence_time")
        if not (home and away and commence and g.get("id")):   # id anchors the opening cache
            continue
        try:
            if datetime.fromisoformat(commence.replace("Z", "+00:00")) <= now:
                continue                      # already started -> live prices, skip
        except ValueError:
            continue
        hp, ap = [], []
        for bk in g.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                px = {o.get("name"): o.get("price") for o in m.get("outcomes", [])}
                dh, da = px.get(g["home_team"]), px.get(g["away_team"])
                if isinstance(dh, (int, float)) and isinstance(da, (int, float)) and dh > 1 and da > 1:
                    hp.append(dh); ap.append(da)
        if len(hp) < 3:                       # need a real consensus
            continue
        out.append({
            "id": g.get("id"), "commence": commence, "date": _et_date(commence),
            "home": home, "away": away,
            "home_dec": round(statistics.median(hp), 4),
            "away_dec": round(statistics.median(ap), 4),
            "n_books": len(hp),
        })
    return out


def value_side(prob_home: float, home_dec: float, away_dec: float) -> dict:
    """The positive-value side per 1u: EV = p*dec - 1. Returns side/ev/dec."""
    ev_h = prob_home * home_dec - 1.0
    ev_a = (1.0 - prob_home) * away_dec - 1.0
    if ev_h >= ev_a:
        return {"side": "home", "ev": ev_h, "dec": home_dec}
    return {"side": "away", "ev": ev_a, "dec": away_dec}
