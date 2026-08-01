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


# Why the last fetch returned what it did. `fetch_consensus` degrades to [] on
# every failure so the board build never breaks — but that collapsed FIVE very
# different situations into one indistinguishable empty list, and downstream all
# of them printed "0 games with >=20% EV", which reads as "no edges qualify
# today". A dead API key looked exactly like a quiet market.
#
# Found live on 2026-08-01: the opening-odds cache had not captured a game since
# 2026-07-29 while the board carried 36 pending picks. The edge layer had been
# silently dead for ~3 days, and the edge ledger — the durable evidence stream
# the live-CLV promotion gate depends on — held zero rows.
LAST_FETCH: dict = {"reason": "not_called"}


def _record(reason: str, **extra) -> None:
    LAST_FETCH.clear()
    LAST_FETCH.update(reason=reason, **extra)


def fetch_status() -> str:
    """One line describing the last fetch, for the refresh log."""
    r = LAST_FETCH.get("reason", "not_called")
    if r == "ok":
        return f"ok: {LAST_FETCH['n_used']} priced of {LAST_FETCH['n_raw']} returned"
    if r == "no_api_key":
        return "NO ODDS: ODDS_API_KEY is unset — the edge layer is disabled"
    if r == "fetch_failed":
        return (f"NO ODDS: the request failed ({LAST_FETCH.get('error')}) — "
                f"check the key, the quota and the endpoint")
    if r == "empty_response":
        return "NO ODDS: the API returned zero games — out of quota, or no slate"
    if r == "all_filtered":
        sk = LAST_FETCH.get("skipped", {})
        return (f"NO ODDS USABLE: {LAST_FETCH['n_raw']} games returned but none "
                f"priced ({sk}) — likely a team-name map drift or thin books")
    return f"odds fetch never ran ({r})"


def fetch_consensus(api_key: str | None = None, now: datetime | None = None,
                    url: str = ODDS_URL,
                    team_map: dict[str, str] = ODDS_TEAM_TO_RETRO) -> list[dict]:
    """Pre-game games with median-consensus decimal prices across US books.

    Defaults to MLB; pass url/team_map for another sport (see market/nfl_edges.py).
    Returns list of {id, commence, date(ET), home, away, home_dec, away_dec,
    n_books}. Only games whose commence_time is in the future are included. Returns []
    on any failure (missing key, network, quota) so the build degrades gracefully.
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        _record("no_api_key")
        return []
    now = now or datetime.now(timezone.utc)
    try:
        req = urllib.request.Request(url.format(key=key), headers={"User-Agent": "mlbwp"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001 — never let odds break the board build
        _record("fetch_failed", error=f"{type(exc).__name__}: {str(exc)[:120]}")
        return []
    skipped = {"unmapped_team": 0, "already_started": 0, "too_few_books": 0}
    out = []
    for g in data:
        home = team_map.get(g.get("home_team"))
        away = team_map.get(g.get("away_team"))
        commence = g.get("commence_time")
        if not (home and away and commence and g.get("id")):   # id anchors the opening cache
            skipped["unmapped_team"] += 1
            continue
        try:
            if datetime.fromisoformat(commence.replace("Z", "+00:00")) <= now:
                skipped["already_started"] += 1
                continue                      # already started -> live prices, skip
        except ValueError:
            skipped["unmapped_team"] += 1
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
            skipped["too_few_books"] += 1
            continue
        out.append({
            "id": g.get("id"), "commence": commence, "date": _et_date(commence),
            "home": home, "away": away,
            "home_dec": round(statistics.median(hp), 4),
            "away_dec": round(statistics.median(ap), 4),
            "n_books": len(hp),
        })
    _record("ok" if out else ("empty_response" if not data else "all_filtered"),
            n_raw=len(data), n_used=len(out), skipped=skipped)
    return out


def value_side(prob_home: float, home_dec: float, away_dec: float) -> dict:
    """The positive-value side per 1u: EV = p*dec - 1. Returns side/ev/dec."""
    ev_h = prob_home * home_dec - 1.0
    ev_a = (1.0 - prob_home) * away_dec - 1.0
    if ev_h >= ev_a:
        return {"side": "home", "ev": ev_h, "dec": home_dec}
    return {"side": "away", "ev": ev_a, "dec": away_dec}
