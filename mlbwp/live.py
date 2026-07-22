"""Live MLB slate + current-season results from the MLB Stats API.

Market-free by construction: only schedule, probable pitchers and final scores are
requested — no odds endpoint is ever touched.

Terms note (surfaced on the site): the MLB Stats API is provided for individual,
non-commercial use. Any public deployment of this project must resolve that with
MLB or move to a licensed feed; see the methodology page.
"""

from __future__ import annotations

import json
import time
import urllib.request

BASE = "https://statsapi.mlb.com/api/v1"
UA = "Mozilla/5.0 (mlbwp research)"

# StatsAPI team id -> Retrosheet team code (the model's team key).
TEAM_ID_TO_RETRO = {
    108: "ANA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHN", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KCA", 119: "LAN",
    # Athletics: Retrosheet switched OAK -> ATH in 2025, so current form is ATH.
    120: "WAS", 121: "NYN", 133: "ATH", 134: "PIT", 135: "SDN", 136: "SEA",
    137: "SFN", 138: "SLN", 139: "TBA", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CHA", 146: "MIA", 147: "NYA", 158: "MIL",
}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def schedule(date: str, *, probables=True) -> list[dict]:
    """Games on a date. Returns dicts with retro team codes, state, pitchers, scores."""
    hydrate = "probablePitcher,team,linescore" if probables else "team,linescore"
    d = _get(f"{BASE}/schedule?sportId=1&date={date}&hydrate={hydrate}")
    out = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") != "R":            # regular season only
                continue
            a, h = g["teams"]["away"], g["teams"]["home"]
            ht, at = h["team"]["id"], a["team"]["id"]
            if ht not in TEAM_ID_TO_RETRO or at not in TEAM_ID_TO_RETRO:
                continue
            st = g["status"]["abstractGameState"]
            rec = {
                "game_pk": g["gamePk"], "date": g["gameDate"][:10],
                "home": TEAM_ID_TO_RETRO[ht], "away": TEAM_ID_TO_RETRO[at],
                "home_name": h["team"]["name"], "away_name": a["team"]["name"],
                "home_abbr": h["team"].get("abbreviation", ""),
                "away_abbr": a["team"].get("abbreviation", ""),
                "state": st,
                "home_sp": (h.get("probablePitcher") or {}).get("fullName"),
                "away_sp": (a.get("probablePitcher") or {}).get("fullName"),
                "start_utc": g.get("gameDate"),
            }
            if st == "Final":
                ls = g.get("teams", {})
                hs = ls["home"].get("score")
                as_ = ls["away"].get("score")
                if hs is not None and as_ is not None and hs != as_:
                    rec["home_win"] = 1.0 if hs > as_ else 0.0
                    rec["home_score"] = hs
                    rec["away_score"] = as_
            out.append(rec)
    return out


def season_finals(year: int, start_md="03-15", end_date=None) -> list[dict]:
    """All completed regular-season games from year-start to end_date (inclusive).

    One request per day is polite and avoids giant payloads; the schedule endpoint
    caps a date-range query, so we walk days.
    """
    from datetime import date as _date, timedelta

    y0 = _date(year, int(start_md[:2]), int(start_md[3:]))
    y1 = _date.fromisoformat(end_date) if end_date else _date(year, 11, 1)
    out = []
    d = y0
    while d <= y1:
        try:
            for g in schedule(d.isoformat(), probables=True):
                if "home_win" in g:
                    out.append(g)
        except Exception:  # noqa: BLE001 — a single bad day must not abort the backfill
            pass
        time.sleep(0.25)
        d += timedelta(days=1)
    out.sort(key=lambda g: g["date"])
    return out
