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
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = "https://statsapi.mlb.com/api/v1"
BASE_V11 = "https://statsapi.mlb.com/api/v1.1"     # feed/live (box score + play-by-play)
UA = "Mozilla/5.0 (mlbwp research)"
ET = ZoneInfo("America/New_York")     # the site's canonical game-day timezone


def et_date(iso_utc: str) -> str:
    """The US-Eastern calendar date of a UTC ISO timestamp (a late ET game whose
    first pitch is after midnight UTC still belongs to the ET game day)."""
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return dt.astimezone(ET).date().isoformat()

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
                "game_pk": g["gamePk"], "date": et_date(g["gameDate"]),
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


# Play outcomes that put the batter on base (matches Retrosheet's on_base
# definition used to train the TrueSkill ratings: hit, walk, HBP, reached-on-error).
_ON_BASE_EVENTS = {
    "single", "double", "triple", "home_run", "walk", "intent_walk",
    "hit_by_pitch", "field_error", "catcher_interf",
}


def _parse_box(box: dict) -> dict:
    """Per-side pitching + batting lines from a boxscore dict."""
    out = {}
    for side in ("home", "away"):
        t = box.get("teams", {}).get(side, {})
        if not t.get("pitchers"):
            raise ValueError("incomplete boxscore")
        players = t.get("players", {})
        sp_id, sp = None, [0, 0, 0, 0, 0]
        rel_num, rel_outs = 0.0, 0
        for pid in t.get("pitchers", []):
            pit = (players.get(f"ID{pid}") or {}).get("stats", {}).get("pitching", {})
            o = int(pit.get("outs", 0) or 0)
            hr = int(pit.get("homeRuns", 0) or 0)
            bb = int(pit.get("baseOnBalls", 0) or 0)
            hbp = int(pit.get("hitByPitch", 0) or 0)
            so = int(pit.get("strikeOuts", 0) or 0)
            if str(pit.get("gamesStarted", 0)) == "1":
                sp_id, sp = pid, [o, hr, bb, hbp, so]
            else:
                rel_num += 13 * hr + 3 * (bb + hbp) - 2 * so
                rel_outs += o
        bat = {}
        for bid in t.get("batters", []):
            b = (players.get(f"ID{bid}") or {}).get("stats", {}).get("batting", {})
            pa = int(b.get("plateAppearances", 0) or 0)
            if pa > 0:
                bat[bid] = [pa, int(b.get("hits", 0) or 0), int(b.get("totalBases", 0) or 0)]
        out[side] = {"sp_id": sp_id, "sp": sp, "rel": [rel_num, rel_outs], "bat": bat}
    return out


def _parse_plays(all_plays: list) -> list:
    """Completed plate appearances as (batter_mlbam, pitcher_mlbam, on_base), in order."""
    pas = []
    for pl in all_plays:
        res = pl.get("result", {})
        if res.get("type") != "atBat" or not pl.get("about", {}).get("isComplete"):
            continue
        m = pl.get("matchup", {})
        b = (m.get("batter") or {}).get("id")
        p = (m.get("pitcher") or {}).get("id")
        if b is None or p is None:
            continue
        pas.append([b, p, 1 if res.get("eventType") in _ON_BASE_EVENTS else 0])
    return pas


def game_data(game_pk) -> dict:
    """Everything the live ratings need from one game, in a single feed/live fetch.
    Market-free — only who pitched/batted and the outcome.

    Returns {"home": side, "away": side, "pa": [[batter,pitcher,on_base], ...]} where
    each side is {"sp_id", "sp":[outs,HR,BB,HBP,SO], "rel":[fip_num,outs], "bat":{mlbam:[PA,H,TB]}}.
    """
    d = _get(f"{BASE_V11}/game/{game_pk}/feed/live")
    live = d.get("liveData", {})
    sides = _parse_box(live.get("boxscore", {}))
    sides["pa"] = _parse_plays(live.get("plays", {}).get("allPlays", []))
    return sides


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
