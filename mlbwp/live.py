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
        sp_id, sp, sp_bf = None, [0, 0, 0, 0, 0], 0
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
                sp_bf = int(pit.get("battersFaced", 0) or 0)
            else:
                rel_num += 13 * hr + 3 * (bb + hbp) - 2 * so
                rel_outs += o
        bat = {}
        for bid in t.get("batters", []):
            b = (players.get(f"ID{bid}") or {}).get("stats", {}).get("batting", {})
            pa = int(b.get("plateAppearances", 0) or 0)
            if pa > 0:
                bat[bid] = [pa, int(b.get("hits", 0) or 0), int(b.get("totalBases", 0) or 0)]
        out[side] = {"sp_id": sp_id, "sp": sp, "sp_bf": sp_bf,
                     "rel": [rel_num, rel_outs], "bat": bat}
    return out


def _batted_balls_by_pitcher(all_plays: list) -> dict:
    """{pitcher_mlbam: [gb, fb, pu]} non-HR batted balls by trajectory (ground/fly/
    pop), for xFIP (fly balls) and SIERA (net ground-ball rate). Matches the
    Retrosheet /G,/F,/P definitions; HRs are excluded (added back in the formulas)."""
    bb: dict = {}
    _idx = {"ground_ball": 0, "fly_ball": 1, "popup": 2}
    for pl in all_plays:
        res = pl.get("result", {})
        if res.get("type") != "atBat" or not pl.get("about", {}).get("isComplete"):
            continue
        if res.get("eventType") == "home_run":
            continue
        traj = None
        for pe in pl.get("playEvents", []):
            hd = pe.get("hitData")
            if hd and hd.get("trajectory"):
                traj = hd["trajectory"]
        i = _idx.get(traj)
        if i is not None:
            pid = (pl.get("matchup", {}).get("pitcher") or {}).get("id")
            if pid is not None:
                bb.setdefault(pid, [0, 0, 0])[i] += 1
    return bb


# Base labels -> number; batter destination bases by result event (for extra-base
# credit). Mirrors the Retrosheet baserunning parser (phase0/baserunning_parse.py):
# same run-value components, computed here from the StatsAPI structured runner
# movements instead of Retrosheet advance strings (different source, same method).
_BR_BASE = {"1B": 1, "2B": 2, "3B": 3, "4B": 4, "score": 4}
_BR_DEST = {"single": 1, "double": 2, "triple": 3, "home_run": 4,
            "walk": 1, "intent_walk": 1, "hit_by_pitch": 1, "catcher_interf": 1,
            "field_error": 1, "fielders_choice": 1, "fielders_choice_out": 1}


def _baserunning_by_player(all_plays: list) -> dict:
    """{player_mlbam: [pa, sb, cs, xb, oob, gidp]} from play-by-play runner movement.
      sb/cs  stolen bases / caught stealing (runner)
      xb     bases taken beyond what the batted ball forces (runner)
      oob    thrown out advancing, not a force out (runner)
      gidp   grounded into a double/triple play (batter)
    pa is the batter's completed plate appearances (the rate denominator)."""
    agg: dict = {}

    def cr(pid):
        return agg.setdefault(pid, [0, 0, 0, 0, 0, 0])
    for pl in all_plays:
        res = pl.get("result", {})
        ev = res.get("eventType")
        complete = pl.get("about", {}).get("isComplete")
        batter = (pl.get("matchup", {}).get("batter") or {}).get("id")
        if res.get("type") == "atBat" and complete and batter is not None:
            cr(batter)[0] += 1
            if ev in ("grounded_into_double_play", "grounded_into_triple_play"):
                cr(batter)[5] += 1
        dest = _BR_DEST.get(ev, 0)
        for rn in pl.get("runners", []):
            mv = rn.get("movement", {})
            de = rn.get("details", {})
            rid = (de.get("runner") or {}).get("id")
            mr = de.get("movementReason") or ""
            if rid is None:
                continue
            if "stolen_base" in mr:
                cr(rid)[1] += 1
                continue
            if "caught_stealing" in mr:
                cr(rid)[2] += 1
                continue
            if "pickoff" in mr:                       # plain pickoff isn't a CS
                continue
            if mv.get("start") is None:               # batter-runner: the hit itself
                continue
            if mv.get("isOut"):
                if mr != "r_force_out":               # a force out isn't a baserunning out
                    cr(rid)[4] += 1
                continue
            end = mv.get("end")
            if end is None:
                continue
            extra = (_BR_BASE.get(end, 0) - _BR_BASE.get(mv["start"], 0)) - dest
            if extra > 0:
                cr(rid)[3] += extra
    return agg


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

    Returns {"home": side, "away": side, "pa": [[batter,pitcher,on_base], ...],
             "baserun": {mlbam: [PA,SB,CS,XB,OOB,GIDP]}} where each side is
    {"sp_id", "sp":[outs,HR,BB,HBP,SO], "rel":[fip_num,outs], "bat":{mlbam:[PA,H,TB]}}.
    """
    d = _get(f"{BASE_V11}/game/{game_pk}/feed/live")
    live = d.get("liveData", {})
    plays = live.get("plays", {}).get("allPlays", [])
    sides = _parse_box(live.get("boxscore", {}))
    bbc = _batted_balls_by_pitcher(plays)
    for side in ("home", "away"):
        gb, fb, pu = bbc.get(sides[side].get("sp_id"), [0, 0, 0])
        sides[side]["sp_fb"] = fb
        sides[side]["sp_gb"] = gb
        sides[side]["sp_pu"] = pu
    sides["pa"] = _parse_plays(plays)
    sides["baserun"] = _baserunning_by_player(plays)
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
    failed = []
    d = y0
    while d <= y1:
        try:
            for g in schedule(d.isoformat(), probables=True):
                if "home_win" in g:
                    out.append(g)
        except Exception as exc:  # noqa: BLE001 — a bad day must not abort the backfill
            # Skipping the day is right; doing it silently is not. This list is
            # the model's replay input AND the ledger's grading source, so a
            # dropped day means ratings walked without those games and picks
            # that can never grade.
            failed.append((d.isoformat(), str(exc)[:80]))
        time.sleep(0.25)
        d += timedelta(days=1)
    if failed:
        print(f"WARNING live: {len(failed)} day(s) failed in the {year} finals walk; "
              f"those games are MISSING from the replay and cannot grade. "
              f"First: {failed[0][0]} ({failed[0][1]})")
    out.sort(key=lambda g: g["date"])
    return out


def finals_write_is_safe(n_new: int, n_prev: int) -> bool:
    """Whether a freshly walked finals list may overwrite the cached one.

    Completed games never un-complete, so within a season the count is
    monotonically non-decreasing. A smaller result is a failed walk, not
    reality — and writing it would replay the model on an incomplete season and
    strand the missing games as ungradeable picks.

    Strict, and self-healing: a block keeps the good file for one cycle, and the
    next walk (with more games completed) clears the bar on its own.
    """
    return n_new >= n_prev
