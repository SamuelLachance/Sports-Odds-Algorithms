"""Official pre-game starting goalies for an NHL game — the CONFIRMED-tier feed.

Serving-feasibility task of the breakthrough program
(documents/breakthrough_program_prereg_2026_09_24.md). Evidence
(data/bt_nhl_starter_feasibility.json):

  * api-web JSON never names tonight's starter before puck drop: landing
    matchup.goalieComparison lists EVERY rostered goalie (2-4 per team) in
    ascending playerId order at every pre-game horizon observed; boxscore
    goalies[].starter exists only once the game is OFF/FINAL; play-by-play
    rosterSpots lists the dressed goalies without a starter mark.
  * the nhl.com Club Playing Roster report (scores/htmlreports/{season}/RO{gg}.HTM,
    "* Starting Lineup in Bold") is posted pre-game with the dressed roster and,
    minutes before the scheduled start, marks each team's starting goalie bold.

This module parses that report and maps the bold goalie to an NHL player id.
It is a PROTOTYPE for the serving pipeline described in the feasibility file:
standard library only, no side effects on import, nothing here is wired into
nhl_update.py / nhl_serve.py yet.

    parse_playing_roster(html_bytes) -> dict
    starters_from_report(parsed, roster_spots=None) -> {"away": {...}, "home": {...}}
    fetch_confirmed(gid, now_utc=None) -> dict   (network: 1-2 requests)

A starter counts as CONFIRMED only when (a) the report marks exactly one
dressed goalie of that team bold and (b) the fetch happened before the
scheduled start with the game still FUT/PRE — enforced by fetch_confirmed().
"""
from __future__ import annotations

import datetime as dt
import gzip
import html as htmlmod
import json
import re
import unicodedata
import urllib.error
import urllib.request

HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
RO_URL = "https://www.nhl.com/scores/htmlreports/{season}/RO{code}.HTM"
PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
PRE_STATES = ("FUT", "PRE")

_CELL = re.compile(r'<td[^>]*class="([^"]*)"[^>]*>([^<]*)</td>', re.S)
_HDR = re.compile(r'<td align="center" style="font-size: 10px;font-weight:bold">([^<]*)</td>')
_GEN = re.compile(r"National Hockey League\s*(?:&nbsp;)?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[- ][0-9.]+)")
_START = re.compile(r"Start(?:&nbsp;|\s)*([0-9]{1,2}:[0-9]{2})?(?:&nbsp;|\s)*([A-Z]{3})")


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\((C|A)\)", "", s.upper())
    return re.sub(r"[^A-Z]", "", s)


def parse_playing_roster(raw: bytes) -> dict:
    """Parse an RO report into per-team dressed players, scratches and bold marks.

    The report prints four '#/Pos/Name' tables in a fixed order: visitor
    dressed, home dressed, visitor scratches, home scratches (the scratch tables
    are absent early). A cell's class contains 'bold' for a starting-lineup
    player. Returns
      {"gen_local": "YYYY-MM-DD-HH.MM.SS" | None,  report generation stamp
       "started": bool, "final": bool,              header state
       "header": [...],
       "away": {"dressed": [{"num","pos","name","bold"}], "scratches": [...]},
       "home": {...}, "n_tables": int}
    """
    t = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"\s+", " ", t)
    header = [htmlmod.unescape(h).strip() for h in _HDR.findall(t)]
    header = [h for h in header if h]
    gen = _GEN.search(t)
    start_m = next((m for m in _START.finditer(t) if m.group(1)), None)
    tables, cur = [], None
    for cls, txt in _CELL.findall(t):
        txt = htmlmod.unescape(txt).strip()
        if "heading" in cls and txt == "#":
            cur = []
            tables.append(cur)
            continue
        if cur is None or "heading" in cls or cls.strip() == "header":
            continue
        cur.append((cls, txt))

    def rows(tab):
        out = []
        for i in range(0, len(tab) - 2, 3):
            (c0, num), (_, pos), (_, name) = tab[i:i + 3]
            out.append({"num": num, "pos": pos, "name": name, "bold": "bold" in c0})
        return out

    tabs = [rows(tb) for tb in tables]
    while len(tabs) < 4:
        tabs.append([])
    return {
        "gen_local": gen.group(1).replace(" ", "-") if gen else None,
        "started": start_m is not None,
        "final": any(h == "Final" for h in header),
        "header": header,
        "away": {"dressed": tabs[0], "scratches": tabs[2]},
        "home": {"dressed": tabs[1], "scratches": tabs[3]},
        "n_tables": len(tables),
    }


def starters_from_report(parsed: dict, roster_spots=None, team_ids=None) -> dict:
    """Per side: dressed goalies, the bold one (if exactly one) and its player id.

    roster_spots: play-by-play rosterSpots (teamId, playerId, sweaterNumber,
    positionCode, firstName/lastName) — used to map (team, sweater number) to a
    player id; team_ids: {"home": id, "away": id}. Without them the id is None
    and the caller maps by name.
    """
    out = {}
    for side in ("away", "home"):
        gl = [p for p in parsed[side]["dressed"] if p["pos"] == "G"]
        bold = [p for p in gl if p["bold"]]
        pick = bold[0] if len(bold) == 1 else None
        pid = None
        if pick and roster_spots and team_ids:
            for r in roster_spots:
                if (r.get("teamId") == team_ids.get(side) and r.get("positionCode") == "G"
                        and str(r.get("sweaterNumber")) == pick["num"]):
                    pid = r.get("playerId")
                    break
            if pid is None:
                want = _norm_name(pick["name"])
                for r in roster_spots:
                    nm = (((r.get("firstName") or {}).get("default") or "") + " "
                          + ((r.get("lastName") or {}).get("default") or ""))
                    if r.get("teamId") == team_ids.get(side) and _norm_name(nm) == want:
                        pid = r.get("playerId")
                        break
        out[side] = {"dressed_goalies": [[p["num"], p["name"]] for p in gl],
                     "n_bold_goalies": len(bold),
                     "starter": [pick["num"], pick["name"]] if pick else None,
                     "starter_id": pid,
                     "confirmed": pick is not None}
    return out


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as ex:
        return ex.code, None
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return 200, raw


def fetch_confirmed(gid: int, now_utc: dt.datetime | None = None) -> dict:
    """One pre-game read of the official starters for game `gid`.

    Requests: play-by-play (state, start time, team ids, dressed roster) and the
    RO report. Returns {"gid", "t_fetch", "min_to_start", "state", "ro_status",
    "pre_game": bool, "away": {...}, "home": {...}}. A side is CONFIRMED only when
    pre_game is True AND its report marks exactly one dressed goalie bold; a read
    at or after the scheduled start, or with the game LIVE, is never a pre-game
    confirmation (rule 3 of the program: played games keep their PRE-GAME number).
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    st, raw = _get(PBP_URL.format(gid=gid))
    pbp = json.loads(raw) if raw else {}
    start = pbp.get("startTimeUTC")
    start_dt = dt.datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
    state = pbp.get("gameState")
    pre = bool(start_dt and now < start_dt and state in PRE_STATES)
    season = str(gid)[:4]
    code = str(gid)[4:]
    rs, rraw = _get(RO_URL.format(season=f"{season}{int(season) + 1}", code=code))
    rec = {"gid": gid, "t_fetch": now.isoformat(timespec="seconds"),
           "min_to_start": round((start_dt - now).total_seconds() / 60, 1) if start_dt else None,
           "state": state, "ro_status": rs, "pre_game": pre}
    if rraw is None:
        rec.update({"away": None, "home": None})
        return rec
    parsed = parse_playing_roster(rraw)
    team_ids = {"home": (pbp.get("homeTeam") or {}).get("id"),
                "away": (pbp.get("awayTeam") or {}).get("id")}
    sides = starters_from_report(parsed, pbp.get("rosterSpots"), team_ids)
    for side in ("away", "home"):
        sides[side]["confirmed"] = bool(pre and sides[side]["confirmed"])
    rec.update({"ro_gen_local": parsed["gen_local"], **sides})
    return rec


if __name__ == "__main__":
    import sys
    for g in sys.argv[1:]:
        print(json.dumps(fetch_confirmed(int(g)), ensure_ascii=False))
