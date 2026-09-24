"""Fetch the NHL site data the model does not need but the pages do.

Everything here is DISPLAY data - rosters, production stats, standings
identity, the full season schedule. None of it is a model input and none of it
is market data. Standard library only; every step is independent, cached to a
data/ file with a max age, and failure-tolerant: a step that cannot reach the
API keeps its previous cache and the serve falls back further to the previous
payload (phase0/nhl_serve.py), so an offline run degrades to "older", never to
"missing".

  data/nhl_player_names.json      current rosters + bios (phase0/nhl_names.py)
  data/nhl_site_standings.json    per season: team names, division, conference,
                                  official clinch indicator and sequences
                                  (api-web /v1/standings/now, /standings/{date})
  data/nhl_site_stats_{season}.json   skater and goalie season lines
                                  (api.nhle.com stats REST summary; the
                                  api-web club-stats endpoint is the fallback)
  data/nhl_site_career.json       regular-season career totals, rostered ids
  data/nhl_schedule_{season}.json every game of the season (32 x
                                  api-web /v1/club-schedule-season/{team}/{season})

Called at the end of phase0/nhl_update.py (the 4-hourly refresh and the weekly
chain both run it before phase0/nhl_serve.py). `python phase0/nhl_site_fetch.py
--force` refetches everything.

Bounded in time, because refresh.py runs nhl_update under a hard subprocess
timeout and a hung NHL API must never cost the MLB deploy: REQ_TIMEOUT seconds
per request, a wall-clock budget for the whole run (BUDGET_S, checked before
every step and inside every per-club loop), and a per-club loop stops after
MAX_CONSEC_FAIL failures in a row. Whatever was not reached keeps its cache.

Season lines name their ice-time unit: skaters `toi_pg` (seconds per game),
goalies `toi_s` (total seconds).
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhl_names  # noqa: E402
from nhl_season_boundary import current_season  # noqa: E402
from nhl_site_teams import TEAMS  # noqa: E402

WEB = "https://api-web.nhle.com/v1"
REST = "https://api.nhle.com/stats/rest/en"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}

SPINE = "data/nhl_games.csv"
UPCOMING = "data/nhl_upcoming.json"
STANDINGS = "data/nhl_site_standings.json"
CAREER = "data/nhl_site_career.json"


REQ_TIMEOUT = nhl_names.REQ_TIMEOUT     # seconds per request (was 30-60)
BUDGET_S = 240.0                        # wall-clock budget for main()
MAX_CONSEC_FAIL = nhl_names.MAX_CONSEC_FAIL
over = nhl_names.over                   # over(deadline) -> budget spent?


def stats_path(season) -> str:
    return f"data/nhl_site_stats_{season}.json"


def schedule_path(season) -> str:
    return f"data/nhl_schedule_{season}.json"


# max cache age per step, hours. The 4-hourly CI job has no cache (these files
# are not committed), so there every step fetches; locally they save requests.
TTL = {"rosters": 6, "standings": 6, "stats_cur": 3, "stats_prev": 24 * 7,
       "career": 24, "schedule": 12}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get(url, timeout=REQ_TIMEOUT):
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def fresh(obj, hours) -> bool:
    return bool(obj) and nhl_names.age_hours(obj.get("fetched_at")) < hours


def _txt(v):
    return v.get("default") if isinstance(v, dict) else v


# ---------------------------------------------------------------- standings --
def parse_standings(d: dict) -> tuple[int | None, dict]:
    """/standings payload -> (season, {code: team block})."""
    season, teams = None, {}
    for s in d.get("standings") or []:
        code = _txt(s.get("teamAbbrev"))
        if not code:
            continue
        season = season or s.get("seasonId")
        teams[code] = {
            "name": _txt(s.get("teamName")), "city": _txt(s.get("placeName")),
            "nick": _txt(s.get("teamCommonName")),
            "div": s.get("divisionName"), "conf": s.get("conferenceName"),
            "clinch": s.get("clinchIndicator"),
            "gp": s.get("gamesPlayed"), "w": s.get("wins"), "l": s.get("losses"),
            "otl": s.get("otLosses"), "pts": s.get("points"),
            "rw": s.get("regulationWins"), "row": s.get("regulationPlusOtWins"),
            "gf": s.get("goalFor"), "ga": s.get("goalAgainst"),
            "div_seq": s.get("divisionSequence"),
            "conf_seq": s.get("conferenceSequence"),
            "league_seq": s.get("leagueSequence"),
            "wc_seq": s.get("wildcardSequence"),
            "date": s.get("date"),
        }
    return season, teams


def fetch_standings(prev_season: int | None, prev_season_last_date: str | None,
                    force=False, get_fn=get) -> str:
    """standings/now into the per-season cache, plus the previous season's
    final table (standings/{its last regular-season date}) if not cached."""
    cache = load(STANDINGS, {}) or {}
    seasons = cache.setdefault("seasons", {})
    msg = []
    if force or not fresh(cache, TTL["standings"]):
        d = get_fn(f"{WEB}/standings/now")
        season, teams = parse_standings(d)
        if season and teams:
            seasons[str(season)] = {"date": d.get("standingsDateTimeUtc"),
                                    "fetched_at": now_iso(), "teams": teams}
            cache["fetched_at"] = now_iso()
            msg.append(f"now={season} ({len(teams)} teams)")
    if prev_season and prev_season_last_date:
        pv = seasons.get(str(prev_season)) or {}
        have = bool(pv.get("teams")) and max(
            (t.get("date") or "") for t in pv["teams"].values()) >= prev_season_last_date
        if not have:
            d = get_fn(f"{WEB}/standings/{prev_season_last_date}")
            season, teams = parse_standings(d)
            if season and teams:
                seasons[str(season)] = {"date": prev_season_last_date,
                                        "fetched_at": now_iso(), "teams": teams}
                msg.append(f"final {season}")
    save(STANDINGS, cache)
    return ", ".join(msg) or "fresh"


# -------------------------------------------------------------------- stats --
def _skater_line(r: dict) -> dict:
    return {"gp": r.get("gamesPlayed"), "g": r.get("goals"), "a": r.get("assists"),
            "p": r.get("points"), "pm": r.get("plusMinus"),
            "ppp": r.get("ppPoints"), "sog": r.get("shots"),
            "pim": r.get("penaltyMinutes"),
            "toi_pg": (round(r["timeOnIcePerGame"])          # seconds per game
                       if r.get("timeOnIcePerGame") is not None else None),
            "tm": (r.get("teamAbbrevs") or "").replace(",", ", ") or None}


def _goalie_line(r: dict) -> dict:
    sv = r.get("savePct")
    gaa = r.get("goalsAgainstAverage")
    return {"gp": r.get("gamesPlayed"), "gs": r.get("gamesStarted"),
            "w": r.get("wins"), "l": r.get("losses"), "otl": r.get("otLosses"),
            "svp": round(sv, 4) if sv is not None else None,
            "gaa": round(gaa, 2) if gaa is not None else None,
            "so": r.get("shutouts"), "sa": r.get("shotsAgainst"),
            "ga": r.get("goalsAgainst"), "toi_s": r.get("timeOnIce"),   # total seconds
            "tm": (r.get("teamAbbrevs") or "").replace(",", ", ") or None}


def _rest_url(kind: str, cayenne: str, aggregate=False) -> str:
    return (f"{REST}/{kind}/summary?isAggregate={'true' if aggregate else 'false'}"
            f"&isGame=false&start=0&limit=-1&cayenneExp=" + urllib.parse.quote(cayenne))


def fetch_season_stats_rest(season: int, get_fn=get) -> dict:
    exp = f"gameTypeId=2 and seasonId<={season} and seasonId>={season}"
    sk = get_fn(_rest_url("skater", exp)).get("data") or []
    gl = get_fn(_rest_url("goalie", exp)).get("data") or []
    return {"skaters": {str(r["playerId"]): _skater_line(r) for r in sk if r.get("playerId")},
            "goalies": {str(r["playerId"]): _goalie_line(r) for r in gl if r.get("playerId")}}


def fetch_season_stats_club(season: int, teams=TEAMS, get_fn=get, sleep=0.15,
                            deadline: float | None = None) -> dict:
    """Fallback: api-web club-stats per team, summed per player across clubs.
    The club endpoint has no power-play assists, so `ppp` is null here.
    `clubs` in the answer = clubs that answered (the loop stops at the
    deadline or after MAX_CONSEC_FAIL failures in a row)."""
    sk, gl = {}, {}
    ok = fails = 0
    for t in teams:
        if over(deadline):
            print(f"  club-stats: time budget spent after {ok} clubs")
            break
        try:
            d = get_fn(f"{WEB}/club-stats/{t}/{season}/2")
            ok += 1
            fails = 0
        except Exception as ex:  # noqa: BLE001
            fails += 1
            print(f"  club-stats {t}: {ex}")
            if fails >= MAX_CONSEC_FAIL:
                print(f"  club-stats: {fails} failed requests in a row, stopping")
                break
            continue
        for r in d.get("skaters") or []:
            pid = str(r.get("playerId"))
            gp = r.get("gamesPlayed") or 0
            a = sk.setdefault(pid, {"gp": 0, "g": 0, "a": 0, "p": 0, "pm": 0,
                                    "ppp": None, "sog": 0, "pim": 0, "_toi": 0.0, "tm": []})
            for k, src in (("gp", "gamesPlayed"), ("g", "goals"), ("a", "assists"),
                           ("p", "points"), ("pm", "plusMinus"), ("sog", "shots"),
                           ("pim", "penaltyMinutes")):
                a[k] += r.get(src) or 0
            a["_toi"] += (r.get("avgTimeOnIcePerGame") or 0.0) * gp
            a["tm"].append(t)
        for r in d.get("goalies") or []:
            pid = str(r.get("playerId"))
            a = gl.setdefault(pid, {"gp": 0, "gs": 0, "w": 0, "l": 0, "otl": 0,
                                    "so": 0, "sa": 0, "ga": 0, "toi_s": 0, "tm": []})
            for k, src in (("gp", "gamesPlayed"), ("gs", "gamesStarted"), ("w", "wins"),
                           ("l", "losses"), ("otl", "overtimeLosses"), ("so", "shutouts"),
                           ("sa", "shotsAgainst"), ("ga", "goalsAgainst"), ("toi_s", "timeOnIce")):
                a[k] += r.get(src) or 0
            a["tm"].append(t)
        if sleep:
            time.sleep(sleep)
    for a in sk.values():
        a["toi_pg"] = round(a.pop("_toi") / a["gp"]) if a["gp"] else None
        a["tm"] = ", ".join(a["tm"])
    for a in gl.values():
        a["svp"] = round(1 - a["ga"] / a["sa"], 4) if a["sa"] else None
        a["gaa"] = round(a["ga"] * 3600 / a["toi_s"], 2) if a["toi_s"] else None
        a["tm"] = ", ".join(a["tm"])
    return {"skaters": sk, "goalies": gl, "clubs": ok}


def fetch_stats(season: int, ttl_h: float, force=False, get_fn=get,
                deadline: float | None = None) -> str:
    path = stats_path(season)
    old = load(path)
    if not force and fresh(old, ttl_h):
        return "fresh"
    had = bool(old and (old.get("skaters") or old.get("goalies")))
    src = "rest"
    try:
        d = fetch_season_stats_rest(season, get_fn)
    except Exception as ex:  # noqa: BLE001
        print(f"  stats REST {season}: {ex}; falling back to club-stats")
        src = "club"
        d = fetch_season_stats_club(season, teams=TEAMS, get_fn=get_fn, deadline=deadline)
        if d["clubs"] < len(TEAMS) and had:
            # a partial club sum would drop every player of the clubs not
            # reached; the previous complete answer is the better line
            return (f"club-stats answered {d['clubs']}/{len(TEAMS)} clubs, "
                    f"previous cache kept")
    if not d["skaters"] and not d["goalies"] and had:
        return "empty answer, previous cache kept"
    save(path, {"fetched_at": now_iso(), "season": season, "src": src, **d})
    return f"{len(d['skaters'])} skaters, {len(d['goalies'])} goalies ({src})"


def fetch_career(through: int, ids: set[str], force=False, get_fn=get) -> str:
    """Regular-season career totals for the given (rostered) ids."""
    old = load(CAREER)
    if not force and fresh(old, TTL["career"]) and old.get("through") == through:
        return "fresh"
    exp = f"gameTypeId=2 and seasonId<={through} and seasonId>=20002001"
    sk = get_fn(_rest_url("skater", exp, aggregate=True)).get("data") or []
    gl = get_fn(_rest_url("goalie", exp, aggregate=True)).get("data") or []
    out_s, out_g = {}, {}
    for r in sk:
        pid = str(r.get("playerId"))
        if pid in ids:
            out_s[pid] = {k: v for k, v in _skater_line(r).items()
                          if k not in ("tm", "toi_pg", "pim")}
    for r in gl:
        pid = str(r.get("playerId"))
        if pid in ids:
            gline = _goalie_line(r)
            out_g[pid] = {k: gline[k] for k in ("gp", "w", "l", "otl", "svp", "gaa", "so")}
    if not out_s and not out_g:
        return "empty answer, previous cache kept"
    save(CAREER, {"fetched_at": now_iso(), "through": through,
                  "skaters": out_s, "goalies": out_g})
    return f"{len(out_s)} skaters, {len(out_g)} goalies"


# ----------------------------------------------------------------- schedule --
def parse_club_schedule(d: dict) -> list[dict]:
    """club-schedule-season answer -> regular-season and playoff games."""
    out = []
    for g in d.get("games") or []:
        if g.get("gameType") not in (2, 3) or g.get("id") is None:
            continue
        h, a = g.get("homeTeam") or {}, g.get("awayTeam") or {}
        row = {"id": g["id"], "d": g.get("gameDate"), "t": g.get("startTimeUTC"),
               "home": (h.get("abbrev") or "").strip(),
               "away": (a.get("abbrev") or "").strip(),
               "type": g["gameType"], "state": g.get("gameState"),
               "sched": g.get("gameScheduleState"),
               "neutral": 1 if g.get("neutralSite") else 0}
        if g.get("gameState") in ("OFF", "FINAL"):
            row["hs"], row["as"] = h.get("score"), a.get("score")
            row["last"] = (g.get("gameOutcome") or {}).get("lastPeriodType")
        out.append(row)
    return out


def merge_schedule(old_games: list[dict], new_games: list[dict], complete: bool) -> list[dict]:
    """New answers override old ones by game id (a postponed game keeps its
    id and gets a new date). Old games absent from a PARTIAL fetch are kept;
    a complete fetch (every club answered) is authoritative."""
    by = {} if complete else {g["id"]: g for g in old_games}
    for g in new_games:
        by[g["id"]] = g
    return sorted(by.values(), key=lambda g: (g.get("d") or "", g["id"]))


def fetch_schedule(season: int, force=False, teams=TEAMS, get_fn=get, sleep=0.15,
                   deadline: float | None = None) -> str:
    path = schedule_path(season)
    old = load(path)
    if not force and fresh(old, TTL["schedule"]):
        return "fresh"
    games, ok, fails = {}, 0, 0
    for t in teams:
        if over(deadline):
            print(f"  schedule: time budget spent after {ok} clubs")
            break
        try:
            for g in parse_club_schedule(get_fn(f"{WEB}/club-schedule-season/{t}/{season}")):
                games[g["id"]] = g
            ok += 1
            fails = 0
        except Exception as ex:  # noqa: BLE001
            fails += 1
            print(f"  schedule {t}: {ex}")
            if fails >= MAX_CONSEC_FAIL:
                print(f"  schedule: {fails} failed requests in a row, stopping")
                break
        if sleep:
            time.sleep(sleep)
    if not ok:
        return "all requests failed, previous cache kept"
    merged = merge_schedule((old or {}).get("games") or [], list(games.values()),
                            complete=ok == len(teams))
    save(path, {"fetched_at": now_iso(), "season": season,
                "clubs": ok, "games": merged})
    n_reg = sum(1 for g in merged if g["type"] == 2)
    return f"{n_reg} regular-season games from {ok}/{len(teams)} clubs"


# --------------------------------------------------------------------- main --
def seasons_now() -> tuple[int | None, int | None, str | None]:
    """(current season, previous season, previous season's last regular-season
    date) - the same derivation the serve uses (completed OR scheduled)."""
    seasons, last_by = set(), {}
    try:
        for r in csv.DictReader(open(SPINE, encoding="utf-8")):
            if r.get("type") != "2":
                continue
            s = int(r["season"])
            seasons.add(s)
            if r["date"] > last_by.get(s, ""):
                last_by[s] = r["date"]
    except FileNotFoundError:
        pass
    up = load(UPCOMING, []) or []
    cur = current_season(seasons, (u.get("id") for u in up))
    prior = [s for s in seasons if cur is not None and s < cur]
    prev = max(prior) if prior else None
    return cur, prev, last_by.get(prev) if prev else None


def _rosters(force: bool, deadline: float | None = None) -> str:
    ok, n = nhl_names.refresh(0 if force else TTL["rosters"], deadline=deadline)
    return "fresh" if n == 0 else f"{ok}/{n} clubs answered"


def main(force: bool = False, budget_s: float = BUDGET_S) -> int:
    """Run every step inside `budget_s` seconds of wall clock. A step that
    would start after the budget is spent is skipped (its cache stands)."""
    deadline = time.monotonic() + budget_s
    cur, prev, prev_last = seasons_now()
    if cur is None:
        print("[nhl_site_fetch] no season derivable (empty spine), nothing fetched")
        return 0
    steps = [
        ("rosters", lambda: _rosters(force, deadline)),
        ("standings", lambda: fetch_standings(prev, prev_last, force)),
        (f"stats {cur}", lambda: fetch_stats(cur, TTL["stats_cur"], force, deadline=deadline)),
        (f"stats {prev}", lambda: fetch_stats(prev, TTL["stats_prev"], force, deadline=deadline)
         if prev else "n/a"),
        ("career", lambda: fetch_career(cur, {k for k, v in nhl_names.load().items()
                                              if v.get("on_roster")}, force)),
        (f"schedule {cur}", lambda: fetch_schedule(cur, force, deadline=deadline)),
    ]
    for name, fn in steps:
        if over(deadline):
            print(f"[nhl_site_fetch] {name}: skipped, {budget_s:.0f}s time budget spent; "
                  f"previous cache kept", flush=True)
            continue
        try:
            print(f"[nhl_site_fetch] {name}: {fn()}", flush=True)
        except Exception as ex:  # noqa: BLE001  (one dead endpoint never stops the rest)
            print(f"[nhl_site_fetch] {name}: FAILED ({ex}); previous cache kept", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(force="--force" in sys.argv))
