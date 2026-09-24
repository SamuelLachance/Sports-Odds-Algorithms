"""Player-value program, NHL -- reduce the raw play-by-play archive to event rows.

Input   data/pv_nhl_pbp/{season}/{gid}.json.gz   (phase0/pv_nhl_fetch.py)
        data/nhl_games.csv                        (spine: gid, season, type)
        data/nhl_shots.csv                        (MoneyPuck unblocked shots with xG)
Output  data/pv_nhl_events.csv       one row per on-ice event (schema below), sorted
                                     by (gid, sortOrder) -> every DEV row
                                     (gid < 2018000000) precedes every TEST row
        data/pv_nhl_events.parquet   same table (fast loads; see pv_nhl_io.py)
        data/pv_nhl_rosters.csv      per-game dressed roster (play-by-play rosterSpots)
        data/pv_nhl_players.csv      player id -> name, position, first/last season
        data/pv_nhl_event_games.csv  per-game reduction status (ok / missing / bad)

This script PRINTS NOTHING computed from the data except row counts. All quality
checks (join rate, goal totals, box-score reconciliation) are in pv_nhl_qa.py and
run on DEV games only.

EVENT SCHEMA (data/pv_nhl_events.csv)
  gid, season, gtype (2 regular / 3 playoff)
  period, ptype (REG / OT / SO), per_sec (seconds elapsed in period),
  game_sec ((period-1)*1200 + per_sec), sort (API sortOrder), eid (API eventId)
  ev         API typeDescKey: goal, shot-on-goal, missed-shot, blocked-shot,
             faceoff, hit, giveaway, takeaway, penalty, failed-shot-attempt
  team       abbrev of the ACTING team, resolved through the game's rosterSpots
             (not eventOwnerTeamId, whose meaning for blocked shots changed across
             eras): shooting team for goal / shot-on-goal / missed-shot /
             blocked-shot / failed-shot-attempt; faceoff winner's team; hitter's
             team; giveaway / takeaway player's team; penalized team
  is_home    1 if `team` is the home team
  shooter    shooter (scorer on goals), all shot events incl. blocked
  assist1, assist2              goals only
  goalie     goalieInNetId as given (defending goalie; blank = empty net / not
             given; never given on blocked shots)
  blocker    blocked-shot blocker
  fo_win, fo_lose               faceoff winner / loser
  hitter, hittee                hit
  player     giveaway / takeaway player
  pen_by, pen_drawn, pen_served committedBy / drawnBy / servedBy
  pen_code   MIN / MAJ / MIS / GAM / MAT / PS / BEN ...; pen_desc (descKey);
  pen_min    duration (minutes)
  shot_type  wrist / slap / snap / backhand / tip-in / deflected / wrap-around ...
  reason     missed-shot reason (wide-of-net, over-net, goalpost, ...) or
             blocked-shot reason (blocked / teammate-blocked) where given
  x, y       raw rink coordinates (rink-fixed frame; teams switch ends by period)
  zone       zoneCode (O / D / N) re-expressed from the ACTING team's perspective
             (the API gives it from eventOwnerTeamId's side; flipped O<->D when
             the owner is not the acting team)
  net_x      shot events only: x of the net the shooting team attacks (+89 / -89),
             by majority vote of the sign of x over the period's unblocked shots
             with |x| > 25 (own shots vote for, opponent's against; shootout: own
             only) -- era-uniform; homeTeamDefendingSide is absent in early
             seasons and is not used
  dist, angle   shot events only: feet / degrees from net_x,0 (blocked shots: the
             block location, not the shot origin)
  sit        situationCode, 4 chars: away goalie in net, away skaters, home
             skaters, home goalie in net; a_g, a_sk, h_sk, h_g its digits
  sk_for, sk_ag, g_ag   skaters for / against `team`, opponent goalie in net (0/1)
             (outside the shootout, sit 1010 / 0101 marks a penalty shot; MoneyPuck
             carries no xG for penalty shots)
  score_h, score_a      score BEFORE the event (shootout goals never counted)
  xg         MoneyPuck xGoal for unblocked non-shootout shots, joined on
             (gid, period, is_home, per_sec) with ordinal tie-break; xg_match 1 =
             exact second and same goal flag, 2 = within +-TOL s and same goal
             flag, 3 = goal flag differs; blank = unmatched / not applicable

ROSTER SCHEMA (data/pv_nhl_rosters.csv)
  gid, season, gtype, side (H/A), team, pid, pos (C/L/R/D/G), num (sweater),
  g_start   goalies: 1 if in net for his team's first opposing shot of the game
  g_net     goalies: 1 if ever named goalieInNetId in the game
  n_ev      number of kept events that name the player in any role (a
            "did he play" signal -- dressed players can be injured / unused)
  toi_s     all-situations time on ice (s) from data/nhl_shifts.csv (shift
            charts); blank when the game has no shift rows
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict

import pandas as pd

ROOT = "data/pv_nhl_pbp"
GAMES = "data/nhl_games.csv"
SHOTS = "data/nhl_shots.csv"
SHIFTS = "data/nhl_shifts.csv"
OUT_EV = "data/pv_nhl_events.csv"
OUT_PQ = "data/pv_nhl_events.parquet"
OUT_RO = "data/pv_nhl_rosters.csv"
OUT_PL = "data/pv_nhl_players.csv"
OUT_GM = "data/pv_nhl_event_games.csv"
DEV_MAX_GID = 2018000000
TOL = 2                       # seconds, fuzzy xG join window

KEEP = ("goal", "shot-on-goal", "missed-shot", "blocked-shot", "faceoff", "hit",
        "giveaway", "takeaway", "penalty", "failed-shot-attempt")
SHOTS_ALL = ("goal", "shot-on-goal", "missed-shot", "blocked-shot",
             "failed-shot-attempt")
UNBLOCKED = ("goal", "shot-on-goal", "missed-shot")

COLS = ["gid", "season", "gtype", "period", "ptype", "per_sec", "game_sec", "sort",
        "eid", "ev", "team", "is_home",
        "shooter", "assist1", "assist2", "goalie", "blocker", "fo_win", "fo_lose",
        "hitter", "hittee", "player", "pen_by", "pen_drawn", "pen_served",
        "pen_code", "pen_desc", "pen_min", "shot_type", "reason",
        "x", "y", "zone", "net_x", "dist", "angle",
        "sit", "a_g", "a_sk", "h_sk", "h_g", "sk_for", "sk_ag", "g_ag",
        "score_h", "score_a", "xg", "xg_match"]
RCOLS = ["gid", "season", "gtype", "side", "team", "pid", "pos", "num",
         "g_start", "g_net", "n_ev", "toi_s"]
PLAYER_ROLES = ("shooter", "assist1", "assist2", "goalie", "blocker", "fo_win",
                "fo_lose", "hitter", "hittee", "player", "pen_by", "pen_drawn",
                "pen_served")


def path_for(gid, season):
    return f"{ROOT}/{season}/{gid}.json.gz"


def mmss(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return None


def iid(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def load_mp():
    """gid -> {(period, is_home): [(per_sec, xg, goal), ...] sorted}."""
    if not os.path.exists(SHOTS):
        return {}
    df = pd.read_csv(SHOTS, usecols=["nhl_game_id", "period", "per_sec", "is_home",
                                     "xg", "goal"])
    df = df.sort_values(["nhl_game_id", "period", "is_home", "per_sec"], kind="stable")
    out = {}
    g = df.nhl_game_id.to_numpy()
    p = df.period.to_numpy()
    h = df.is_home.to_numpy()
    s = df.per_sec.to_numpy()
    x = df.xg.to_numpy()
    gl = df.goal.to_numpy()
    for i in range(len(df)):
        d = out.setdefault(int(g[i]), {})
        d.setdefault((int(p[i]), int(h[i])), []).append((int(s[i]), float(x[i]),
                                                          int(gl[i])))
    return out


def load_toi():
    """(gid, pid) -> all-situations time on ice (s) from the NHL shift charts
    (data/nhl_shifts.csv, phase0/nhl_pull_shifts.py); exact duplicate shift rows
    are dropped before summing."""
    if not os.path.exists(SHIFTS):
        return {}
    parts = []
    for ch in pd.read_csv(SHIFTS, chunksize=2_000_000,
                          usecols=["game_id", "player_id", "period", "start_s", "end_s"]):
        ch = ch.drop_duplicates()
        ch = ch.assign(d=(ch.end_s - ch.start_s).clip(lower=0))
        parts.append(ch.groupby(["game_id", "player_id"]).d.sum())
    tot = pd.concat(parts).groupby(level=[0, 1]).sum()
    return {(int(g), int(p)): int(v) for (g, p), v in tot.items()}


def join_xg(rows, mp_game):
    """Assign MoneyPuck xG to the game's unblocked non-SO shots in place.

    Greedy min-cost matching within (period, is_home): cost = |dt| + 5 * goal
    flag mismatch, |dt| <= TOL; ties resolved by order (ordinal pairing)."""
    if not mp_game:
        return
    by = defaultdict(list)
    for r in rows:
        if r["ev"] in UNBLOCKED and r["ptype"] != "SO" and r["is_home"] != "":
            by[(r["period"], r["is_home"])].append(r)
    for key, P in by.items():
        M = mp_game.get(key)
        if not M:
            continue
        pairs = []
        mt = [m[0] for m in M]           # M is sorted by time; P (sortOrder) may not be
        for i, r in enumerate(P):
            t = r["per_sec"]
            if t is None:
                continue
            j = bisect.bisect_left(mt, t - TOL)
            while j < len(M) and M[j][0] <= t + TOL:
                dt = abs(M[j][0] - t)
                gm = int((r["ev"] == "goal") != bool(M[j][2]))
                pairs.append((dt + 5 * gm, i, j, dt, gm))
                j += 1
        pairs.sort()
        ui, uj = set(), set()
        for cost, i, j, dt, gm in pairs:
            if i in ui or j in uj:
                continue
            ui.add(i)
            uj.add(j)
            P[i]["xg"] = M[j][1]
            P[i]["xg_match"] = 3 if gm else (1 if dt == 0 else 2)


def parse_game(gid, season, gtype, d, mp_game, toi=None):
    home = d.get("homeTeam") or {}
    away = d.get("awayTeam") or {}
    hid, aid = home.get("id"), away.get("id")
    hab, aab = home.get("abbrev", ""), away.get("abbrev", "")
    tab = {hid: hab, aid: aab}
    ptm = {}                     # pid -> teamId
    roster = []
    for rs in d.get("rosterSpots") or []:
        pid = iid(rs.get("playerId"))
        if pid is None:
            continue
        ptm[pid] = rs.get("teamId")
        roster.append(rs)

    def team_of(pid):
        return ptm.get(pid) if pid is not None else None

    rows = []
    for pl in sorted(d.get("plays") or [], key=lambda p: p.get("sortOrder", 0)):
        ev = pl.get("typeDescKey")
        if ev not in KEEP:
            continue
        de = pl.get("details") or {}
        pdsc = pl.get("periodDescriptor") or {}
        per = iid(pdsc.get("number"))
        ptype = pdsc.get("periodType", "")
        sec = mmss(pl.get("timeInPeriod"))
        owner = de.get("eventOwnerTeamId")
        r = dict.fromkeys(COLS, "")
        r.update(gid=gid, season=season, gtype=gtype, period=per, ptype=ptype,
                 per_sec=sec, game_sec=(per - 1) * 1200 + sec
                 if (per is not None and sec is not None) else "",
                 sort=pl.get("sortOrder", ""), eid=pl.get("eventId", ""), ev=ev)
        tid = None
        if ev in SHOTS_ALL:
            sh = iid(de.get("scoringPlayerId") if ev == "goal"
                     else de.get("shootingPlayerId"))
            r["shooter"] = sh
            r["goalie"] = iid(de.get("goalieInNetId"))
            r["shot_type"] = de.get("shotType", "")
            r["reason"] = de.get("reason", "")
            if ev == "goal":
                r["assist1"] = iid(de.get("assist1PlayerId"))
                r["assist2"] = iid(de.get("assist2PlayerId"))
            tid = team_of(sh)
            if ev == "blocked-shot":
                bl = iid(de.get("blockingPlayerId"))
                r["blocker"] = bl
                if tid is None and team_of(bl) is not None:
                    tid = aid if team_of(bl) == hid else hid
        elif ev == "faceoff":
            r["fo_win"] = iid(de.get("winningPlayerId"))
            r["fo_lose"] = iid(de.get("losingPlayerId"))
            tid = team_of(r["fo_win"])
            if tid is None and team_of(r["fo_lose"]) is not None:
                tid = aid if team_of(r["fo_lose"]) == hid else hid
        elif ev == "hit":
            r["hitter"] = iid(de.get("hittingPlayerId"))
            r["hittee"] = iid(de.get("hitteePlayerId"))
            tid = team_of(r["hitter"])
            if tid is None and team_of(r["hittee"]) is not None:
                tid = aid if team_of(r["hittee"]) == hid else hid
        elif ev in ("giveaway", "takeaway"):
            r["player"] = iid(de.get("playerId"))
            tid = team_of(r["player"])
        elif ev == "penalty":
            r["pen_by"] = iid(de.get("committedByPlayerId"))
            r["pen_drawn"] = iid(de.get("drawnByPlayerId"))
            r["pen_served"] = iid(de.get("servedByPlayerId"))
            r["pen_code"] = de.get("typeCode", "")
            r["pen_desc"] = de.get("descKey", "")
            r["pen_min"] = de.get("duration", "")
            tid = team_of(r["pen_by"]) or team_of(r["pen_served"])
        if tid is None and owner in (hid, aid):
            tid = owner
        if tid in (hid, aid):
            r["team"] = tab[tid]
            r["is_home"] = 1 if tid == hid else 0
        for k in ("xCoord", "yCoord"):
            v = de.get(k)
            r["x" if k == "xCoord" else "y"] = v if v is not None else ""
        z = de.get("zoneCode", "")
        if owner in (hid, aid) and tid in (hid, aid) and owner != tid:
            z = {"O": "D", "D": "O"}.get(z, z)       # re-express for the acting team
        r["zone"] = z
        sit = pl.get("situationCode")
        if sit is not None and str(sit).isdigit():
            s = str(sit).zfill(4)
            r["sit"] = s
            r["a_g"], r["a_sk"], r["h_sk"], r["h_g"] = (int(c) for c in s)
            if r["is_home"] == 1:
                r["sk_for"], r["sk_ag"], r["g_ag"] = r["h_sk"], r["a_sk"], r["a_g"]
            elif r["is_home"] == 0:
                r["sk_for"], r["sk_ag"], r["g_ag"] = r["a_sk"], r["h_sk"], r["h_g"]
        rows.append(r)

    # score before each event (shootout goals never count toward the score)
    sh = sa = 0
    for r in rows:
        r["score_h"], r["score_a"] = sh, sa
        if r["ev"] == "goal" and r["ptype"] != "SO":
            if r["is_home"] == 1:
                sh += 1
            elif r["is_home"] == 0:
                sa += 1

    # attacked net by (period, team): majority vote over unblocked shots |x| > 25
    vote = defaultdict(int)
    for r in rows:
        if r["ev"] in UNBLOCKED and r["x"] != "" and abs(r["x"]) > 25 \
                and r["is_home"] != "":
            vote[(r["period"], r["is_home"])] += 1 if r["x"] > 0 else -1
    for r in rows:
        if r["ev"] not in SHOTS_ALL or r["x"] == "" or r["y"] == "" \
                or r["is_home"] == "":
            continue
        own = vote.get((r["period"], r["is_home"]), 0)
        opp = vote.get((r["period"], 1 - r["is_home"]), 0)
        # teams attack opposite nets within a period, so both teams' shots vote;
        # in a shootout both shoot at the same net, so only the team's own vote
        v = own if r["ptype"] == "SO" else own - opp
        if v == 0:
            v = 1 if r["x"] >= 0 else -1
        nx = 89 if v > 0 else -89
        r["net_x"] = nx
        dx = abs(nx - r["x"])
        r["dist"] = round(math.hypot(dx, r["y"]), 1)
        r["angle"] = round(math.degrees(math.atan2(abs(r["y"]), dx)), 1)

    join_xg(rows, mp_game)

    # rosters
    n_ev = defaultdict(int)
    gnet = set()
    for r in rows:
        for k in PLAYER_ROLES:
            if r[k] != "" and r[k] is not None:
                n_ev[r[k]] += 1
        if r["goalie"] not in ("", None):
            gnet.add(r["goalie"])
    first_g = {}                 # defending side (is_home of the goalie) -> goalie
    for r in rows:
        if r["ev"] in UNBLOCKED and r["goalie"] not in ("", None) \
                and r["is_home"] != "" and r["ptype"] != "SO":
            side = 1 - r["is_home"]
            if side not in first_g:
                first_g[side] = r["goalie"]
        if len(first_g) == 2:
            break
    rrows = []
    for rs in roster:
        pid = iid(rs["playerId"])
        tid = rs.get("teamId")
        if tid not in (hid, aid):
            continue
        is_h = 1 if tid == hid else 0
        pos = rs.get("positionCode", "")
        rrows.append({"gid": gid, "season": season, "gtype": gtype,
                      "side": "H" if is_h else "A", "team": tab[tid], "pid": pid,
                      "pos": pos, "num": rs.get("sweaterNumber", ""),
                      "g_start": int(first_g.get(is_h) == pid) if pos == "G" else "",
                      "g_net": int(pid in gnet) if pos == "G" else "",
                      "n_ev": n_ev.get(pid, 0),
                      "toi_s": (toi.get((gid, pid), 0) if toi is not None else "")})
    for r in rows:
        for k in PLAYER_ROLES:
            if r[k] is None:
                r[k] = ""
        if r["per_sec"] is None:
            r["per_sec"] = ""
    return rows, rrows, roster


def name_of(rs):
    f = (rs.get("firstName") or {}).get("default", "")
    l = (rs.get("lastName") or {}).get("default", "")
    return f, l


def write_parquet():
    """Stream the CSV into parquet (row groups ~1M rows, so a gid < DEV_MAX_GID
    filter skips TEST groups without decoding them)."""
    import pyarrow as pa
    import pyarrow.csv as pc
    import pyarrow.parquet as pq
    ints = ("gid", "season", "gtype", "period", "per_sec", "game_sec", "sort", "eid",
            "is_home", "net_x", "a_g", "a_sk", "h_sk", "h_g", "sk_for", "sk_ag",
            "g_ag", "score_h", "score_a", "xg_match", "pen_min") + PLAYER_ROLES
    strs = ("ptype", "ev", "team", "pen_code", "pen_desc", "shot_type", "reason",
            "zone", "sit")
    types = {c: pa.int64() for c in ints}
    types.update({c: pa.string() for c in strs})
    types.update({c: pa.float64() for c in ("x", "y", "dist", "angle", "xg")})
    assert set(types) == set(COLS), set(COLS) ^ set(types)
    rd = pc.open_csv(OUT_EV, read_options=pc.ReadOptions(block_size=1 << 26),
                     convert_options=pc.ConvertOptions(
                         column_types=types, null_values=[""],
                         strings_can_be_null=True))
    schema = pa.schema([(c, types[c]) for c in COLS])
    tmp = OUT_PQ + ".tmp"
    w = pq.ParquetWriter(tmp, schema, compression="zstd")
    buf, nb = [], 0
    for batch in rd:
        buf.append(batch.select(COLS) if hasattr(batch, "select") else batch)
        nb += batch.num_rows
        if nb >= 1_000_000:
            w.write_table(pa.Table.from_batches(buf, schema=schema))
            buf, nb = [], 0
    if buf:
        w.write_table(pa.Table.from_batches(buf, schema=schema))
    w.close()
    os.replace(tmp, OUT_PQ)
    print(f"wrote {OUT_PQ}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-only", action="store_true",
                    help="reduce DEV games only (development runs)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--tol", type=int, default=TOL, help="fuzzy xG join window (s)")
    args = ap.parse_args()
    globals()["TOL"] = args.tol

    spine = []
    for r in csv.DictReader(open(GAMES, encoding="utf-8")):
        spine.append((int(r["game_id"]), r["season"], int(r["type"])))
    spine.sort()
    if args.dev_only:
        spine = [s for s in spine if s[0] < DEV_MAX_GID]
    if args.limit:
        spine = spine[:args.limit]
    print(f"reducing {len(spine):,} games ...", flush=True)
    mp = load_mp()
    print(f"MoneyPuck shots loaded for {len(mp):,} games", flush=True)
    toi = load_toi()
    toi_games = {g for g, _ in toi}
    print(f"shift-chart TOI loaded for {len(toi_games):,} games", flush=True)

    tmp_ev, tmp_ro = OUT_EV + ".tmp", OUT_RO + ".tmp"
    fe = open(tmp_ev, "w", newline="", encoding="utf-8")
    we = csv.DictWriter(fe, fieldnames=COLS)
    we.writeheader()
    fr = open(tmp_ro, "w", newline="", encoding="utf-8")
    wr = csv.DictWriter(fr, fieldnames=RCOLS)
    wr.writeheader()
    gstat = []
    players = {}                 # pid -> [first, last, pos, first_season, last_season]
    n_rows = 0
    for k, (gid, season, gtype) in enumerate(spine):
        p = path_for(gid, season)
        if not os.path.exists(p):
            gstat.append((gid, season, gtype, "missing", 0, 0))
            continue
        try:
            with gzip.open(p, "rb") as fh:
                d = json.loads(fh.read())
            rows, rrows, roster = parse_game(gid, season, gtype, d, mp.get(gid),
                                             toi if gid in toi_games else None)
        except Exception as ex:  # noqa: BLE001
            gstat.append((gid, season, gtype, f"bad:{type(ex).__name__}", 0, 0))
            continue
        we.writerows(rows)
        wr.writerows(rrows)
        n_rows += len(rows)
        gstat.append((gid, season, gtype, "ok", len(rows), len(rrows)))
        for rs in roster:
            pid = iid(rs.get("playerId"))
            f, l = name_of(rs)
            cur = players.get(pid)
            if cur is None:
                players[pid] = [f, l, rs.get("positionCode", ""), season, season]
            else:
                cur[0], cur[1], cur[2], cur[4] = f, l, rs.get("positionCode", ""), season
        if (k + 1) % 2000 == 0:
            print(f"  {k+1:,}/{len(spine):,} games  {n_rows:,} event rows", flush=True)
    fe.close()
    fr.close()
    os.replace(tmp_ev, OUT_EV)
    os.replace(tmp_ro, OUT_RO)
    with open(OUT_GM, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gid", "season", "gtype", "status", "n_ev", "n_roster"])
        w.writerows(gstat)
    with open(OUT_PL, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "first", "last", "pos", "first_season", "last_season"])
        for pid in sorted(players):
            w.writerow([pid] + players[pid])
    n_ok = sum(1 for g in gstat if g[3] == "ok")
    print(f"wrote {OUT_EV}: {n_rows:,} rows | {OUT_RO} | {OUT_PL}: {len(players):,} "
          f"players | games ok {n_ok:,} / {len(gstat):,}", flush=True)

    if not args.no_parquet:
        write_parquet()
    return 0


if __name__ == "__main__":
    sys.exit(main())
