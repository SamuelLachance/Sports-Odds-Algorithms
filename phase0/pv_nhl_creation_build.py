"""Player-value program, NHL -- component "creation", STEP 1: per player-game facts.

DEV ONLY (gid < 2018000000). For every dressed skater in every DEV game this
writes what happened IN THAT GAME (no rating yet, nothing walk-forward here):

  toi_<st>            seconds on ice in strength state <st> (from shift charts)
  individual (<st>):  icf  iff  isf  ixg  g  a1  a2
                      ireb irebxg  (rebound attempts: <=3 s after a same-team SOG)
                      irush irushxg (rush attempts: <=4 s after an event in the
                                     shooter's neutral/defensive zone)
                      irc           (rebounds CREATED: own SOG followed <=3 s by a
                                     same-team attempt)
  on-ice (ev, pp):    on_cf on_ff on_xgf on_gf / on_ca on_fa on_xga on_ga
  team (ev, pp):      tm_toi tm_cf tm_xgf tm_gf ... (team totals, for off-ice)
  penalties:          pdrawn ptaken (minor + major, all states)

Strength states, always from the player's own team's side:
  ev = 5v5 with both goalies in; pp = more skaters, both goalies in;
  sh = fewer skaters, both goalies in; eo = other even (4v4, 3v3);
  xx = anything else (an empty net, penalty shots).
Individual and on-ice events take their state from the pbp situationCode at the
event; TOI takes it from the shift charts (per-second on-ice counts). The QA
block records how often the two agree at event times.

On-ice convention: a skater is on the ice for an event at second t of a period
when one of his shifts has start < t <= end (the standard goal convention: the
player whose shift ends at t is on, the one whose shift starts at t is not).

Inputs: data/pv_nhl_events.parquet (via pv_nhl_io, DEV guard), pv_nhl_rosters,
data/nhl_shifts.csv (filtered to DEV gids on read, cached to
data/pv_nhl_creation_shifts_dev.parquet), data/nhl_games.csv (dates).
Output: data/pv_nhl_creation_pg.parquet, data/pv_nhl_creation_build_qa.json.

    python phase0/pv_nhl_creation_build.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_nhl_io import DEV_MAX_GID, load_events, load_rosters  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIFT_CACHE = os.path.join(ROOT, "data/pv_nhl_creation_shifts_dev.parquet")
OUT = os.path.join(ROOT, "data/pv_nhl_creation_pg.parquet")
QA_OUT = os.path.join(ROOT, "data/pv_nhl_creation_build_qa.json")
STATES = ["ev", "pp", "sh", "eo", "xx"]
IND = ["icf", "iff", "isf", "ixg", "g", "a1", "a2", "ireb", "irebxg", "irush",
       "irushxg", "irc"]
ONST = ["ev", "pp", "sh"]
ONV = ["cf", "ff", "sf", "xgf", "gf"]          # 'for' names; 'against' = ca fa sa xga ga
ATT = {"goal", "shot-on-goal", "missed-shot", "blocked-shot"}
UNB = {"goal", "shot-on-goal", "missed-shot"}
SOG = {"goal", "shot-on-goal"}
PLEN = 1200


def state_code(skf, ska, gf, ga):
    skf, ska, gf, ga = (np.asarray(a) for a in (skf, ska, gf, ga))
    s = np.full(len(skf), 4, np.int8)
    both = (gf >= 1) & (ga >= 1)
    s[both & (skf == 5) & (ska == 5)] = 0
    s[both & (skf > ska)] = 1
    s[both & (skf < ska)] = 2
    s[both & (skf == ska) & (skf != 5)] = 3
    return s


# ------------------------------------------------------------------ inputs ---
def load_shifts():
    if os.path.exists(SHIFT_CACHE):
        s = pd.read_parquet(SHIFT_CACHE)
    else:
        parts = []
        for ch in pd.read_csv(os.path.join(ROOT, "data/nhl_shifts.csv"),
                              chunksize=3_000_000, dtype={"team": str},
                              low_memory=False):
            ch = ch[pd.to_numeric(ch.game_id, errors="coerce").notna()].copy()
            ch["game_id"] = pd.to_numeric(ch.game_id).astype(np.int64)
            ch = ch[ch.game_id < DEV_MAX_GID]
            for c in ("player_id", "period", "start_s", "end_s"):
                ch[c] = pd.to_numeric(ch[c]).astype(np.int64)
            parts.append(ch[["game_id", "player_id", "period", "start_s", "end_s"]])
        s = pd.concat(parts, ignore_index=True)
        s.to_parquet(SHIFT_CACHE, index=False)
    assert int(s.game_id.max()) < DEV_MAX_GID, "TEST gid leaked"
    return s


def prep_events():
    cols = ["season", "gtype", "period", "ptype", "per_sec", "sort", "ev", "team",
            "is_home", "shooter", "assist1", "assist2", "zone", "sit", "a_g", "a_sk",
            "h_sk", "h_g", "xg", "pen_by", "pen_drawn", "pen_code"]
    e = load_events(columns=cols)
    e = e[e.ptype != "SO"].copy()
    e = e.sort_values(["gid", "period", "per_sec", "sort"], kind="stable").reset_index(drop=True)
    home = e.is_home.to_numpy() == 1
    # state from the home side and from the away side (pbp situationCode)
    st_h = state_code(e.h_sk, e.a_sk, e.h_g, e.a_g)
    st_a = state_code(e.a_sk, e.h_sk, e.a_g, e.h_g)
    e["st_h"], e["st_a"] = st_h, st_a
    e["st_for"] = np.where(home, st_h, st_a)            # acting team's state
    ev = e.ev.to_numpy()
    att = np.isin(ev, list(ATT))
    unb = np.isin(ev, list(UNB))
    sog = np.isin(ev, list(SOG))
    e["att"], e["unb"], e["sog"] = att, unb, sog
    e["goal"] = ev == "goal"
    e["xg0"] = np.where(unb, e.xg.fillna(0.0).to_numpy(), 0.0)
    e["xg_missing"] = unb & e.xg.isna().to_numpy()
    # zone from the acting team's side; blocked-shot rows carry the BLOCKER's zone
    # in this archive (99% 'D'), so flip them to the shooting team's side.
    z = e.zone.fillna("").to_numpy().astype(object)
    blk = ev == "blocked-shot"
    flip = {"O": "D", "D": "O", "N": "N", "": ""}
    z[blk] = [flip[v] for v in z[blk]]
    e["zone_act"] = z
    # previous event in the same game and period
    same = (e.gid.to_numpy()[1:] == e.gid.to_numpy()[:-1]) & \
           (e.period.to_numpy()[1:] == e.period.to_numpy()[:-1])
    prev_ok = np.r_[False, same]
    dt = np.r_[np.inf, np.diff(e.per_sec.to_numpy()).astype(float)]
    dt[~prev_ok] = np.inf
    pteam = np.r_[[""], e.team.to_numpy()[:-1]].astype(object)
    pev = np.r_[[""], ev[:-1]].astype(object)
    pz = np.r_[[""], z[:-1]].astype(object)
    team = e.team.to_numpy().astype(object)
    same_team = pteam == team
    pz_shooter = np.where(same_team, pz, np.vectorize(flip.get)(pz.astype(str)))
    reb = att & prev_ok & (pev == "shot-on-goal") & same_team & (dt <= 3)
    rush = att & prev_ok & (dt <= 4) & np.isin(pz_shooter, ["N", "D"]) & ~reb
    e["reb"], e["rush"] = reb, rush
    # rebound CREATED: flag the previous SOG row
    rc = np.zeros(len(e), bool)
    rc[np.where(reb)[0] - 1] = True
    e["rc"] = rc
    return e


# ---------------------------------------------------------- individual part ---
def individual_counts(e):
    """(gid, pid) x '<stat>_<state>' individual counts, all DEV games at once."""
    out = []
    a = e[e.att & e.shooter.notna()]
    a = a.assign(pid=a.shooter.astype(np.int64), st=a.st_for)
    agg = pd.DataFrame({
        "gid": a.gid, "pid": a.pid, "st": a.st,
        "icf": 1.0, "iff": a.unb.astype(float), "isf": a.sog.astype(float),
        "ixg": a.xg0, "g": a.goal.astype(float),
        "ireb": a.reb.astype(float), "irebxg": np.where(a.reb, a.xg0, 0.0),
        "irush": a.rush.astype(float), "irushxg": np.where(a.rush, a.xg0, 0.0),
        "irc": a.rc.astype(float), "xgmiss": a.xg_missing.astype(float)})
    out.append(agg.groupby(["gid", "pid", "st"], as_index=False).sum())
    gl = e[e.goal]
    for col, name in (("assist1", "a1"), ("assist2", "a2")):
        x = gl[gl[col].notna()]
        out.append(pd.DataFrame({"gid": x.gid, "pid": x[col].astype(np.int64),
                                 "st": x.st_for, name: 1.0})
                   .groupby(["gid", "pid", "st"], as_index=False).sum())
    long = pd.concat(out, ignore_index=True).fillna(0.0)
    long = long.groupby(["gid", "pid", "st"], as_index=False).sum()
    stats = [c for c in long.columns if c not in ("gid", "pid", "st")]
    wide = long.pivot_table(index=["gid", "pid"], columns="st", values=stats,
                            aggfunc="sum", fill_value=0.0)
    wide.columns = [f"{s}_{STATES[int(k)]}" for s, k in wide.columns]
    wide = wide.reset_index()
    # penalties drawn / taken (minor + major), all states
    p = e[(e.ev == "penalty") & e.pen_code.isin(["MIN", "MAJ"])]
    pd_ = p[p.pen_drawn.notna()].groupby(["gid", p.pen_drawn[p.pen_drawn.notna()]
                                          .astype(np.int64).rename("pid")]).size()
    pt_ = p[p.pen_by.notna()].groupby(["gid", p.pen_by[p.pen_by.notna()]
                                       .astype(np.int64).rename("pid")]).size()
    pen = pd.concat([pd_.rename("pdrawn"), pt_.rename("ptaken")], axis=1).fillna(0.0)
    wide = wide.merge(pen.reset_index(), on=["gid", "pid"], how="outer")
    return wide


# ------------------------------------------------------------ on-ice part ---
def _game(args):
    gid, ro, sh, ev = args
    pids = ro.pid.to_numpy()
    side_h = (ro.side.to_numpy() == "H")
    isg = ro.pos.to_numpy() == "G"
    pix = {int(p): i for i, p in enumerate(pids)}
    n = len(pids)
    sh = sh[sh.player_id.isin(pix)]
    nper = int(max(sh.period.max() if len(sh) else 3, ev.period.max() if len(ev) else 3, 3))
    T = nper * PLEN
    D = np.zeros((n, T + 1), np.int16)
    if len(sh):
        idx = sh.player_id.map(pix).to_numpy()
        base = (sh.period.to_numpy() - 1) * PLEN
        s0 = base + np.clip(sh.start_s.to_numpy(), 0, PLEN)
        s1 = base + np.clip(sh.end_s.to_numpy(), 0, PLEN)
        ok = s1 > s0
        np.add.at(D, (idx[ok], s0[ok]), 1)
        np.add.at(D, (idx[ok], s1[ok]), -1)
    P = np.cumsum(D, axis=1)[:, :T] > 0
    skh, ska = P[side_h & ~isg].sum(0), P[~side_h & ~isg].sum(0)
    gh, ga = P[side_h & isg].any(0), P[~side_h & isg].any(0)
    st_h = state_code(skh, ska, gh, ga)
    st_a = state_code(ska, skh, ga, gh)
    res = {"pid": pids}
    stp = np.where(side_h[:, None], st_h[None, :], st_a[None, :])   # n x T
    for k, name in enumerate(STATES):
        res[f"toi_{name}"] = (P & (stp == k)).sum(1).astype(float)
    # team EV/PP seconds per side (seconds where the side is in state k)
    tm_sec = {("H", k): float(((st_h == k) & (skh + ska > 0)).sum()) for k in (0, 1)}
    tm_sec.update({("A", k): float(((st_a == k) & (skh + ska > 0)).sum()) for k in (0, 1)})

    q = {}
    if len(ev):
        a = ev[ev.att]
        b = ((a.period.to_numpy() - 1) * PLEN + np.maximum(a.per_sec.to_numpy() - 1, 0))
        b = np.clip(b, 0, T - 1)
        on = P[:, b].astype(np.float32)                          # n x m
        ish = a.is_home.to_numpy() == 1
        vals = {"cf": np.ones(len(a)), "ff": a.unb.to_numpy().astype(float),
                "sf": a.sog.to_numpy().astype(float), "xgf": a.xg0.to_numpy(),
                "gf": a.goal.to_numpy().astype(float)}
        sth, sta = a.st_h.to_numpy(), a.st_a.to_numpy()
        cols, mats_h, mats_a = [], [], []
        for k in (0, 1, 2):
            for v, arr in vals.items():
                vn = v[:-1] if v.endswith("f") else v
                # 'for' for a home player: home events in home-state k
                mats_h.append(arr * (ish & (sth == k)))
                mats_a.append(arr * (~ish & (sta == k)))
                cols.append(f"on_{v}_{ONST[k]}")
                mats_h.append(arr * (~ish & (sth == k)))
                mats_a.append(arr * (ish & (sta == k)))
                cols.append(f"on_{vn}a_{ONST[k]}")
        Mh, Ma = np.column_stack(mats_h), np.column_stack(mats_a)
        oh = on[side_h] @ Mh
        oa = on[~side_h] @ Ma
        O = np.zeros((n, len(cols)))
        O[side_h], O[~side_h] = oh, oa
        for j, c in enumerate(cols):
            res[c] = O[:, j]
        tot_h, tot_a = Mh.sum(0), Ma.sum(0)
        for j, c in enumerate(cols):
            res["tm_" + c[3:]] = np.where(side_h, tot_h[j], tot_a[j])
        # QA: on-ice skater counts at 5v5 pbp events, and shift-vs-pbp state
        ev5 = (sth == 0)
        if ev5.any():
            onsk_h = on[side_h & ~isg][:, ev5].sum(0)
            onsk_a = on[~side_h & ~isg][:, ev5].sum(0)
            q["ev5_events"] = int(ev5.sum())
            q["ev5_both5"] = int(((onsk_h == 5) & (onsk_a == 5)).sum())
        q["att"] = int(len(a))
        q["state_agree"] = int((st_h[b] == sth).sum())
    res["tm_toi_ev"] = np.where(side_h, tm_sec[("H", 0)], tm_sec[("A", 0)])
    res["tm_toi_pp"] = np.where(side_h, tm_sec[("H", 1)], tm_sec[("A", 1)])
    df = pd.DataFrame(res)
    df.insert(0, "gid", gid)
    q["gid"] = gid
    q["shift_players"] = int(P.any(1).sum())
    q["dressed"] = n
    return df[~isg], q


def _season(args):
    season, ro, sh, ev = args
    rg = dict(tuple(ro.groupby("gid")))
    sg = dict(tuple(sh.groupby("game_id")))
    eg = dict(tuple(ev.groupby("gid")))
    frames, qas = [], []
    empty_sh = sh.iloc[:0]
    empty_ev = ev.iloc[:0]
    for gid in sorted(rg):
        df, q = _game((gid, rg[gid], sg.get(gid, empty_sh), eg.get(gid, empty_ev)))
        frames.append(df)
        qas.append(q)
    return season, pd.concat(frames, ignore_index=True), qas


def main():
    t0 = time.time()
    e = prep_events()
    print(f"events {len(e):,} ({time.time()-t0:.0f}s)", flush=True)
    ro = load_rosters()
    sh = load_shifts()
    sh = sh[(sh.end_s > sh.start_s)]
    # regular-season period 5 is the shootout: no skating time
    gtype_sh = (sh.game_id // 10000) % 10
    sh = sh[~((gtype_sh == 2) & (sh.period >= 5))]
    print(f"rosters {len(ro):,} shifts {len(sh):,} ({time.time()-t0:.0f}s)", flush=True)
    ind = individual_counts(e)
    print(f"individual rows {len(ind):,} ({time.time()-t0:.0f}s)", flush=True)

    keep_ev = ["gid", "period", "per_sec", "is_home", "att", "unb", "sog", "goal",
               "xg0", "st_h", "st_a"]
    evs = e[keep_ev]
    seasons = sorted(ro.season.unique())
    sh_season = (sh.game_id // 1000000).astype(np.int64)
    jobs = []
    for s in seasons:
        y0 = s // 10000
        jobs.append((s, ro[ro.season == s], sh[sh_season == y0], evs[evs.gid // 1000000 == y0]))
    with Pool(len(jobs)) as pool:
        outs = pool.map(_season, jobs)
    onice = pd.concat([o[1] for o in outs], ignore_index=True)
    qas = [q for o in outs for q in o[2]]
    print(f"on-ice rows {len(onice):,} ({time.time()-t0:.0f}s)", flush=True)

    base = ro[ro.pos != "G"][["gid", "season", "gtype", "side", "team", "pid", "pos", "toi_s"]]
    games = pd.read_csv(os.path.join(ROOT, "data/nhl_games.csv"), usecols=["game_id", "date"])
    games = games[games.game_id < DEV_MAX_GID].rename(columns={"game_id": "gid"})
    df = base.merge(games, on="gid", how="left").merge(onice, on=["gid", "pid"], how="left")
    n_before = len(df)
    df = df.merge(ind, on=["gid", "pid"], how="left")
    assert len(df) == n_before
    # individual events by players not in the dressed-skater base (goalies etc.)
    orphan = ind.merge(base[["gid", "pid"]], on=["gid", "pid"], how="left", indicator=True)
    orphan_n = int((orphan._merge == "left_only").sum())
    num = [c for c in df.columns if c not in ("gid", "season", "gtype", "side", "team",
                                              "pid", "pos", "date")]
    df[num] = df[num].fillna(0.0)
    for c in num:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df = df.sort_values(["gid", "side", "pid"]).reset_index(drop=True)
    assert int(df.gid.max()) < DEV_MAX_GID
    df.to_parquet(OUT, index=False)

    Q = pd.DataFrame(qas)
    toi_all = df[[f"toi_{s}" for s in STATES]].sum(1)
    has = df.toi_s > 0
    qa = {
        "rows": len(df), "games": int(df.gid.nunique()),
        "seasons": [int(s) for s in seasons],
        "events_nonSO": len(e),
        "xg_missing_unblocked": int(e.xg_missing.sum()),
        "unblocked": int(e.unb.sum()),
        "rebound_share_of_attempts": float(e.reb[e.att].mean()),
        "rush_share_of_attempts": float(e.rush[e.att].mean()),
        "individual_rows_not_in_skater_base": orphan_n,
        "ev5_events_with_5v5_onice_share": float(Q.ev5_both5.sum() / Q.ev5_events.sum()),
        "shift_state_equals_pbp_state_at_attempts": float(Q.state_agree.sum() / Q.att.sum()),
        "toi_shift_sum_vs_roster_toi_exact_share": float(
            (np.abs(toi_all[has] - df.toi_s[has]) <= 1).mean()),
        "toi_shift_sum_vs_roster_toi_within60_share": float(
            (np.abs(toi_all[has] - df.toi_s[has]) <= 60).mean()),
        "goals_ind_vs_onice_ev": [float(df.g_ev.sum()), float(df.on_gf_ev.sum() / 5)],
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(qa, open(QA_OUT, "w"), indent=1)
    print(json.dumps(qa, indent=1))


if __name__ == "__main__":
    main()
