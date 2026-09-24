"""pv_nhl_defense_onice STEP 1 -- DEV 5v5 stints carrying GOALS as well as xG.

Player-value program, NHL, component `defense_onice`: the defensive side that has
no individual event. The stint table here is the input of the walk-forward on-ice
RAPM for goals-against / xG-against / shot-attempts-against
(phase0/pv_nhl_defense_onice_fit.py).

Why a new stint table: data/bt_nhl_rapmel_<season>_*.npy carries 5v5 xG only, and
no period/clock, so goals and faceoff zone starts cannot be attached to it.

Inputs (all read-only, all DEV, regular season):
  data/bt_nhl_rapmel_sh_<season>.npy   shift rows (gid, pid, team idx, period,
                                       start_s, end_s), DEV regular season only
  data/pv_nhl_rosters.csv              side (H/A) and position per dressed player
  data/pv_nhl_events.parquet           shot attempts, goals, faceoffs (pv_nhl_io,
                                       DEV guard asserted)
  data/nhl_games.csv                   game dates (DEV rows only)

Missing xG (amendment 1, data/pv_nhl_serve_prereg.json): an unblocked attempt with
no MoneyPuck xG is filled by phase0/pv_nhl_xg_impute.py -- the mean xG of its type
over known-xG shots dated STRICTLY BEFORE its date (running, across seasons). The
earlier whole-season mean leaked shots after the in-season RAPM fit cutoffs.

Interval convention (checked in the QA block, see `qa` in the output json):
  personnel intervals are the elementary pieces between consecutive shift
  breakpoints of a period; a SHOT/GOAL at clock t belongs to the interval with
  t0 < t <= t1 (shifts that end on a goal end AT the goal second, the players
  coming on for the ensuing faceoff start AT it); a FACEOFF at t belongs to the
  interval with t0 <= t < t1, and it is the interval's zone start when t == t0.
  An interval is kept when each side has exactly 5 non-goalie skaters and 1
  goalie on the ice. An event is counted only when its own situationCode is
  1551 (both sources must agree it is 5v5).

Output per season: data/pv_nhl_defense_onice_st_<season>.npz with, per kept
interval: gid, date, period, t0, t1, dur, hs[5], as_[5], hg, ag, sd (home-minus-
away goals before t0, clipped +/-3), zs (home-perspective zone of a faceoff at t0:
0 on-the-fly, 1 home O-zone, 2 home D-zone, 3 neutral), and for each side
(h*/a* = that side ATTACKING): g (goals), xg (MoneyPuck xG, unblocked), c (all
attempts), f (unblocked attempts), s (shots on goal incl. goals).
Plus data/pv_nhl_defense_onice_toi5_<season>.csv: gid, pid, side, sec5 (all kept
5v5 interval seconds).

PROTOCOL: DEV seasons only (2010-11 .. 2017-18, gid < 2018000000). Nothing here
is scored against game outcomes.

Usage: python phase0/pv_nhl_defense_onice_stints.py [season ...]   (default all 8,
       run in parallel processes)
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
import pv_nhl_xg_impute as XI  # noqa: E402

SEASONS = [20102011, 20112012, 20122013, 20132014, 20142015, 20152016, 20162017,
           20172018]
OUT = "data/pv_nhl_defense_onice_"
SHOTS = ("goal", "shot-on-goal", "missed-shot", "blocked-shot")


def game_dates():
    g = pd.read_csv("data/nhl_games.csv", usecols=["game_id", "date", "season", "type"])
    g = g[(g.game_id < DEV_MAX_GID) & (g.type == 2)]
    return dict(zip(g.game_id.astype(np.int64), g.date.str.replace("-", "").astype(int)))


def build_season(season):
    assert season in SEASONS
    t_start = time.time()
    dates = game_dates()
    sh = np.load(f"data/bt_nhl_rapmel_sh_{season}.npy")
    assert sh[:, 0].max() < DEV_MAX_GID
    ro = load_rosters()
    ro = ro[(ro.season == season) & (ro.gtype == 2)]
    side_of = {(int(g), int(p)): (1 if s == "H" else 0, pos)
               for g, p, s, pos in zip(ro.gid, ro.pid, ro.side, ro.pos)}
    ev = load_events(columns=["season", "gtype", "period", "ptype", "per_sec", "ev",
                              "is_home", "xg", "sit", "zone"], seasons={season})
    ev = ev[(ev.gtype == 2) & (ev.ptype != "SO")]
    assert ev.gid.max() < DEV_MAX_GID
    ev = ev[ev.ev.isin(SHOTS + ("faceoff",))].copy()
    # fill the few unmatched unblocked xG values WALK-FORWARD (amendment 1): the mean
    # xG of the type over known-xG shots dated strictly before the shot's date, one
    # shared implementation (pv_nhl_xg_impute) for DEV and the full wrapper. The
    # table is built with this module's `load_events` (the full wrapper patches it).
    unb = ev.ev.isin(XI.UNBLOCKED)
    xg_table = XI.prior_mean_table(load_events, dates, season)
    n_fill, n_fb = XI.fill_missing_xg(ev, dates, xg_table)
    ev.loc[~unb, "xg"] = 0.0
    evg = {g: d for g, d in ev.groupby("gid", sort=True)}

    order = np.lexsort((sh[:, 4], sh[:, 3], sh[:, 0]))
    sh = sh[order]
    gids, starts = np.unique(sh[:, 0], return_index=True)
    ends = list(starts[1:]) + [len(sh)]

    cols = {k: [] for k in ("gid", "date", "period", "t0", "t1", "dur", "hs", "as_",
                            "hg", "ag", "sd", "zs", "hg_", "hxg", "hc", "hf", "hs_",
                            "ag_", "axg", "ac", "af", "as2")}
    toi = {}
    qa = dict(games=0, games_no_events=0, shift_rows_no_roster=0, dup_player_intervals=0,
              ev1551=0, ev1551_in_kept=0, ev_in_kept_not1551=0, goals1551=0,
              goals1551_in_kept=0, goals_all_nonSO=0, kept_seconds=0.0,
              faceoff_starts=0, kept_intervals=0, alt_conv_ev1551_in_kept=0,
              xg_filled=n_fill, xg_fill_fallback=n_fb)
    for gid, a, b in zip(gids.tolist(), starts.tolist(), ends):
        if gid not in dates:
            continue
        blk = sh[a:b]
        e = evg.get(gid)
        qa["games"] += 1
        if e is None:
            qa["games_no_events"] += 1
            continue
        pid = blk[:, 1]
        sd_info = [side_of.get((gid, int(p))) for p in pid]
        okr = np.array([x is not None for x in sd_info])
        qa["shift_rows_no_roster"] += int((~okr).sum())
        blk = blk[okr]
        sd_info = [x for x in sd_info if x is not None]
        home = np.array([x[0] == 1 for x in sd_info])
        isg = np.array([x[1] == "G" for x in sd_info])
        good = blk[:, 5] > blk[:, 4]
        blk, home, isg = blk[good], home[good], isg[good]
        goals = e[e.ev == "goal"]
        qa["goals_all_nonSO"] += len(goals)
        gkey = goals.period.to_numpy() * 10000 + goals.per_sec.to_numpy()
        gk_h = np.sort(gkey[goals.is_home.to_numpy() == 1])
        gk_a = np.sort(gkey[goals.is_home.to_numpy() == 0])
        for pe in np.unique(blk[:, 3]):
            m = blk[:, 3] == pe
            rows = blk[m]
            hm, gm = home[m], isg[m]
            bps = np.unique(np.concatenate([rows[:, 4], rows[:, 5]]))
            if len(bps) < 2:
                continue
            mids = (bps[:-1] + bps[1:]) / 2.0
            on = (rows[:, 4][:, None] <= mids[None, :]) & (mids[None, :] < rows[:, 5][:, None])
            hsk = on[hm & ~gm]
            ask = on[~hm & ~gm]
            nhs, nas = hsk.sum(0), ask.sum(0)
            nhg, nag = on[hm & gm].sum(0), on[~hm & gm].sum(0)
            keep = (nhs == 5) & (nas == 5) & (nhg == 1) & (nag == 1)
            J = len(mids)
            pe_ev = e[e.period == pe]
            shots = pe_ev[pe_ev.ev.isin(SHOTS)]
            st = shots.per_sec.to_numpy()
            j_ev = np.searchsorted(bps, st, side="left") - 1
            j_alt = np.searchsorted(bps, st, side="right") - 1
            valid = (j_ev >= 0) & (j_ev < J)
            is5 = (shots.sit.to_numpy() == "1551")
            qa["ev1551"] += int(is5.sum())
            kept_ev = np.zeros(len(st), bool)
            kept_ev[valid] = keep[j_ev[valid]]
            va = (j_alt >= 0) & (j_alt < J)
            kept_alt = np.zeros(len(st), bool)
            kept_alt[va] = keep[j_alt[va]]
            qa["ev1551_in_kept"] += int((is5 & kept_ev).sum())
            qa["alt_conv_ev1551_in_kept"] += int((is5 & kept_alt).sum())
            qa["ev_in_kept_not1551"] += int((~is5 & kept_ev).sum())
            isgoal = shots.ev.to_numpy() == "goal"
            qa["goals1551"] += int((is5 & isgoal).sum())
            qa["goals1551_in_kept"] += int((is5 & isgoal & kept_ev).sum())
            use = is5 & kept_ev
            jj = j_ev[use]
            ih = shots.is_home.to_numpy()[use] == 1
            typ = shots.ev.to_numpy()[use]
            xg = shots.xg.to_numpy()[use]
            acc = {}
            for side_flag, pre in ((True, "h"), (False, "a")):
                sel = ih == side_flag
                j_ = jj[sel]
                t_ = typ[sel]
                acc[pre + "g"] = np.bincount(j_[t_ == "goal"], minlength=J)
                acc[pre + "xg"] = np.bincount(j_, weights=xg[sel], minlength=J)
                acc[pre + "c"] = np.bincount(j_, minlength=J)
                acc[pre + "f"] = np.bincount(j_[t_ != "blocked-shot"], minlength=J)
                acc[pre + "s"] = np.bincount(j_[(t_ == "goal") | (t_ == "shot-on-goal")],
                                             minlength=J)
            # zone starts: faceoff exactly at an interval start
            fo = pe_ev[pe_ev.ev == "faceoff"]
            zs = np.zeros(J, np.int8)
            if len(fo):
                ft = fo.per_sec.to_numpy()
                jf = np.searchsorted(bps, ft, side="right") - 1
                okf = (jf >= 0) & (jf < J)
                for t_, j_, z_, h_ in zip(ft[okf], jf[okf], fo.zone.to_numpy()[okf],
                                          fo.is_home.to_numpy()[okf]):
                    if t_ != bps[j_]:
                        continue
                    if z_ == "N":
                        zs[j_] = 3
                    elif (z_ == "O") == (h_ == 1):
                        zs[j_] = 1
                    elif z_ in ("O", "D"):
                        zs[j_] = 2
            kj = np.nonzero(keep)[0]
            if not len(kj):
                continue
            hp = rows[hm & ~gm][:, 1]
            ap = rows[~hm & ~gm][:, 1]
            hgp = rows[hm & gm][:, 1]
            agp = rows[~hm & gm][:, 1]
            HS = hp[np.nonzero(hsk[:, kj].T)[1]].reshape(-1, 5)
            AS = ap[np.nonzero(ask[:, kj].T)[1]].reshape(-1, 5)
            HG = hgp[np.nonzero(on[hm & gm][:, kj].T)[1]]
            AG = agp[np.nonzero(on[~hm & gm][:, kj].T)[1]]
            HS.sort(1)
            AS.sort(1)
            dup = ((np.diff(HS, axis=1) == 0).any(1) | (np.diff(AS, axis=1) == 0).any(1))
            qa["dup_player_intervals"] += int(dup.sum())
            ok = ~dup
            kj, HS, AS, HG, AG = kj[ok], HS[ok], AS[ok], HG[ok], AG[ok]
            t0 = bps[kj]
            t1 = bps[kj + 1]
            dur = (t1 - t0).astype(float)
            k0 = pe * 10000 + t0
            sdiff = (np.searchsorted(gk_h, k0, side="right")
                     - np.searchsorted(gk_a, k0, side="right"))
            n = len(kj)
            cols["gid"].append(np.full(n, gid, np.int64))
            cols["date"].append(np.full(n, dates[gid], np.int64))
            cols["period"].append(np.full(n, pe, np.int8))
            cols["t0"].append(t0.astype(np.int16))
            cols["t1"].append(t1.astype(np.int16))
            cols["dur"].append(dur)
            cols["hs"].append(HS)
            cols["as_"].append(AS)
            cols["hg"].append(HG)
            cols["ag"].append(AG)
            cols["sd"].append(np.clip(sdiff, -3, 3).astype(np.int8))
            cols["zs"].append(zs[kj])
            cols["hg_"].append(acc["hg"][kj].astype(np.int16))
            cols["hxg"].append(acc["hxg"][kj])
            cols["hc"].append(acc["hc"][kj].astype(np.int16))
            cols["hf"].append(acc["hf"][kj].astype(np.int16))
            cols["hs_"].append(acc["hs"][kj].astype(np.int16))
            cols["ag_"].append(acc["ag"][kj].astype(np.int16))
            cols["axg"].append(acc["axg"][kj])
            cols["ac"].append(acc["ac"][kj].astype(np.int16))
            cols["af"].append(acc["af"][kj].astype(np.int16))
            cols["as2"].append(acc["as"][kj].astype(np.int16))
            qa["kept_seconds"] += float(dur.sum())
            qa["kept_intervals"] += n
            qa["faceoff_starts"] += int((zs[kj] > 0).sum())
            for P, sflag in ((HS, 1), (AS, 0)):
                for c in range(5):
                    for p_, d_ in zip(P[:, c].tolist(), dur.tolist()):
                        k = (gid, p_, sflag)
                        toi[k] = toi.get(k, 0.0) + d_
    out = {k: np.concatenate(v) for k, v in cols.items()}
    assert out["gid"].max() < DEV_MAX_GID
    np.savez_compressed(f"{OUT}st_{season}.npz",
                        gid=out["gid"], date=out["date"], period=out["period"],
                        t0=out["t0"], t1=out["t1"], dur=out["dur"], hs=out["hs"],
                        as_=out["as_"], hg=out["hg"], ag=out["ag"], sd=out["sd"],
                        zs=out["zs"], h_g=out["hg_"], h_xg=out["hxg"], h_c=out["hc"],
                        h_f=out["hf"], h_s=out["hs_"], a_g=out["ag_"], a_xg=out["axg"],
                        a_c=out["ac"], a_f=out["af"], a_s=out["as2"])
    pd.DataFrame([(g, p, s, v) for (g, p, s), v in toi.items()],
                 columns=["gid", "pid", "side", "sec5"]).to_csv(
        f"{OUT}toi5_{season}.csv", index=False)
    qa["season"] = season
    qa["share_1551_events_in_kept"] = qa["ev1551_in_kept"] / max(1, qa["ev1551"])
    qa["share_1551_events_in_kept_ALT_convention"] = (qa["alt_conv_ev1551_in_kept"]
                                                     / max(1, qa["ev1551"]))
    qa["share_1551_goals_in_kept"] = qa["goals1551_in_kept"] / max(1, qa["goals1551"])
    qa["share_kept_events_not_1551"] = qa["ev_in_kept_not1551"] / max(
        1, qa["ev_in_kept_not1551"] + qa["ev1551_in_kept"])
    qa["kept_goals"] = int(out["hg_"].sum() + out["ag_"].sum())
    qa["kept_xg"] = float(out["hxg"].sum() + out["axg"].sum())
    qa["kept_hours"] = qa["kept_seconds"] / 3600
    qa["seconds"] = round(time.time() - t_start, 1)
    json.dump(qa, open(f"{OUT}st_{season}_qa.json", "w"), indent=1)
    print(json.dumps(qa), flush=True)
    return qa


if __name__ == "__main__":
    ss = [int(x) for x in sys.argv[1:]] or SEASONS
    for s in ss:
        assert s in SEASONS, f"not a DEV season: {s}"
    if len(ss) == 1:
        build_season(ss[0])
    else:
        with Pool(len(ss)) as pool:
            res = pool.map(build_season, ss)
        json.dump(res, open(f"{OUT}st_qa.json", "w"), indent=1)
