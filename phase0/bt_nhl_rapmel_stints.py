"""bt_nhl_rapmel STEP 1 -- DEV-only 5v5 stints for walk-forward stint RAPM.

Breakthrough program (documents/breakthrough_program_prereg_2026_09_24.md),
candidate nhl_rapmel. Replicates nhl_rapm2.build_stints with keep = DEV
regular-season game ids. NEVER calls nhl_rapm2.main() / build_stints() (those
fit the TEST-era window and overwrite data/nhl_rapm2_ratings.json); only the
pure helper stints_for_game and the PROX_WINDOW constant are imported.

Replication notes (deliberate, documented):
  * shot / goal side is read from the shot row's own is_home column. nhl_rapm2
    compares shooter_team to the games-file home code as raw strings; in the DEV
    era the shots file spells L.A / N.J / S.J / T.B / ARI(for PHX) while the
    games file says LAK / NJD / SJS / TBL / PHX, so the literal string compare
    would hand every shot of those teams to the other side (56% vs 44% home
    match on DEV). is_home agrees with the TEAM_FIX-normalised comparison on
    100.0% of DEV shots, so this is the intended logic, not a change of it.
  * shift team codes are checked against the games-file codes per game.
  * toi5 = seconds on ice in ALL 5-v-5 personnel intervals (the true 5v5 TOI,
    including intervals < 3 s that the ridge drops).

Stages (each cached):
  split            one pass over nhl_shifts.csv -> per-season DEV shift arrays
  season <s>       stints for one season (run 8 in parallel, background)
  merge            toi5 parts -> data/bt_nhl_rapmel_toi5.csv, S4d tallies

PROTOCOL: every file is filtered to game_id < 2018000000 AND to the DEV
regular-season id set on read. Nothing here is scored.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
from nhl_depth_eval import dev_games  # noqa: E402  (asserts season <= DEV_END)
from nhl_glicko2_eval import DEV_END, TEST_START  # noqa: E402
from nhl_rapm2 import PROX_WINDOW, stints_for_game  # noqa: E402  (pure helpers only)

MAX_GID = 2018000000
PFX = "data/bt_nhl_rapmel_"
assert DEV_END == 20172018 and TEST_START == 20182019


def dev_meta():
    games = dev_games()
    meta = {g["game_id"]: (g["date"], g["home"], g["away"], g["season"]) for g in games}
    assert max(meta) < MAX_GID and max(m[3] for m in meta.values()) <= DEV_END
    return meta


def dev_goalie_ids(ids):
    gx = pd.read_csv("data/nhl_goalie_xg.csv", usecols=["nhl_game_id", "goalie_id"])
    gx = gx[(gx.nhl_game_id < MAX_GID) & gx.nhl_game_id.isin(ids)]
    return set(int(x) for x in gx.goalie_id.unique())


# ------------------------------------------------------------------- split ---
def stage_split():
    meta = dev_meta()
    ids = np.array(sorted(meta), dtype=np.int64)
    season_of = {g: m[3] for g, m in meta.items()}
    teams = {}
    parts = defaultdict(list)
    t0 = time.time()
    for ch in pd.read_csv("data/nhl_shifts.csv", chunksize=2_000_000,
                          dtype={"team": str}, low_memory=False):
        ch = ch[pd.to_numeric(ch.game_id, errors="coerce").notna()]
        ch["game_id"] = pd.to_numeric(ch.game_id).astype(np.int64)
        ch = ch[(ch.game_id < MAX_GID) & ch.game_id.isin(ids)].copy()
        if not len(ch):
            continue
        for c in ("player_id", "period", "start_s", "end_s"):
            ch[c] = pd.to_numeric(ch[c]).astype(np.int64)
        for t in ch.team.unique():
            teams.setdefault(t, len(teams))
        tc = ch.team.map(teams).astype(np.int64).to_numpy()
        arr = np.column_stack([ch.game_id.to_numpy(np.int64),
                               ch.player_id.to_numpy(np.int64), tc,
                               ch.period.to_numpy(np.int64),
                               ch.start_s.to_numpy(np.int64),
                               ch.end_s.to_numpy(np.int64)])
        ss = np.array([season_of[g] for g in arr[:, 0]])
        for s in np.unique(ss):
            parts[int(s)].append(arr[ss == s])
        print(f"  chunk done {time.time()-t0:.0f}s", flush=True)
    for s, lst in parts.items():
        a = np.concatenate(lst)
        assert a[:, 0].max() < MAX_GID
        np.save(f"{PFX}sh_{s}.npy", a)
        print(f"season {s}: {len(a):,} DEV shift rows, {len(np.unique(a[:,0])):,} games")
    json.dump({"teams": teams}, open(f"{PFX}sh_teams.json", "w"))


# ------------------------------------------------------------------ season ---
def stage_season(season):
    meta = dev_meta()
    keep = {g for g, m in meta.items() if m[3] == season}
    assert keep and max(keep) < MAX_GID
    goalies = dev_goalie_ids(set(meta))
    tmap = json.load(open(f"{PFX}sh_teams.json"))["teams"]
    inv = {v: k for k, v in tmap.items()}

    # shots, parsed exactly as nhl_rapm2.build_stints (side from is_home)
    shots = defaultdict(lambda: defaultdict(lambda: ([], [], [], [])))
    goals = defaultdict(list)
    all5 = 0.0
    with open("data/nhl_shots.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        assert hdr[:10] == ["nhl_game_id", "period", "per_sec", "shooter_team", "is_home",
                            "xg", "goal", "home_sk", "away_sk", "goalie_id"], hdr
        for r in rd:
            gid = int(r[0])
            if gid >= MAX_GID or gid not in keep:
                continue
            sec, xg, is5 = int(r[2]), float(r[5]), r[7] == "5" and r[8] == "5"
            home_shot = r[4] == "1"
            sec_l, hom_l, xg5_l, xga_l = shots[gid][int(r[1])]
            sec_l.append(sec)
            hom_l.append(home_shot)
            xg5_l.append(xg if is5 else 0.0)
            xga_l.append(xg)
            if is5:
                all5 += xg
            if r[6] == "1":
                goals[gid].append((int(r[1]), sec, home_shot))

    sh = np.load(f"{PFX}sh_{season}.npy")
    order = np.argsort(sh[:, 0], kind="stable")
    sh = sh[order]
    gids_sh, starts = np.unique(sh[:, 0], return_index=True)
    bounds = dict(zip(gids_sh.tolist(), zip(starts.tolist(),
                                          list(starts[1:].tolist()) + [len(sh)])))

    H, A, DUR, XGF, XGA, SDF, PRX, GID, DT = [], [], [], [], [], [], [], [], []
    toi5 = defaultdict(float)
    ngames = 0
    leak_all = leak_5 = 0.0
    all5_withshifts = 0.0
    code_mismatch = 0
    t0 = time.time()
    for gid in sorted(shots):
        if gid not in bounds:
            continue
        a, b = bounds[gid]
        blk = sh[a:b]
        gshifts = [(int(r[1]), inv[int(r[2])], int(r[3]), int(r[4]), int(r[5])) for r in blk]
        date, home, away = meta[gid][0], meta[gid][1], meta[gid][2]
        steams = {t for (_, t, _, _, _) in gshifts}
        if not ({home, away} <= steams):
            code_mismatch += 1
        dint = int(date.replace("-", ""))
        per_map = shots[gid]
        prep = {}
        for pe, (sec_l, hom_l, xg5_l, xga_l) in per_map.items():
            o = sorted(range(len(sec_l)), key=lambda i: sec_l[i])
            secs = [sec_l[i] for i in o]
            hom = [hom_l[i] for i in o]
            xg5 = [xg5_l[i] for i in o]
            all5_withshifts += sum(xg5)
            prep[pe] = (secs, hom,
                        np.cumsum([0.0] + [x if h else 0.0 for x, h in zip(xg5, hom)]),
                        np.cumsum([0.0] + [x if not h else 0.0 for x, h in zip(xg5, hom)]),
                        np.cumsum([0.0] + [xga_l[i] for i in o]))
        gg = sorted(goals.get(gid, ()))
        ngames += 1
        for pe, t0_, t1, on in stints_for_game(gshifts, goalies):
            hh, aa = on.get(home, []), on.get(away, [])
            if len(hh) != 5 or len(aa) != 5:
                continue
            dur = t1 - t0_
            for p in hh:
                toi5[(gid, 1, p)] += dur
            for p in aa:
                toi5[(gid, 0, p)] += dur
            if dur < 3:
                continue
            secs, hom, ch, ca, call = prep.get(
                pe, ([], [], np.zeros(1), np.zeros(1), np.zeros(1)))
            lo = bisect.bisect_left(secs, t0_)
            hi = bisect.bisect_left(secs, t1)
            leak_all += float(call[hi] - call[lo])
            leak_5 += float(ch[hi] - ch[lo]) + float(ca[hi] - ca[lo])
            prox = 0
            for k in range(bisect.bisect_left(secs, t0_ - PROX_WINDOW), lo):
                prox = 1 if hom[k] else -1
            gh = ga = 0
            for (gp, gs, gt_home) in gg:
                if (gp, gs) >= (pe, t0_):
                    break
                if gt_home:
                    gh += 1
                else:
                    ga += 1
            H.append(hh); A.append(aa); DUR.append(dur)
            XGF.append(float(ch[hi] - ch[lo])); XGA.append(float(ca[hi] - ca[lo]))
            PRX.append(prox); SDF.append(max(-3, min(3, gh - ga)))
            GID.append(gid); DT.append(dint)
        if ngames % 200 == 0:
            print(f"  {season}: {ngames} games {len(H):,} stints {time.time()-t0:.0f}s",
                  flush=True)

    out = {"hi": np.asarray(H, dtype=np.int64), "ai": np.asarray(A, dtype=np.int64),
           "dur": np.asarray(DUR, dtype=float), "xgf": np.asarray(XGF, dtype=float),
           "xga": np.asarray(XGA, dtype=float), "sdiff": np.asarray(SDF, dtype=np.int8),
           "prox": np.asarray(PRX, dtype=np.int8), "gid": np.asarray(GID, dtype=np.int64),
           "date": np.asarray(DT, dtype=np.int64)}
    assert out["gid"].max() < MAX_GID
    for k, v in out.items():
        np.save(f"{PFX}{season}_{k}.npy", v)
    rows = [(g, s, p, v) for (g, s, p), v in toi5.items()]
    pd.DataFrame(rows, columns=["gid", "side", "pid", "sec5"]).to_csv(
        f"{PFX}toi5_{season}.csv", index=False)
    tally = {"season": season, "games_with_shots": len(shots), "games_stinted": ngames,
             "games_in_keep": len(keep), "stints": len(H),
             "xg_in_kept_stints_all_strength": leak_all,
             "xg_in_kept_stints_strict5v5": leak_5,
             "all_strict5v5_xg_season": all5,
             "all_strict5v5_xg_games_with_shifts": all5_withshifts,
             "team_code_mismatch_games": code_mismatch,
             "seconds": round(time.time() - t0, 1)}
    json.dump(tally, open(f"{PFX}stints_{season}.json", "w"), indent=1)
    print(json.dumps(tally), flush=True)


def stage_merge():
    meta = dev_meta()
    seasons = sorted({m[3] for m in meta.values()})
    parts = [pd.read_csv(f"{PFX}toi5_{s}.csv") for s in seasons]
    t = pd.concat(parts, ignore_index=True)
    assert t.gid.max() < MAX_GID
    t.to_csv(f"{PFX}toi5.csv", index=False)
    tallies = [json.load(open(f"{PFX}stints_{s}.json")) for s in seasons]
    num = sum(x["xg_in_kept_stints_strict5v5"] for x in tallies)
    den = sum(x["all_strict5v5_xg_season"] for x in tallies)
    lk = sum(x["xg_in_kept_stints_all_strength"] for x in tallies)
    summ = {"per_season": tallies, "S4d_kept_over_all_strict5v5": num / den,
            "strength_leak_dropped_share": 1 - num / lk,
            "toi5_rows": len(t)}
    json.dump(summ, open(f"{PFX}stints_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_season"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "season", "merge"])
    ap.add_argument("season", nargs="?", type=int)
    a = ap.parse_args()
    if a.stage == "split":
        stage_split()
    elif a.stage == "season":
        stage_season(a.season)
    else:
        stage_merge()
