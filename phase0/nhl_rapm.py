"""Regularized adjusted plus-minus (RAPM) for NHL skaters, from shot-level xG.

Where shift-TrueSkill credits the on-ice unit for outcomes (deployment-confounded),
RAPM isolates each player by ridge regression that controls for teammates AND
opponents simultaneously:

    xGoal_shot = sum_{o in offense on-ice} off[o]
               + sum_{d in defense on-ice} def[d]   (def<0 = suppresses danger)
               + home_ice + eps

Ridge (L2) shrinks thinly-sampled players toward 0. Fit LEAK-SAFE per season on a
trailing window of prior seasons; a player's net value = off - def. Emits both a
per-player rating and a per-game pre-game team aggregate (home minus away), for
the combined feature test against the shipped model.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, "phase0")
from nhl_shift_trueskill import index_shifts, load_shifts, on_ice  # noqa: E402

LAMBDA = 300.0            # ridge penalty (NHL RAPM standard territory)
TRAIL = 3                 # trailing seasons per fit
MIN_SHOTS = 200           # min on-ice shots to be rated


def main():
    meta = {}
    for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")):
        meta[int(r["game_id"])] = (r["date"], r["home"], r["away"], int(r["season"]))
    shifts = load_shifts()
    goalies = set()
    for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
        try:
            goalies.add(int(r["goalie_id"]))
        except (KeyError, ValueError):
            pass
    shots = defaultdict(list)
    for r in csv.DictReader(open("data/nhl_shots.csv", encoding="utf-8")):
        shots[int(r["nhl_game_id"])].append(
            (int(r["period"]), int(r["per_sec"]), r["shooter_team"], float(r["xg"])))

    # ---- build the shot-participation table once ----
    recs = []      # (season, off_ids, def_ids, xg, is_home)
    on_ct = defaultdict(int)
    for gid in shots:
        if gid not in meta or gid not in shifts:
            continue
        date, home, away, season = meta[gid]
        gidx = index_shifts(shifts[gid])
        for period, sec, steam, xg in shots[gid]:
            dteam = away if steam == home else home
            O = [p for p in on_ice(gidx, period, sec, steam) if p not in goalies]
            D = [p for p in on_ice(gidx, period, sec, dteam) if p not in goalies]
            if not (2 <= len(O) <= 6 and 2 <= len(D) <= 6):
                continue
            recs.append((season, tuple(O), tuple(D), xg, 1 if steam == home else 0))
            for p in O + D:
                on_ct[p] += 1
    print(f"{len(recs):,} shot records, {len(on_ct):,} skaters", flush=True)

    players = sorted(on_ct)
    pidx = {p: i for i, p in enumerate(players)}
    n = len(players)

    seasons = sorted({r[0] for r in recs})   # 8-digit ids (20102011, 20112012, ...)
    sidx = {s: i for i, s in enumerate(seasons)}
    net_by_season = {}         # eval_season -> {player: net}
    for s_eval in seasons:
        ei = sidx[s_eval]
        prior = set(seasons[max(0, ei - TRAIL):ei])   # up to TRAIL seasons before
        train = [r for r in recs if r[0] in prior]
        if len(train) < 20000:
            continue
        rows, cols, vals, ys = [], [], [], []
        for i, (_, O, D, xg, ih) in enumerate(train):
            for p in O:
                rows.append(i); cols.append(pidx[p]); vals.append(1.0)
            for p in D:
                rows.append(i); cols.append(n + pidx[p]); vals.append(1.0)
            ys.append(xg)
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(train), 2 * n))
        # append a home-ice intercept column
        home_col = sparse.csr_matrix(np.array([[r[4]] for r in train]))
        X = sparse.hstack([X, home_col]).tocsr()
        m = Ridge(alpha=LAMBDA, fit_intercept=True, solver="sparse_cg", max_iter=500)
        m.fit(X, np.array(ys))
        coef = m.coef_
        net = {players[i]: float(coef[i] - coef[n + i]) for i in range(n)}
        net_by_season[s_eval] = net
        print(f"  fit season {s_eval} on {len(train):,} shots (trail {TRAIL})", flush=True)

    # ---- per-game pre-game team aggregate (leak-safe: uses coefs fit on prior seasons) ----
    feat_rows = []
    for gid in sorted(shots):
        if gid not in meta or gid not in shifts:
            continue
        date, home, away, season = meta[gid]
        net = net_by_season.get(season)
        if net is None:
            continue
        gidx = index_shifts(shifts[gid])

        def agg(tm):
            sk = {p for (t, pe), lst in gidx.items() if t == tm
                  for (_, _, p) in lst if p not in goalies}
            vals = [net.get(p, 0.0) for p in sk if on_ct.get(p, 0) >= MIN_SHOTS]
            return sum(vals) if vals else 0.0
        feat_rows.append((gid, round(agg(home), 5), round(agg(away), 5)))

    with open("data/nhl_rapm_features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "home_rapm", "away_rapm"])
        w.writerows(feat_rows)
    print(f"wrote data/nhl_rapm_features.csv: {len(feat_rows):,} games", flush=True)

    # ---- final-season player ratings (display) ----
    last = max(net_by_season)
    net = net_by_season[last]
    rated = {p: net[p] for p in net if on_ct.get(p, 0) >= MIN_SHOTS}
    v = np.array(list(rated.values()))
    z = (v - v.mean()) / (v.std() + 1e-9)
    out = {str(p): {"net_xg": round(rated[p], 4), "rating": round(float(50 + 15 * zz), 1),
                    "n": on_ct[p]} for p, zz in zip(rated, z)}
    json.dump(out, open("data/nhl_rapm_ratings.json", "w"))
    top = sorted(out.items(), key=lambda x: -x[1]["rating"])[:10]
    print(f"{len(out):,} rated -> data/nhl_rapm_ratings.json; top {[p for p,_ in top]}")


if __name__ == "__main__":
    main()
