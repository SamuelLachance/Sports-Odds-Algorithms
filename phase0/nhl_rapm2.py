"""Stint-based xG/60 RAPM for NHL skaters — a trustworthy player-value display.

The per-shot RAPM (nhl_rapm.py) failed because raw xGoal is always positive, so
every offensive indicator absorbed a baseline and the ridge flattened. The
industry-standard fix (Evolving-Hockey style):

  1. Reconstruct STINTS — intervals of constant on-ice personnel (between any line
     change), keeping only 5v5 to control for strength state; each has a duration.
  2. Per stint, net xG = (home xGF - away xGA within the interval); response is
     the per-60 RATE = net_xg * 3600 / duration, weighted by duration (WLS).
  3. Ridge on home-skater(+1)/away-skater(-1) indicators + a home-ice term. Each
     player's coefficient is their teammate/opponent-adjusted net xG per 60.

Fit on a trailing window for stable current ratings. Display only (the predictive
team-aggregate test came back null, ledger row 8).
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
from nhl_shift_trueskill import load_shifts  # noqa: E402

LAMBDA = 150.0
FIT_SEASONS = {20222023, 20232024, 20242025, 20252026}   # trailing window for current ratings
MIN_TOI = 400.0        # min 5v5 seconds to be rated


def stints_for_game(gshifts, goalies):
    """Yield (period, t0, t1, home_on, away_on) constant-personnel 5v5 intervals.
    gshifts rows: (player, team, period, start_s, end_s). Team codes come from meta."""
    by_period = defaultdict(list)
    for (p, t, pe, s0, s1) in gshifts:
        by_period[pe].append((p, t, s0, s1))
    for pe, rows in by_period.items():
        bps = sorted({s for (_, _, s0, s1) in rows for s in (s0, s1)})
        for i in range(len(bps) - 1):
            t0, t1 = bps[i], bps[i + 1]
            if t1 <= t0:
                continue
            mid = (t0 + t1) / 2.0
            on = defaultdict(list)
            for (p, t, s0, s1) in rows:
                if s0 <= mid < s1 and p not in goalies:
                    on[t].append(p)
            yield pe, t0, t1, on


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
        gid = int(r["nhl_game_id"])
        if meta.get(gid, (None, None, None, 0))[3] in FIT_SEASONS:
            shots[gid].append((int(r["period"]), int(r["per_sec"]), r["shooter_team"], float(r["xg"])))

    # two-way RAPM: each stint -> two rows (each team attacking). A player has an
    # OFFENSE column (cols 0..n) and a DEFENSE column (cols n..2n). Response = the
    # attacking team's xGF per 60. off coef = danger created; def coef = danger
    # allowed (lower/negative = better suppressor).
    rows, cols, wts, ys, srow_w = [], [], [], [], []
    pcount = defaultdict(float)
    pid = {}
    ri = 0
    ngames = 0
    for gid in shots:
        date, home, away, season = meta[gid]
        gshifts = shifts.get(gid)
        if not gshifts:
            continue
        ngames += 1
        gshots = shots[gid]
        for pe, t0, t1, on in stints_for_game(gshifts, goalies):
            H, A = on.get(home, []), on.get(away, [])
            if len(H) != 5 or len(A) != 5:
                continue
            dur = t1 - t0
            if dur < 3:
                continue
            xgf = sum(xg for (p2, s2, st, xg) in gshots if p2 == pe and t0 <= s2 < t1 and st == home)
            xga = sum(xg for (p2, s2, st, xg) in gshots if p2 == pe and t0 <= s2 < t1 and st == away)
            for p in H + A:
                pid.setdefault(p, len(pid))
                pcount[p] += dur
            # row A: home attacks (offense=H, defense=A), response = home xGF/60
            for att, dfn, yv in ((H, A, xgf), (A, H, xga)):
                for p in att:
                    rows.append(ri); cols.append(pid[p]); wts.append(1.0)
                for p in dfn:
                    rows.append(ri); cols.append(pid[p] + 100000); wts.append(1.0)
                ys.append(yv * 3600.0 / dur); srow_w.append(dur); ri += 1
    n = len(pid)
    # remap defense placeholder offset now that n is known
    cols = [c - 100000 + n if c >= 100000 else c for c in cols]
    print(f"{ngames:,} games, {ri:,} stint-sides, {n:,} skaters", flush=True)

    X = sparse.csr_matrix((wts, (rows, cols)), shape=(ri, 2 * n))
    m = Ridge(alpha=LAMBDA, fit_intercept=True, solver="sparse_cg", max_iter=1000)
    m.fit(X, np.array(ys), sample_weight=np.array(srow_w))
    coef = m.coef_
    idx2p = {i: p for p, i in pid.items()}
    lg_off = float(np.mean([coef[i] for i in range(n)]))
    lg_def = float(np.mean([coef[n + i] for i in range(n)]))
    out = {}
    for i in range(n):
        p = idx2p[i]
        if pcount[p] < MIN_TOI:
            continue
        off = float(coef[i] - lg_off)              # xGF/60 above avg (create)
        dfn = float(lg_def - coef[n + i])          # xGA/60 below avg (suppress) -> higher=better
        out[str(p)] = {"off": round(off, 3), "def": round(dfn, 3),
                       "net": round(off + dfn, 3), "toi_min": round(pcount[p] / 60.0, 1)}
    nets = np.array([v["net"] for v in out.values()])
    for v in out.values():
        v["rating"] = round(float(50 + 15 * (v["net"] - nets.mean()) / (nets.std() + 1e-9)), 1)
    json.dump(out, open("data/nhl_rapm2_ratings.json", "w"))
    print(f"{len(out):,} rated -> data/nhl_rapm2_ratings.json")


if __name__ == "__main__":
    main()
