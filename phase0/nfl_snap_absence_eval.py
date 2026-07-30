"""Snap-count absence features — SECONDARY PROTOCOL (locked before results):
warm 2013, DEV 2014-2020 (5-fold CV screen, gate -0.0020), TEST 2021-2025 once if met.

Exact participation for EVERYONE incl. OL/defense. Per player: EWMA of snap share.
Expected starters = roster players with EWMA share >= thresh, by unit:
  OL (T/G/C/...), DEF (front+secondary), SKILL (RB/WR/TE/FB, QB excluded).
Absent = expected starter with NO row in this game's snap table (= inactive; the
pre-kickoff inactives list at serve time). Features: unit missing-share diffs.
Base = current 8-feature model, blend refit on secondary DEV.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

src = open("phase0/nfl_big_test.py").read()
exec(src.split("# ---------- variants, DEV CV selection ----------")[0])  # noqa: S102

XBASE = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts])
GATE = 0.0020
OL = {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}
DEF = {"DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "CB", "S", "FS", "SS", "DB", "EDGE", "DL"}
SKILL = {"RB", "WR", "TE", "FB", "HB"}

# snap tables: (gid, franchise_team) -> {pid: (pos, off_pct, def_pct)}
snaps = defaultdict(dict)
for yr in range(2013, 2026):
    for r in csv.DictReader(open(f"data/snap_{yr}.csv")):
        t = PBP_FIX.get(r["team"], r["team"])
        try:
            op, dp = float(r["offense_pct"] or 0), float(r["defense_pct"] or 0)
        except ValueError:
            continue
        snaps[(r["game_id"], t)][r["pfr_player_id"]] = (r["position"], op, dp)
print(f"snap tables for {len(snaps)} team-games")

seasons_arr = seasons
warm = 2013
dev2 = (seasons_arr >= 2014) & (seasons_arr <= 2020)
test2 = (seasons_arr >= 2021) & (seasons_arr <= 2025)
print(f"secondary split: DEV {int(dev2.sum())}  TEST {int(test2.sum())}")


def absence_features(decay, thresh):
    share = {}                    # pid -> [ewma_share, weight]  (unit-appropriate pct)
    pos_of, team_of = {}, {}
    roster = defaultdict(set)
    f_ol = np.zeros(len(games)); f_def = np.zeros(len(games)); f_sk = np.zeros(len(games))

    def unit_missing(team, gid):
        tbl = snaps.get((gid, team))
        if tbl is None:
            return 0.0, 0.0, 0.0
        m = [0.0, 0.0, 0.0]
        for pid in roster.get(team, ()):
            st = share.get(pid)
            if not st or st[1] <= 0:
                continue
            s = st[0] / st[1]
            if s < thresh or pid in tbl:
                continue
            p = pos_of.get(pid, "")
            if p in OL:
                m[0] += s
            elif p in DEF:
                m[1] += s
            elif p in SKILL:
                m[2] += s
        return m[0], m[1], m[2]

    for i, g in enumerate(games):
        if g["season"] < warm:
            continue
        h, a = g["home"], g["away"]
        oh = unit_missing(h, g["gid"]); oa = unit_missing(a, g["gid"])
        f_ol[i] = oh[0] - oa[0]
        f_def[i] = oh[1] - oa[1]
        f_sk[i] = oh[2] - oa[2]
        for team in (h, a):
            tbl = snaps.get((g["gid"], team))
            if not tbl:
                continue
            for pid, (pos, op, dp) in tbl.items():
                pct = dp if pos in DEF else op
                st = share.setdefault(pid, [0.0, 0.0])
                st[0] = decay * st[0] + pct
                st[1] = decay * st[1] + 1.0
                pos_of[pid] = pos
                if team_of.get(pid) != team:
                    if team_of.get(pid) in roster:
                        roster[team_of[pid]].discard(pid)
                    team_of[pid] = team
                    roster[team].add(pid)
    return f_ol, f_def, f_sk


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def cv_ll(X, mask):
    idx = np.where(mask)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fitm = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=5000).fit(X[fitm], y[fitm])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()


base_cv = cv_ll(XBASE, dev2)
print(f"current model, secondary DEV CV {base_cv:.5f}   (gate: <= {base_cv - GATE:.5f})\n")

rng = np.random.default_rng(20260723)
best = {"ol": None, "def": None, "skill": None, "all": None}
t0 = time.time()
for i in range(60):
    cand = dict(decay=float(rng.uniform(0.70, 0.98)),
                thresh=float(rng.uniform(0.35, 0.75)))
    f_ol, f_def, f_sk = absence_features(**cand)
    for kind, cols in (("ol", [f_ol]), ("def", [f_def]), ("skill", [f_sk]),
                       ("all", [f_ol, f_def, f_sk])):
        s = cv_ll(np.column_stack([XBASE] + cols), dev2)
        if best[kind] is None or s < best[kind][0]:
            best[kind] = (s, cand)
    if (i + 1) % 15 == 0:
        b = min(v[0] for v in best.values())
        print(f"  ...{i+1}/60  best CV {b:.5f} ({b-base_cv:+.5f})  ({time.time()-t0:.0f}s)",
              flush=True)

print()
for kind, (s, cand) in best.items():
    print(f"  {kind:<6} CV {s:.5f} ({s-base_cv:+.5f})  "
          f"{({k: round(v, 3) for k, v in cand.items()})}")
winner = min(best, key=lambda k: best[k][0])
s, cand = best[winner]
gate_met = s <= base_cv - GATE
print(f"\nDEV verdict: {winner} ({s-base_cv:+.5f}) -> "
      f"{'GATE MET, taking the ONE secondary-TEST look' if gate_met else 'gate not met, parked'}")
json.dump({"winner": winner, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "params": cand, "gate_met": bool(gate_met)},
          open("data/nfl_snap_absence.json", "w"), indent=1)

if gate_met:
    f_ol, f_def, f_sk = absence_features(**cand)
    cols = {"ol": [f_ol], "def": [f_def], "skill": [f_sk], "all": [f_ol, f_def, f_sk]}[winner]
    Xw = np.column_stack([XBASE] + cols)
    fitm = dev2 & (y != 0.5)
    cb = LogisticRegression(C=1e6, max_iter=5000).fit(XBASE[fitm], y[fitm])
    cw = LogisticRegression(C=1e6, max_iter=5000).fit(Xw[fitm], y[fitm])
    pb = cb.predict_proba(XBASE[test2])[:, 1]
    pw = cw.predict_proba(Xw[test2])[:, 1]
    yt = y[test2]
    d = llv(yt, pb) - llv(yt, pw)
    rng2 = np.random.default_rng(7)
    n = len(d)
    bs = d[rng2.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
    print(f"\nsecondary TEST 2021-2025 (n={n}, ONE look):")
    print(f"  current model  LL {llv(yt, pb).mean():.5f}")
    print(f"  + snap absence LL {llv(yt, pw).mean():.5f}")
    print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
