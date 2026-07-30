"""Sample-size floor for roster_quality: players below n_eff contribute NOTHING.

Samuel's rule: too-low sample -> don't count the rating when predicting games.
(Distinct from rep-centering, which PENALIZED unknowns and was SIG worse; this makes
them SILENT.) Grid the floor T on the primary-metric effective weight; DEV CV selects,
ONE TEST compare vs current model (T~1, i.e. nearly everyone counts). Keep if better.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

full_src = open("phase0/nfl_player_rating_system.py", encoding="utf-8").read()
exec(full_src.split("# pass 1")[0])  # noqa: S102
exec("#" + full_src.split("# pass 1")[1].split("# ---------------- screens")[0])  # noqa: S102
# gives: Z tables, current f_rq (floor ~1) + all base features

PRIMARY = {"QB": "dak", "SK": "eff", "DL": "rush", "LB": "rush", "DB": "ball"}
HAND = {"QB": [("dak", 0.50), ("negsk", 0.25), ("repa", 0.25)],
        "SK": [("eff", 0.40), ("yacr", 0.15), ("wopr", 0.45)],
        "DL": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)],
        "LB": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)],
        "DB": [("rush", 0.45), ("ball", 0.35), ("tak", 0.20)]}


def make_rate_floor(T):
    def rate(pid):
        st = STATE.get(pid)
        if st is None:
            return None
        b = st.get("bucket")
        if b not in Z:
            return None
        if st.get(PRIMARY[b] + "_w", 0.0) < T:
            return None                    # too little evidence -> SILENT
        tot, any_ = 0.0, False
        for k, wt in HAND[b]:
            w = st.get(k + "_w", 0.0)
            if w <= 1 or k not in Z[b]:
                continue
            mu, sd = Z[b][k]
            v = (st.get(k, 0.0) + 6.0 * mu) / (w + 6.0)
            tot += wt * (v - mu) / sd
            any_ = True
        return tot if any_ else None
    return rate


def build_rq(rates):
    STATE.clear()
    share2, pos2, team2 = {}, {}, {}
    roster2 = defaultdict(set)
    outs = [np.zeros(len(games)) for _ in rates]
    done2 = set()

    def rq(team, gid, rate):
        tbl = snaps.get((gid, team))
        if tbl is None:
            return 0.0
        tot = 0.0
        for pid in tbl:
            st = share2.get(pid)
            if not st or st[1] <= 0:
                continue
            g_ = pfr2gsis.get(pid)
            r_ = rate(g_) if g_ else None
            if r_ is not None:
                tot += (st[0] / st[1]) * max(-2.0, min(2.0, r_))
        return tot

    for i, g in enumerate(games):
        s, w = g["season"], g["week"]
        if s >= 2013:
            for j, rate in enumerate(rates):
                outs[j][i] = rq(g["home"], g["gid"], rate) - rq(g["away"], g["gid"], rate)
        for team in (g["home"], g["away"]):
            for pid in OFF_TW.get((team, s, w), []) + DEF_TW.get((team, s, w), []):
                if (pid, s, w) in done2:
                    continue
                done2.add((pid, s, w))
                apply_week(pid, s, w)
        for team in (g["home"], g["away"]):
            tbl = snaps.get((g["gid"], team))
            if not tbl:
                continue
            for pid, (pos, op, dp) in tbl.items():
                pct = dp if pos in DEFPOS else op
                st = share2.setdefault(pid, [0.0, 0.0])
                st[0] = snapP["decay"] * st[0] + pct
                st[1] = snapP["decay"] * st[1] + 1.0
                pos2[pid] = pos
                if team2.get(pid) != team:
                    if team2.get(pid) in roster2:
                        roster2[team2.get(pid)].discard(pid)
                    team2[pid] = team
                    roster2[team].add(pid)
    return outs


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def cv_ll(X):
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fitm = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=5000).fit(X[fitm], y[fitm])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()


GRID = [5.0, 10.0, 20.0, 40.0]
outs = build_rq([make_rate_floor(T) for T in GRID])
X_cur = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                         f_ol, f_def, f_sk, f_rq])
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}")
best = None
for T, fr in zip(GRID, outs):
    X = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                         f_ol, f_def, f_sk, fr])
    s = cv_ll(X)
    tag = ""
    if best is None or s < best[0]:
        best = (s, T, fr); tag = "  <-- best"
    print(f"  floor n_eff>={T:>4.0f}   DEV CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, T, fr = best
Xn = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts,
                      f_ol, f_def, f_sk, fr])
fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(Xn[fitm], y[fitm])
p0 = c0.predict_proba(X_cur[test])[:, 1]
p1 = c1.predict_proba(Xn[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"\nmain TEST 2016-2025 (ONE compare):")
print(f"  current (no floor)     LL {llv(yt, p0).mean():.5f}")
print(f"  floor n_eff>={T:.0f}       LL {llv(yt, p1).mean():.5f}")
print(f"  delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  -> "
      f"{'ADOPT floor' if better else 'keep current'}")
json.dump({"grid": GRID, "winner_T": T, "dev_cv_base": round(base_cv, 5),
           "dev_cv": round(s, 5), "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_neff_floor.json", "w"), indent=1)
