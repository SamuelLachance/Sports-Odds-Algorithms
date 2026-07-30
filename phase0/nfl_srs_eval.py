"""CBB.py SRS ported to NFL: as-of schedule-adjusted point differential as a blend feature.

power_i = avg_margin_i + avg(opponent power)  ==  ridge Massey solve, computed strictly
as-of (only games earlier in the same season). Knobs selected on DEV 5-fold CV only:
ridge strength + blowout cap on margins. PRE-SET GATE: DEV CV improvement >= 0.0020
over the frozen blend (CV 0.62865) to earn the ONE TEST look. MLB port of the same
script was null (+0.00008); this is the game-margin channel (~EPA rating, +0.0015 n.s.).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, run_elo  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

GATE = 0.0020

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
shipped = json.load(open("data/nfl_qb_replacement.json"))
lg_qb = rep["qb_elo"]["lg_rate"]
SP = shipped["params"]

raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
for g, r in zip(games, raw):
    g["margin"] = int(r["home_score"]) - int(r["away_score"])

# ---- frozen base signals ----
pe, y = run_elo(games, score_from=1999, **base["params"])
m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=SP["delta"],
          decay=SP["decay"], prior_db=SP["prior_db"], season_decay=SP["season_decay"])
qd, prev = np.zeros(len(games)), None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        m.new_season()
    prev = g["season"]
    qd[i] = m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"])
    m.predict(g)
    m.update(g, 0.5, qbw)
eps = 1e-12
lgt = np.log(np.clip(pe, eps, 1 - eps) / np.clip(1 - pe, eps, 1 - eps))
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))
test = (seasons >= TEST_YEARS.start) & (seasons <= TEST_YEARS.stop - 1)
BASE = np.column_stack([lgt, qd])


def srs_feature(ridge, cap):
    """As-of ridge-Massey SRS diff per game (within-season)."""
    out = np.zeros(len(games))
    by_season = defaultdict(list)
    for i, g in enumerate(games):
        by_season[g["season"]].append(i)
    for s, idxs in by_season.items():
        teams = sorted({t for i in idxs for t in (games[i]["home"], games[i]["away"])})
        ti = {t: j for j, t in enumerate(teams)}
        n = len(teams)
        N = np.zeros(n); A = np.zeros((n, n)); marg = np.zeros(n)
        cur_date, r = None, np.zeros(n)
        for i in idxs:                                # idxs already date-ordered
            g = games[i]
            if g["date"] != cur_date:
                Mm = np.diag(N) - A + ridge * np.eye(n)
                r = np.linalg.solve(Mm, marg)
                cur_date = g["date"]
            out[i] = r[ti[g["home"]]] - r[ti[g["away"]]]
            mg = max(-cap, min(cap, g["margin"]))
            ih, ia = ti[g["home"]], ti[g["away"]]
            N[ih] += 1; N[ia] += 1
            A[ih, ia] += 1; A[ia, ih] += 1
            marg[ih] += mg; marg[ia] -= mg
    return out


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def cv_ll(X):
    idx = np.where(dev)[0]
    scores = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fit = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
        scores[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return scores.mean()


base_cv = cv_ll(BASE)
print(f"frozen blend DEV CV {base_cv:.5f}   (TEST-look gate: <= {base_cv - GATE:.5f})\n")
best = None
for ridge in (0.5, 1.0, 2.0, 4.0, 8.0):
    for cap in (14, 21, 28, 99):
        f = srs_feature(ridge, cap)
        s = cv_ll(np.column_stack([BASE, f]))
        tag = ""
        if best is None or s < best[0]:
            best = (s, ridge, cap); tag = "  <-- best"
        print(f"  ridge {ridge:>4}  cap {cap:>3}   CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, ridge, cap = best
print(f"\nbest: ridge={ridge} cap={cap}  CV {s:.5f} ({s-base_cv:+.5f})")
if s > base_cv - GATE:
    print(f"GATE NOT MET (needs -{GATE:.4f}) -> no TEST look; frozen model stands.")
    json.dump({"verdict": "gate_not_met", "dev_cv_base": round(base_cv, 5),
               "dev_cv": round(s, 5), "ridge": ridge, "cap": cap},
              open("data/nfl_srs.json", "w"), indent=1)
    sys.exit(0)

f = srs_feature(ridge, cap)
Xw = np.column_stack([BASE, f])
fit = dev & (y != 0.5)
cb = LogisticRegression(C=1e6, max_iter=3000).fit(BASE[fit], y[fit])
cw = LogisticRegression(C=1e6, max_iter=3000).fit(Xw[fit], y[fit])
pb = cb.predict_proba(BASE[test])[:, 1]
pw = cw.predict_proba(Xw[test])[:, 1]
yt = y[test]
d = llv(yt, pb) - llv(yt, pw)
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
print(f"\nTEST 2016-2025 (n={n}, ONE look):")
print(f"  frozen blend LL {llv(yt, pb).mean():.5f}")
print(f"  + SRS        LL {llv(yt, pw).mean():.5f}")
print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
json.dump({"ridge": ridge, "cap": cap, "dev_cv_base": round(base_cv, 5),
           "dev_cv": round(s, 5), "test_base": round(float(llv(yt, pb).mean()), 5),
           "test_new": round(float(llv(yt, pw).mean()), 5),
           "delta_ll": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "verdict": sig},
          open("data/nfl_srs.json", "w"), indent=1)
