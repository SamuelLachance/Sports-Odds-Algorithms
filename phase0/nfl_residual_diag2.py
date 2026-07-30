"""Residual diagnostic #2: what does the CURRENT 8-feature model still misread?

DEV-only out-of-fold calibration slices (bias = actual - predicted home rate, z-score).
Slices are strictly pre-game-known. TEST never touched.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# build all model features exactly as the spec does
src = open("phase0/nfl_big_test.py").read()
exec(src.split("# ---------- variants, DEV CV selection ----------")[0])  # noqa: S102

X = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts])

# extra slice ingredients
raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
for g, r in zip(games, raw):
    g["week"] = int(r["week"])
    g["gtype"] = r["game_type"]
    g["div"] = r["div_game"] == "1"
    g["roof"] = r["roof"]
    g["wind"] = float(r["wind"]) if r["wind"] else None
    g["temp"] = float(r["temp"]) if r["temp"] else None
    g["wday"] = r["weekday"]

# rematch: teams already met this season (pre-game known)
met = defaultdict(int)
rematch = np.zeros(len(games), bool)
prev_season = None
for i, g in enumerate(games):
    if g["season"] != prev_season:
        met.clear()
        prev_season = g["season"]
    key = tuple(sorted((g["home"], g["away"])))
    rematch[i] = met[key] > 0
    met[key] += 1

# QB experience known at game time (EWMA dropbacks from the pedigree walk)
mq = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, lg_rate=lg_qb, rep_delta=pedP["delta"],
           decay=SP["decay"], prior_db=SP["prior_db"], season_decay=SP["season_decay"])
qdb_h, qdb_a, prev = np.zeros(len(games)), np.zeros(len(games)), None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        mq.new_season()
    prev = g["season"]
    qdb_h[i] = mq.Q.get(g["home_qb"], [0, 0])[1]
    qdb_a[i] = mq.Q.get(g["away_qb"], [0, 0])[1]
    mq.predict(g)
    mq.update(g, 0.5, qbw)

# out-of-fold DEV probs for the CURRENT model
p_cv = np.full(len(games), np.nan)
idx = np.where(dev)[0]
for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
    itr, iva = idx[tr], idx[va]
    fitm = itr[y[itr] != 0.5]
    clf = LogisticRegression(C=1e6, max_iter=5000).fit(X[fitm], y[fitm])
    p_cv[iva] = clf.predict_proba(X[iva])[:, 1]

final_wk = np.array([(g["season"] <= 2020 and g["week"] == 17 and g["gtype"] == "REG")
                     or (g["season"] >= 2021 and g["week"] == 18) for g in games])
S = {
    "away QB unproven (<150 db)": qdb_a < 150,
    "home QB unproven (<150 db)": qdb_h < 150,
    "divisional game": np.array([g["div"] for g in games]),
    "rematch (met earlier this yr)": rematch,
    "final reg-season week": final_wk,
    "playoffs": np.array([g["gtype"] != "REG" for g in games]),
    "dome/closed roof": np.array([g["roof"] in ("dome", "closed") for g in games]),
    "outdoor wind>=15": np.array([g["roof"] == "outdoors" and (g["wind"] or 0) >= 15 for g in games]),
    "outdoor temp<=32F": np.array([g["roof"] == "outdoors" and g["temp"] is not None and g["temp"] <= 32 for g in games]),
    "Thursday game": np.array([g["wday"] == "Thursday" for g in games]),
    "early-kick circadian": f_early > 0,
    "home off bye": np.array([g["hrest"] >= 13 for g in games]),
    "away off bye": np.array([g["arest"] >= 13 for g in games]),
    "big home favorite (p>=.75)": p_cv >= 0.75,
    "big home dog (p<=.25)": p_cv <= 0.25,
    "toss-up (.45-.55)": (p_cv >= 0.45) & (p_cv <= 0.55),
    "week 1-2": np.array([g["week"] <= 2 for g in games]),
}

ll = -(y[dev] * np.log(p_cv[dev]) + (1 - y[dev]) * np.log(1 - p_cv[dev])).mean()
print(f"CURRENT model DEV OOF (n={dev.sum()}, LL {ll:.5f})")
print(f"{'slice':<32}{'n':>6}{'pred':>8}{'actual':>8}{'bias':>8}{'z':>6}")
rows = []
for name, mask in S.items():
    mm = mask & dev
    n = int(mm.sum())
    if n < 30:
        continue
    pm, am = p_cv[mm].mean(), y[mm].mean()
    sd = np.sqrt((p_cv[mm] * (1 - p_cv[mm])).sum()) / n
    z = (am - pm) / sd if sd > 0 else 0.0
    rows.append((abs(z), name, n, pm, am, z))
for _, name, n, pm, am, z in sorted(rows, reverse=True):
    flag = "  <-- misread" if abs(z) >= 2 else ""
    print(f"{name:<32}{n:>6}{pm:>8.3f}{am:>8.3f}{am-pm:>+8.3f}{z:>+6.1f}{flag}")
