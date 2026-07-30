"""DEV-only residual diagnostic: where is the blend (elo_logit + qb_diff) mis-calibrated?

Out-of-fold CV probs on DEV (2001-2015), then per-slice bias = actual home win rate
minus mean predicted home prob, with a normal-approx z-score (sum p(1-p) variance).
|z| >~ 2 marks a slice the model systematically misreads. TEST is never touched.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, run_elo  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

games = load_games_qb()
qbw = load_qb_weeks()
base = json.load(open("data/nfl_elo_base.json"))
rep = json.load(open("data/nfl_qb_elo.json"))
P, lg = rep["qb_elo"]["params"], rep["qb_elo"]["lg_rate"]

raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
for g, r in zip(games, raw):
    g.update(week=int(r["week"]), gtype=r["game_type"], div=r["div_game"] == "1",
             roof=r["roof"], wind=float(r["wind"]) if r["wind"] else None,
             temp=float(r["temp"]) if r["temp"] else None,
             hrest=int(r["home_rest"]), arest=int(r["away_rest"]),
             wday=r["weekday"])

pe, y = run_elo(games, score_from=1999, **base["params"])
m = QbElo(k=0.0, hfa=0.0, regress=0.0, beta=1.0, decay=P["decay"],
          prior_db=P["prior_db"], season_decay=P["season_decay"], lg_rate=lg)
qb_diff, qdb_h, qdb_a, prev = [], [], [], None
for g in games:
    if prev is not None and g["season"] != prev:
        m.new_season()
    prev = g["season"]
    qb_diff.append(m.qb_adj(g["home_qb"]) - m.qb_adj(g["away_qb"]))
    qdb_h.append(m.Q.get(g["home_qb"], [0, 0])[1])   # EWMA dropbacks known so far
    qdb_a.append(m.Q.get(g["away_qb"], [0, 0])[1])
    m.predict(g)
    m.update(g, 0.5, qbw)
qb_diff = np.array(qb_diff); qdb_h = np.array(qdb_h); qdb_a = np.array(qdb_a)
eps = 1e-12
lgt = np.log(np.clip(pe, eps, 1 - eps) / np.clip(1 - pe, eps, 1 - eps))
X = np.column_stack([lgt, qb_diff])
seasons = np.array([g["season"] for g in games])
dev = (seasons >= DEV_SCORE_FROM) & (seasons <= max(DEV_YEARS))

# out-of-fold CV probs on DEV
p_cv = np.full(len(games), np.nan)
idx = np.where(dev)[0]
for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
    itr, iva = idx[tr], idx[va]
    fit = itr[y[itr] != 0.5]
    clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
    p_cv[iva] = clf.predict_proba(X[iva])[:, 1]

final_wk = np.array([(g["season"] <= 2020 and g["week"] == 17 and g["gtype"] == "REG")
                     or (g["season"] >= 2021 and g["week"] == 18) for g in games])
S = {
    "final reg-season week": final_wk,
    "playoffs": np.array([g["gtype"] != "REG" for g in games]),
    "divisional game": np.array([g["div"] for g in games]),
    "dome/closed roof": np.array([g["roof"] in ("dome", "closed") for g in games]),
    "outdoor wind>=15mph": np.array([g["roof"] == "outdoors" and (g["wind"] or 0) >= 15 for g in games]),
    "outdoor temp<=32F": np.array([g["roof"] == "outdoors" and g["temp"] is not None and g["temp"] <= 32 for g in games]),
    "home off bye (rest>=13)": np.array([g["hrest"] >= 13 for g in games]),
    "away off bye (rest>=13)": np.array([g["arest"] >= 13 for g in games]),
    "Thursday game": np.array([g["wday"] == "Thursday" for g in games]),
    "home QB unproven (<150 db)": qdb_h < 150,
    "away QB unproven (<150 db)": qdb_a < 150,
    "big home favorite (p>=.75)": p_cv >= 0.75,
    "big home underdog (p<=.25)": p_cv <= 0.25,
    "toss-up (.45-.55)": (p_cv >= 0.45) & (p_cv <= 0.55),
    "week 1-2 (stale ratings)": np.array([g["week"] <= 2 for g in games]),
}

print(f"DEV out-of-fold calibration slices (n={dev.sum()}, overall LL "
      f"{-(y[dev]*np.log(p_cv[dev])+(1-y[dev])*np.log(1-p_cv[dev])).mean():.5f})")
print(f"{'slice':<30}{'n':>6}{'pred':>8}{'actual':>8}{'bias':>8}{'z':>6}")
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
    print(f"{name:<30}{n:>6}{pm:>8.3f}{am:>8.3f}{am-pm:>+8.3f}{z:>+6.1f}{flag}")
