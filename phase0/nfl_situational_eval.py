"""Last Tier-2 screens (DEV-ONLY): travel/timezone + per-team HFA.

TRAVEL  tz_shift = tz(away home city, era-aware) - tz(game site); plus the classic
        west-coast-team-at-early-ET-kick flag (circadian disadvantage).
TEAMHFA as-of EWMA per team of home-game residual (y - p_elo), EB-shrunk — Denver
        altitude / Seattle noise vs the league-average HFA already in Elo.
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

TZ = {  # hours behind ET
    "BUF": 0, "MIA": 0, "NE": 0, "NYJ": 0, "BAL": 0, "CIN": 0, "CLE": 0, "PIT": 0,
    "IND": 0, "JAX": 0, "ATL": 0, "CAR": 0, "TB": 0, "WAS": 0, "NYG": 0, "PHI": 0,
    "DET": 0, "CHI": 1, "GB": 1, "MIN": 1, "DAL": 1, "HOU": 1, "TEN": 1, "KC": 1,
    "NO": 1, "DEN": 2, "ARI": 2, "SEA": 3, "SF": 3, "LAC": 3, "LV": 3, "LA": 3,
}
def tz_of(team, season):
    if team == "LA" and season < 2016:
        return 1                        # St. Louis era (CT)
    return TZ[team]

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
    try:
        g["kick"] = int(r["gametime"].split(":")[0]) + int(r["gametime"].split(":")[1]) / 60.0
    except (ValueError, IndexError, AttributeError):
        g["kick"] = 13.0

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
BASE = np.column_stack([lgt, qd])

# ---- travel features (static, no tuning needed) ----
f_tz = np.zeros(len(games)); f_early = np.zeros(len(games))
for i, g in enumerate(games):
    if g["neutral"]:
        continue
    ta, th = tz_of(g["away"], g["season"]), tz_of(g["home"], g["season"])
    f_tz[i] = ta - th                       # +3 = west-coast team traveling to ET
    f_early[i] = 1.0 if (ta - th >= 2 and g["kick"] <= 14.0) else 0.0

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

def cv_ll(X):
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fit = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=1e6, max_iter=3000).fit(X[fit], y[fit])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()

base_cv = cv_ll(BASE)
print(f"frozen blend DEV CV {base_cv:.5f}\n")
print("TRAVEL:")
for name, cols in (("tz shift", [f_tz]), ("early-kick flag", [f_early]),
                   ("both", [f_tz, f_early])):
    s = cv_ll(np.column_stack([BASE] + cols))
    print(f"  + {name:<16} CV {s:.5f} ({s-base_cv:+.5f})")

# ---- per-team HFA (small tune) ----
print("\nPER-TEAM HFA:")
def team_hfa(decay, prior_n):
    st = {}
    f = np.zeros(len(games))
    for i, g in enumerate(games):
        if not g["neutral"]:
            s = st.get(g["home"], (0.0, 0.0))
            f[i] = s[0] / (s[1] + prior_n)
            resid = g["y"] - pe[i]
            st[g["home"]] = (decay * s[0] + resid, decay * s[1] + 1.0)
    return f

best = None
rng = np.random.default_rng(20260723)
for _ in range(60):
    decay = float(rng.uniform(0.90, 1.0))
    prior_n = float(np.exp(rng.uniform(np.log(5.0), np.log(400.0))))
    s = cv_ll(np.column_stack([BASE, team_hfa(decay, prior_n)]))
    if best is None or s < best[0]:
        best = (s, decay, prior_n)
print(f"  best CV {best[0]:.5f} ({best[0]-base_cv:+.5f})  "
      f"decay={best[1]:.3f} prior_n={best[2]:.0f}")
json.dump({"base_cv": round(base_cv, 5),
           "team_hfa_cv": round(best[0], 5)},
          open("data/nfl_situational.json", "w"), indent=1)
