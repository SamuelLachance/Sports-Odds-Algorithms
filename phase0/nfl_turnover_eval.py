"""Turnover features (DEV-ONLY, bundle candidate).

From pbp aggregates (data/nfl_turnovers.csv): per team as-of EWMAs of
  MARGIN  per-game turnover margin (takeaways - giveaways), EB-shrunk to 0
  LUCK    fumble-recovery rate above 50% (recoveries of ALL fumbles in team's
          games vs the coin-flip expectation) — pure luck, should REGRESS
Blend decides signs vs elo_logit (Elo absorbs TO-fueled wins at face value).
Variants: margin / luck / both, tuned on DEV 5-fold CV. NO TEST look here.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, "phase0")
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, run_elo  # noqa: E402
from nfl_elo import FRANCHISE  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

PBP_FIX = {"JAC": "JAX", "WSH": "WAS", **FRANCHISE}

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
    g["gid"] = r["game_id"]

# per-(game, team): int_thrown, fumbles, fumbles_lost
tv = defaultdict(dict)
for r in csv.DictReader(open("data/nfl_turnovers.csv")):
    t = PBP_FIX.get(r["posteam"], r["posteam"])
    tv[r["game_id"]][t] = (float(r["int_thrown"]), float(r["fumbles"]),
                           float(r["fumbles_lost"]))
print(f"turnover aggregates for {len(tv)} games")

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


def to_features(decay, prior_n, season_decay):
    mar = defaultdict(lambda: [0.0, 0.0])     # team -> [margin_sum, games]
    luck = defaultdict(lambda: [0.0, 0.0])    # team -> [recoveries - expected, fumbles]
    f_mar = np.zeros(len(games)); f_luck = np.zeros(len(games))
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st in list(mar.values()) + list(luck.values()):
                st[0] *= (1.0 - season_decay); st[1] *= (1.0 - season_decay)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f_mar[i] = (mar[h][0] / (mar[h][1] + prior_n)
                    - mar[a][0] / (mar[a][1] + prior_n))
        f_luck[i] = (luck[h][0] / (luck[h][1] + prior_n)
                     - luck[a][0] / (luck[a][1] + prior_n))
        gt = tv.get(g["gid"])
        if gt and h in gt and a in gt:
            ih, fh, flh = gt[h]
            ia, fa, fla = gt[a]
            give_h = ih + flh; give_a = ia + fla
            m_h = give_a - give_h                        # home TO margin
            for team, mg in ((h, m_h), (a, -m_h)):
                s = mar[team]; s[0] = decay * s[0] + mg; s[1] = decay * s[1] + 1.0
            # fumble luck: team's recoveries vs 0.5 * all fumbles on the field
            tot_f = fh + fa
            rec_h = (fh - flh) + fla                     # own kept + opp taken
            for team, rec in ((h, rec_h), (a, tot_f - rec_h)):
                s = luck[team]
                s[0] = decay * s[0] + (rec - 0.5 * tot_f)
                s[1] = decay * s[1] + tot_f
    return f_mar, f_luck


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

rng = np.random.default_rng(20260723)
best = {"margin": None, "luck": None, "both": None}
t0 = time.time()
for i in range(120):
    cand = dict(
        decay=float(rng.uniform(0.80, 1.0)),
        prior_n=float(np.exp(rng.uniform(np.log(2.0), np.log(120.0)))),
        season_decay=float(rng.uniform(0.0, 0.8)),
    )
    f_mar, f_luck = to_features(**cand)
    for kind, cols in (("margin", [f_mar]), ("luck", [f_luck]), ("both", [f_mar, f_luck])):
        s = cv_ll(np.column_stack([BASE] + cols))
        if best[kind] is None or s < best[kind][0]:
            best[kind] = (s, cand)
    if (i + 1) % 30 == 0:
        b = min(v[0] for v in best.values())
        print(f"  ...{i+1}/120  best CV {b:.5f} ({b-base_cv:+.5f})  ({time.time()-t0:.0f}s)",
              flush=True)

print()
for kind, (s, cand) in best.items():
    print(f"  {kind:<7} CV {s:.5f} ({s-base_cv:+.5f})  "
          f"{({k: round(v, 3) for k, v in cand.items()})}")
winner = min(best, key=lambda k: best[k][0])
s, cand = best[winner]
print(f"\nDEV verdict: {winner} ({s-base_cv:+.5f}) — "
      f"{'bundle candidate' if s < base_cv - 0.0005 else 'too weak, drop'}. NO TEST look.")
json.dump({"winner": winner, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "delta_cv": round(s - base_cv, 5), "params": cand,
           "all": {k: {"cv": round(v[0], 5)} for k, v in best.items()}},
          open("data/nfl_turnover.json", "w"), indent=1)
