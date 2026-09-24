"""Breakthrough program 2026-09-24 -- NFL gap anatomy, step 1 (build).

Reproduces the SHIPPED NFL main model on DEV seasons only:
  * the exact 14-feature matrix of phase0/nfl_season_serve.py (coord_tune prelude
    -> 12 cols, + pass/run channel block copied verbatim, col 7 := v7 player
    ratings from data/nfl_v7_feature.npy)
  * the exact shipped training protocol (walk-forward per season, train on all
    seasons < s with ties dropped, recency half-life 3 seasons, C=100)
applied to DEV seasons 2006..2015 (the DEV years with closing moneylines).

NOTHING about TEST seasons is computed or written: the feature walks run over the
full file (a walk-forward feature for a DEV game only depends on earlier games),
but every row with season >= 2016 is dropped before anything is fit, scored, or
saved. Output: data/bt_nfl_anatomy_dev.csv (one row per 1999-2015 game with the
14 features; p_model filled for 2006-2015).
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # full feature prelude  # noqa: S102
print(f"[{time.time()-T0:.0f}s] prelude done ({len(games)} games)", flush=True)

# ---------------- pass/run channels (verbatim from nfl_season_serve.py) ----------------
aggc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); hdr = next(rd); ix = {c: i for i, c in enumerate(hdr)}
    for r in rd:
        t_ = PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]])
        a = aggc[r[ix["game_id"]]][t_]
        if r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]:
            a[0] += float(r[ix["epa"]]); a[1] += 1
        else:
            a[2] += float(r[ix["epa"]]); a[3] += 1
ps = pn = rs = rn = 0.0
for gid, tm in aggc.items():
    if int(gid[:4]) in DEV_YEARS:
        for t_, (a_, b_, c_, d_) in tm.items():
            ps += a_; pn += b_; rs += c_; rn += d_
LGP, LGR = ps / pn, rs / rn
dec_, pn_, sd_ = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
offP = defaultdict(lambda: [0.0, 0.0]); offR = defaultdict(lambda: [0.0, 0.0])
dfaP = defaultdict(lambda: [0.0, 0.0]); dfaR = defaultdict(lambda: [0.0, 0.0])
def crate(st_, lg_): return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)
f_pass = np.zeros(len(games)); f_run = np.zeros(len(games)); prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        for st_ in (list(offP.values()) + list(offR.values())
                    + list(dfaP.values()) + list(dfaR.values())):
            st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
    prev = g["season"]
    h, a = g["home"], g["away"]
    f_pass[i] = (crate(offP[h], LGP) - crate(dfaP[h], LGP)) - (crate(offP[a], LGP) - crate(dfaP[a], LGP))
    f_run[i] = (crate(offR[h], LGR) - crate(dfaR[h], LGR)) - (crate(offR[a], LGR) - crate(dfaR[a], LGR))
    tm = aggc.get(g["gid"])
    if tm:
        for t_off, opp in ((h, a), (a, h)):
            pS, pN, rS, rN = tm.get(t_off, (0.0, 0, 0.0, 0))
            o = offP[t_off]; o[0] = dec_ * o[0] + pS; o[1] = dec_ * o[1] + pN
            o = offR[t_off]; o[0] = dec_ * o[0] + rS; o[1] = dec_ * o[1] + rN
            d2 = dfaP[opp]; d2[0] = dec_ * d2[0] + pS; d2[1] = dec_ * d2[1] + pN
            d2 = dfaR[opp]; d2[0] = dec_ * d2[0] + rS; d2[1] = dec_ * d2[1] + rN
X14 = np.column_stack([X_of(F), f_pass, f_run])
v7 = np.load("data/nfl_v7_feature.npy")
assert len(v7) == len(games)
X14[:, 7] = v7
COLS = ["lgt", "qd", "epa", "early", "thfa", "rest", "luck", "v7", "ol", "de", "sk", "rq",
        "pass", "run"]

# ---------------- DEV-ONLY from here: drop every TEST row ----------------
keep = seasons <= 2015
assert keep.sum() < len(games)
Xd, yd, sd = X14[keep], y[keep], seasons[keep]
gd = [g for g, k in zip(games, keep) if k]
print(f"[{time.time()-T0:.0f}s] kept {int(keep.sum())} games (seasons <= 2015)", flush=True)
print("v7 nonzero on DEV rows:", int((Xd[:, 7] != 0).sum()),
      "| rq nonzero by first season:",
      int(min(sd[Xd[:, 11] != 0])) if (Xd[:, 11] != 0).any() else None)

# ---------------- shipped protocol, walk-forward DEV 2006..2015 ----------------
HL, BC = 3.0, 100.0
p_model = np.full(len(yd), np.nan)
p_elo = np.full(len(yd), np.nan)        # reference: Elo-logit-only, same protocol
coefs = {}
for s_ in range(2006, 2016):
    tr = (sd < s_) & (yd != 0.5)
    te = sd == s_
    w_ = 0.5 ** ((s_ - 1 - sd[tr]) / HL)
    m_ = LogisticRegression(C=BC, max_iter=5000).fit(Xd[tr], yd[tr], sample_weight=w_)
    p_model[te] = m_.predict_proba(Xd[te])[:, 1]
    coefs[s_] = [round(float(c), 5) for c in m_.coef_[0]] + [round(float(m_.intercept_[0]), 5)]
    m2 = LogisticRegression(C=BC, max_iter=5000).fit(Xd[tr][:, [0]], yd[tr], sample_weight=w_)
    p_elo[te] = m2.predict_proba(Xd[te][:, [0]])[:, 1]

with open("data/bt_nfl_anatomy_dev.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.writer(fh)
    wr.writerow(["gid", "season", "week", "home", "away", "neutral", "home_qb", "away_qb",
                 "hrest", "arest", "kick", "y", "p_model", "p_elo"] + COLS)
    for i, g in enumerate(gd):
        wr.writerow([g["gid"], g["season"], g["week"], g["home"], g["away"], int(bool(g["neutral"])),
                     g["home_qb"], g["away_qb"], g["hrest"], g["arest"], round(g["kick"], 3),
                     yd[i],
                     "" if np.isnan(p_model[i]) else f"{p_model[i]:.6f}",
                     "" if np.isnan(p_elo[i]) else f"{p_elo[i]:.6f}"]
                    + [f"{v:.6g}" for v in Xd[i]])
json.dump({"cols": COLS + ["intercept"], "coefs_by_season": coefs},
          open("data/bt_nfl_anatomy_coefs.json", "w"), indent=1)
m = ~np.isnan(p_model)
print(f"[{time.time()-T0:.0f}s] DEV 2006-2015 walk-forward: n={int(m.sum())} "
      f"LL model {llv(yd[m], p_model[m]).mean():.5f} | elo-only {llv(yd[m], p_elo[m]).mean():.5f}",
      flush=True)
print("wrote data/bt_nfl_anatomy_dev.csv, data/bt_nfl_anatomy_coefs.json")
