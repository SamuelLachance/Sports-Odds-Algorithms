"""PFF-WAR T2b: adjusted-wins TRAINING weights in the MAIN 14-feature model.

PFF counts games decided by <=8 points as half a win when fitting team strength
(their year-to-year stability trick). Protocol-safe transplant: evaluation
labels and scored LL untouched; only the logistic FIT gives close games less
sample weight, so coin-flip outcomes pull coefficients less.

DEV 5-fold CV screen (2001-2015), gate -0.0020. TEST look only if gate met.
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

T0 = time.time()
coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102
print(f"[{time.time()-T0:.0f}s] prelude done", flush=True)

# ---- rebuild the shipped 14-col matrix (passrun cols + v7 ratings col) ----
from collections import defaultdict
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
f_pass = np.zeros(len(games)); f_run = np.zeros(len(games)); prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        for st_ in (list(offP.values()) + list(offR.values())
                    + list(dfaP.values()) + list(dfaR.values())):
            st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
    prev = g["season"]
    def crate(st_, lg_): return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)
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
X14[:, 7] = np.load("data/nfl_v7_feature.npy")
cur_cv = cv_ll(X14)
print(f"[{time.time()-T0:.0f}s] current model DEV CV {cur_cv:.5f}", flush=True)

# ---- close-game margins ----
_mg = {}
for _r in csv.DictReader(open("data/nfl_games.csv")):
    if _r["home_score"] != "":
        _mg[_r["game_id"]] = abs(float(_r["home_score"]) - float(_r["away_score"]))
margin = np.array([_mg.get(g["gid"], 99.0) for g in games])

def cv_ll_w(X, w, C=1e6):
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fitm = itr[y[itr] != 0.5]
        clf = LogisticRegression(C=C, max_iter=5000).fit(
            X[fitm], y[fitm], sample_weight=w[fitm])
        sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
    return sc.mean()

print("\nT2b adjusted-wins fit weights, MAIN model (DEV 5-fold, gate -0.0020):", flush=True)
out = {"base": round(cur_cv, 5)}
for T_, cw in ((8, 0.5), (8, 0.7), (3, 0.5), (3, 0.7)):
    w = np.where(margin <= T_, cw, 1.0)
    s = cv_ll_w(X14, w)
    out[f"T{T_}_cw{cw}"] = round(s, 5)
    print(f"  T<={T_} cw={cw}:  CV {s:.5f}  ({cur_cv-s:+.5f})", flush=True)
w = np.minimum(1.0, margin / 8.0)
s = cv_ll_w(X14, w)
out["graded"] = round(s, 5)
print(f"  graded |m|/8: CV {s:.5f}  ({cur_cv-s:+.5f})", flush=True)

json.dump(out, open("data/nfl_adjwin_main.json", "w"), indent=1)
print("wrote data/nfl_adjwin_main.json", flush=True)
