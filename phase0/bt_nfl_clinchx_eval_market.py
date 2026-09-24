"""Breakthrough program (NFL) - nfl_clinchx MARKET READOUT. EVALUATION ONLY, DEV 2006-2015 ONLY.

Where do the exact-lock games sit relative to the de-vigged closing moneyline? Odds are
read here ONLY for DEV rows (2006-2015) and ONLY to benchmark; nothing in this file feeds
a feature, a fit or a threshold, and no builder imports it.
Output: data/bt_nfl_clinchx_eval_market.json
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

TEST_ERA = 2016
D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]; N = len(G)
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
Y = np.array([g["y"] for g in G], float)
FL = np.load("data/bt_nfl_clinchx_flags.npy"); LKX = FL[:, 1] - FL[:, 0]


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def oof(extra):
    P = np.full(N, np.nan)
    for s_ in range(2006, 2016):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        Xs = X14
        if extra is not None:
            E = extra.reshape(-1, 1); mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X14, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / 3.0)
        P[te] = LogisticRegression(C=100.0, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w).predict_proba(Xs[te])[:, 1]
    return P


P0 = oof(None); P1 = oof(LKX)
dev_gids = {g["gid"] for g in G if 2006 <= g["season"] <= 2015}
ML = {}
with open("data/nfl_games.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["game_id"] in dev_gids and r["home_moneyline"] and r["away_moneyline"]:
            ML[r["game_id"]] = (float(r["home_moneyline"]), float(r["away_moneyline"]))
imp = lambda v: (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))  # noqa: E731
PC = np.full(N, np.nan)
for i, g in enumerate(G):
    if g["gid"] in ML:
        h, a = ML[g["gid"]]; qh, qa = imp(h), imp(a); PC[i] = qh / (qh + qa)
dev = (SEAS >= 2006) & (SEAS <= 2015) & ~np.isnan(PC)
out = {"note": "EVAL ONLY; de-vigged closing moneyline; DEV 2006-2015", "n_dev_with_close": int(dev.sum())}
for nm, m_ in (("home_locked", FL[:, 0] == 1), ("away_locked", FL[:, 1] == 1),
               ("any_locked", FL.sum(1) > 0), ("not_locked", FL.sum(1) == 0)):
    m = dev & m_
    out[nm] = {"n": int(m.sum()), "p_base": round(float(P0[m].mean()), 4), "p_A1": round(float(P1[m].mean()), 4),
               "p_close": round(float(PC[m].mean()), 4), "home_win": round(float(Y[m].mean()), 4),
               "ll_base": round(float(llv(Y[m], P0[m]).mean()), 5), "ll_A1": round(float(llv(Y[m], P1[m]).mean()), 5),
               "ll_close": round(float(llv(Y[m], PC[m]).mean()), 5)}
    print(nm, out[nm])
json.dump(out, open("data/bt_nfl_clinchx_eval_market.json", "w"), indent=1)
print("wrote data/bt_nfl_clinchx_eval_market.json")
