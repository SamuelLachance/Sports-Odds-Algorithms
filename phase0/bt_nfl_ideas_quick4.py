"""Breakthrough program (NFL, ideas leg) - round 4: starting-QB health + interim coach. DEV ONLY.

Two untested day-of channels:
  H  starting-QB health. The actual starter (games.csv, known pre-kickoff) is
     looked up on THAT week's official injury report (Friday report, 2009+):
       H1 practice status DNP or Limited (era-stable: practice reports were not
          touched by the 2016 Probable abolition that broke ledger row 37)
       H2 game status Questionable or Doubtful (NOT era-stable - reported for
          anatomy only, never to be served)
     Rows 37/38/60 tested ALL-position tag shares weighted by snap share (live
     only 2013+ on DEV, a QB counted like any WR). The starting QB alone, weighted
     by his own QB rating magnitude, was never screened.
       H3 = H1 x (starter's as-of QB rating - replacement)  (a hurt good QB loses more)
  I  interim coach: head coach differs from the team's previous game's coach
     within the same season (games.csv coach columns, known pre-game).
Market odds: EVALUATION ONLY (residual anatomy, DEV 2006-2015). Output:
data/bt_nfl_ideas_quick4.json
"""
from __future__ import annotations

import csv
import glob
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

TEST_ERA = 2016
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
RNG = np.random.default_rng(20260927)
D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]; X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
QH, QA = np.array(D["qh"]), np.array(D["qa"])
N = len(G)
SEAS = np.array([g["season"] for g in G]); assert SEAS.max() < TEST_ERA
WK = np.array([g["week"] for g in G]); TYP = np.array([g["type"] for g in G])
Y = np.array([g["y"] for g in G], float)
PRED = np.array([np.nan if g["pred"] is None else g["pred"] for g in G])
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
RAW = {r["game_id"]: r for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8"))
       if r["season"] and int(r["season"]) < TEST_ERA}


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ------------------------------------------------ injury report lookup -----
INJ = {}
for f in sorted(glob.glob("data/inj_20*.csv")):
    s = int(f[-8:-4])
    if s >= TEST_ERA:
        continue
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["game_type"] not in ("REG", "WC", "DIV", "CON", "SB"):
            continue
        try:
            key = (r["gsis_id"], int(r["season"]), int(r["week"]))
        except ValueError:
            continue
        INJ[key] = (r["report_status"] or "", r["practice_status"] or "")
BADPRAC = ("Did Not Participate", "Limited Participation")
H1 = np.zeros(N); H2 = np.zeros(N); H3 = np.zeros(N)
cnt = {"h1": 0, "h2": 0}
REP = -0.28533125161330286   # shipped replacement delta (rating scale of qh/qa)
for i, g in enumerate(G):
    if g["season"] < 2009:
        continue
    for side, qid, q in ((1, g["home_qb"], QH[i]), (-1, g["away_qb"], QA[i])):
        rs, ps = INJ.get((qid, g["season"], g["week"]), ("", ""))
        bad = 1.0 if ps.startswith(BADPRAC) else 0.0
        tag = 1.0 if rs in ("Questionable", "Doubtful") else 0.0
        H1[i] -= side * bad; H2[i] -= side * tag           # positive = AWAY starter hurt
        H3[i] -= side * bad * max(q - REP, 0.0)
        cnt["h1"] += bad; cnt["h2"] += tag
dv = SEAS >= 2009
print(f"starter-QB tags 2009-15: practice DNP/Ltd {int(sum(abs(H1[dv]) > 0))} games, Q/D {int(sum(abs(H2[dv]) > 0))} games")

# ------------------------------------------------ interim coach ------------
last_coach = {}
IC = np.zeros(N)
for i, g in enumerate(G):
    r = RAW[g["gid"]]
    for side, team, co in ((1, g["home"], r["home_coach"]), (-1, g["away"], r["away_coach"])):
        lc = last_coach.get(team)
        if lc and lc[0] == g["season"] and lc[1] != co and g["type"] == "REG":
            IC[i] += side
        last_coach[team] = (g["season"], co)
print(f"interim-coach flags DEV-era games: {int((IC != 0).sum())}")

# ------------------------------------------------ anatomy (EVAL ONLY) ------
PC = np.full(N, np.nan)
imp = lambda v: (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))  # noqa: E731
for i, g in enumerate(G):
    if DEV_LO <= g["season"] <= DEV_HI:
        r = RAW[g["gid"]]
        if r["home_moneyline"] and r["away_moneyline"]:
            a_, b_ = imp(float(r["home_moneyline"])), imp(float(r["away_moneyline"]))
            PC[i] = a_ / (a_ + b_)
DEVM = (SEAS >= DEV_LO) & ~np.isnan(PC) & ~np.isnan(PRED) & (Y != 0.5)
lg = lambda p: np.log(p / (1 - p))  # noqa: E731
RES = np.where(DEVM, lg(np.clip(np.nan_to_num(PC, nan=.5), 1e-6, 1 - 1e-6))
               - lg(np.clip(np.nan_to_num(PRED, nan=.5), 1e-6, 1 - 1e-6)), np.nan)
GAPV = llv(Y, np.nan_to_num(PRED, nan=.5)) - llv(Y, np.nan_to_num(PC, nan=.5))
OUT = {"anatomy_eval_only": {}, "results": {}}
for nm, x in (("H1 starter practice DNP/Ltd (away-home)", H1), ("H2 starter Q/D (away-home)", H2),
              ("H3 H1 x rating", H3), ("I interim coach (home-away)", IC)):
    m = DEVM & (x != 0)
    sgn = np.sign(x[m])
    # residual oriented toward the flagged direction: + = close moves the same way as the flag
    r_or = RES[m] * sgn
    OUT["anatomy_eval_only"][nm] = {"n": int(m.sum()), "oriented_resid": round(float(r_or.mean()), 4),
                                   "se": round(float(r_or.std(ddof=1) / np.sqrt(max(m.sum(), 2))), 4),
                                   "gap_in_slice": round(float(GAPV[m].mean()), 4)}
    print(f"  anatomy {nm:<42}", OUT["anatomy_eval_only"][nm])


# ------------------------------------------------ outcome indication -------
def wf(extra=None):
    per, vec, co = {}, [], {}
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        assert SEAS[tr].max() < s_ and SEAS[te].max() < TEST_ERA
        Xs = X14
        if extra is not None:
            E = extra.reshape(-1, 1); mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X14, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1]); per[s_] = float(v.mean()); vec.append(v)
        co[s_] = round(float(m.coef_[0][-1]), 4) if extra is not None else None
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec), co


b_ll, b_per, b_vec, _ = wf()
assert abs(b_ll - 0.62292) < 5e-4
for nm, x in (("H1 starter practice DNP/Ltd", H1), ("H3 H1 x rating", H3), ("I interim coach", IC)):
    ll, per, vec, co = wf(x)
    d = b_vec - vec
    bs = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if b_per[s] - per[s] > 0)
    OUT["results"][nm] = {"gain": round(b_ll - ll, 6), "ci": [round(lo, 6), round(hi, 6)],
                          "seasons_pos": pos, "coef_by_fold": co}
    print(f"  {nm:<32} gain {b_ll-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10 coefs {list(co.values())}")
json.dump(OUT, open("data/bt_nfl_ideas_quick4.json", "w"), indent=1)
