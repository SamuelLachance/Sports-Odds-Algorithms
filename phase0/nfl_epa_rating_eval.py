"""Efficiency team rating: off/def EPA-per-play EWMAs as blend features.

Continuous magnitude version of what off/def TrueSkill binarized (success only).
Per team two states: OFF (epa generated / play) and DEFALLOW (epa allowed / play),
per-game decayed sums, EB-shrunk toward the DEV-era league EPA/play, season regress.

Variants (selected on DEV 5-fold CV only):
  net:  ((off_h - def_h) - (off_a - def_a))               1 feature
  two:  [off_h - def_a, off_a - def_h]                    2 features (matchup sides)

PRE-SET GATE (before results): DEV CV must improve >= 0.0020 over the shipped blend
to earn the ONE TEST look. Base = elo_logit + qb_diff(rep prior), CV 0.62865, TEST 0.62806.
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
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, run_elo  # noqa: E402
from nfl_elo import FRANCHISE  # noqa: E402
from nfl_qb_elo import QbElo, load_games_qb, load_qb_weeks  # noqa: E402

PBP_FIX = {"JAC": "JAX", "WSH": "WAS", **FRANCHISE}
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
    g["gid"] = r["game_id"]

# ---- per-(game, team) offensive EPA aggregates ----
agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))    # gid -> team -> [sum, n]
for r in csv.reader(open("data/nfl_plays.csv")):
    if r[0] == "game_id":
        continue
    gid, _, pos, _, epa = r
    t = PBP_FIX.get(pos, pos)
    a = agg[gid][t]
    a[0] += float(epa); a[1] += 1
# DEV-era league EPA/play
num = den = 0.0
for gid, tm in agg.items():
    if int(gid[:4]) in DEV_YEARS:
        for t, (s, n) in tm.items():
            num += s; den += n
LG = num / den
print(f"league EPA/play (DEV era): {LG:+.4f} | games with plays {len(agg)}")

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


def epa_features(decay, prior_n, season_decay):
    off = defaultdict(lambda: [0.0, 0.0])       # team -> [sum, n]
    dfa = defaultdict(lambda: [0.0, 0.0])       # epa allowed
    f_net = np.zeros(len(games))
    f_h = np.zeros(len(games)); f_a = np.zeros(len(games))
    prev = None

    def rate(st):
        s, n = st
        return (s + prior_n * LG) / (n + prior_n) if (n + prior_n) > 0 else LG

    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st in list(off.values()) + list(dfa.values()):
                st[0] *= (1.0 - season_decay); st[1] *= (1.0 - season_decay)
        prev = g["season"]
        h, a = g["home"], g["away"]
        oh, oa = rate(off[h]), rate(off[a])
        dh, da = rate(dfa[h]), rate(dfa[a])
        f_h[i] = oh - da                      # home offense vs away defense
        f_a[i] = oa - dh
        f_net[i] = (oh - dh) - (oa - da)
        tm = agg.get(g["gid"])
        if tm:
            for t_off, opp in ((h, a), (a, h)):
                s, n = tm.get(t_off, (0.0, 0))
                o = off[t_off]; o[0] = decay * o[0] + s; o[1] = decay * o[1] + n
                d = dfa[opp];   d[0] = decay * d[0] + s; d[1] = decay * d[1] + n
    return f_net, f_h, f_a


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
print(f"shipped blend DEV CV {base_cv:.5f}   (TEST-look gate: CV <= {base_cv - GATE:.5f})\n")

rng = np.random.default_rng(20260723)
best, t0 = None, time.time()
for i in range(150):
    cand = dict(
        decay=float(rng.uniform(0.70, 1.0)),
        prior_n=float(np.exp(rng.uniform(np.log(50.0), np.log(5000.0)))),
        season_decay=float(rng.uniform(0.0, 0.8)),
    )
    f_net, f_h, f_a = epa_features(**cand)
    s_net = cv_ll(np.column_stack([BASE, f_net]))
    s_two = cv_ll(np.column_stack([BASE, f_h, f_a]))
    for kind, s in (("net", s_net), ("two", s_two)):
        if best is None or s < best[0]:
            best = (s, kind, cand)
    if (i + 1) % 30 == 0:
        print(f"  ...{i+1}/150  best CV {best[0]:.5f} ({best[0]-base_cv:+.5f}, {best[1]})"
              f"  ({time.time()-t0:.0f}s)", flush=True)

s, kind, cand = best
print(f"\nbest: {kind}  {({k: round(v, 4) for k, v in cand.items()})}")
print(f"DEV CV: base {base_cv:.5f} -> +epa {s:.5f} ({s-base_cv:+.5f})")
if s > base_cv - GATE:
    print(f"GATE NOT MET (needs {-GATE:+.4f}) -> no TEST look. Feature parked.")
    json.dump({"verdict": "gate_not_met", "dev_cv_base": round(base_cv, 5),
               "dev_cv": round(s, 5), "kind": kind, "params": cand},
              open("data/nfl_epa_rating.json", "w"), indent=1)
    sys.exit(0)

# ---- gate met: ONE TEST look ----
f_net, f_h, f_a = epa_features(**cand)
Xw = np.column_stack([BASE, f_net]) if kind == "net" else np.column_stack([BASE, f_h, f_a])
fit = dev & (y != 0.5)
cb = LogisticRegression(C=1e6, max_iter=3000).fit(BASE[fit], y[fit])
cw = LogisticRegression(C=1e6, max_iter=3000).fit(Xw[fit], y[fit])
pb = cb.predict_proba(BASE[test])[:, 1]
pw = cw.predict_proba(Xw[test])[:, 1]
yt = y[test]
d = llv(yt, pb) - llv(yt, pw)
rng2 = np.random.default_rng(7)
n = len(d)
bs = d[rng2.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
sig = "SIG improvement" if lo > 0 else ("SIG worse" if hi < 0 else "n.s.")
print(f"\nTEST 2016-2025 (n={n}, ONE look):")
print(f"  shipped blend      LL {llv(yt, pb).mean():.5f}")
print(f"  + epa rating ({kind}) LL {llv(yt, pw).mean():.5f}")
print(f"  delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
json.dump({"kind": kind, "params": cand, "dev_cv_base": round(base_cv, 5),
           "dev_cv": round(s, 5), "test_base": round(float(llv(yt, pb).mean()), 5),
           "test_new": round(float(llv(yt, pw).mean()), 5),
           "delta_ll": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "verdict": sig},
          open("data/nfl_epa_rating.json", "w"), indent=1)
