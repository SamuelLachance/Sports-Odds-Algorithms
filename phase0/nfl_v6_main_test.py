"""MAIN-PROTOCOL TEST: the v6 participation-TrueSkill feature in the GLASSBOX model.

The prior TS-in-model rejections (ledger 48-49) used the untuned engine; v6 is
Elo-strength on its own (0.638/65% ratings-only; blend coefs prove ~co-equal
independent info vs Elo). Variants, retrained with the adopted method
(walk-forward + recency hl=3, C=100), scored ONCE on the locked TEST 2016-2025:
  A ADD      X15 = X14 + v6
  B REPLACE ts_edge (near-dead coef .013) with v6
  C REPLACE rq (the weekly composite) with v6
Feature is zero pre-2016 (like snap absence pre-2013). Paired bootstrap vs the
current model; adopt-if-better per the point-estimate rule.
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
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102
print(f"[{time.time()-T0:.0f}s] prelude done", flush=True)

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
v6 = np.load("data/nfl_v6_feature.npy")
assert len(v6) == len(games)
print(f"[{time.time()-T0:.0f}s] X14 + v6 feature ready "
      f"(corr v6 vs rq {np.corrcoef(v6[test], X14[test,11])[0,1]:.3f}, "
      f"vs elo_lgt {np.corrcoef(v6[test], X14[test,0])[0,1]:.3f})", flush=True)

HL, BC = 3.0, 100.0
def wf_probs(X):
    p_wf = np.zeros(int(test.sum()))
    tidx = np.where(test)[0]
    pos_ = {gi: j for j, gi in enumerate(tidx)}
    for s_ in sorted(np.unique(seasons[test])):
        tr = (seasons < s_) & (y != 0.5)
        te = test & (seasons == s_)
        w_ = 0.5 ** ((s_ - 1 - seasons[tr]) / HL)
        m_ = LogisticRegression(C=BC, max_iter=5000).fit(X[tr], y[tr], sample_weight=w_)
        pp = m_.predict_proba(X[te])[:, 1]
        for k2, gi in enumerate(np.where(te)[0]):
            p_wf[pos_[gi]] = pp[k2]
    return p_wf

X_add = np.column_stack([X14, v6])
X_rts = X14.copy(); X_rts[:, 7] = v6         # replace ts_edge
X_rrq = X14.copy(); X_rrq[:, 11] = v6        # replace rq
yt = y[test]
p_base = wf_probs(X14)
ll_base = float(llv(yt, p_base).mean())
rng = np.random.default_rng(7)
def paired(p_new):
    d = llv(yt, p_base) - llv(yt, p_new)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
print(f"\nmain TEST (n={len(yt)}), walk-forward + recency:")
print(f"  current model (14 features)     LL {ll_base:.5f}")
rows = {}
for tag, X in (("A ADD v6 as 15th", X_add),
               ("B REPLACE ts_edge -> v6", X_rts),
               ("C REPLACE rq -> v6", X_rrq)):
    pp = wf_probs(X)
    ll_ = float(llv(yt, pp).mean())
    d, lo, hi = paired(pp)
    msk = yt != 0.5
    acc = float(((pp > 0.5) == (yt > 0.5))[msk].mean())
    rows[tag] = {"ll": round(ll_, 5), "delta": round(d, 5),
                 "ci": [round(lo, 5), round(hi, 5)], "acc": round(acc, 4)}
    print(f"  {tag:<28} LL {ll_:.5f}  (delta {d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  acc {acc*100:.1f}%")
best = min(rows.items(), key=lambda kv: kv[1]["ll"])
verdict = f"ADOPT {best[0]}" if best[1]["ll"] < ll_base else "keep current"
print(f"\n-> {verdict}")
json.dump({"base_ll": round(ll_base, 5), "variants": rows, "verdict": verdict},
          open("data/nfl_v6_main_test.json", "w"), indent=1)
