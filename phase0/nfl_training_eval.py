"""NFL training-method suite on the final 14 features. Era drift is PROVEN in this
sport (SRS/injury/reunion), so the drift-aware methods (recency weights, walk-forward)
are the genuine candidates; the learner shootout confirms the MLB linearity verdict.

  A  logistic vs XGBoost vs RF vs MLP           (DEV-CV compared)
  B  ridge C sweep on the blend                 (DEV-selected)
  C  recency-weighted fit (season half-life)    (selected on late-DEV 2013-15 holdout)
  D  walk-forward expanding refit per season    (procedure; plain + recency variant)
One reporting pass on TEST with paired CIs vs current; adopt best if better.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102

# pass/run channels (current model)
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
seas = seasons

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

def cv14(model_fn):
    from sklearn.model_selection import KFold
    idx = np.where(dev)[0]
    sc = np.zeros(len(idx))
    for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
        itr, iva = idx[tr], idx[va]
        fitm = itr[y[itr] != 0.5]
        m = model_fn().fit(X14[fitm], y[fitm])
        sc[va] = llv(y[iva], m.predict_proba(X14[iva])[:, 1])
    return sc.mean()

print("A. learner shootout (DEV CV):")
LEARNERS = {
    "logistic C=1e6": lambda: LogisticRegression(C=1e6, max_iter=5000),
    "xgboost": lambda: XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05,
                                     subsample=0.8, eval_metric="logloss", verbosity=0),
    "random forest": lambda: RandomForestClassifier(n_estimators=400, max_depth=8,
                                                    min_samples_leaf=200, n_jobs=-1),
    "mlp (16)": lambda: MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2,
                                      max_iter=500, early_stopping=True),
}
for nm, fn in LEARNERS.items():
    print(f"  {nm:<16} {cv14(fn):.5f}", flush=True)

print("\nB. ridge C sweep (DEV CV):")
bestC, bestS = 1e6, cv14(lambda: LogisticRegression(C=1e6, max_iter=5000))
for C in (300.0, 100.0, 30.0, 10.0, 3.0):
    s = cv14(lambda C=C: LogisticRegression(C=C, max_iter=5000))
    print(f"  C={C:<8} {s:.5f}", flush=True)
    if s < bestS:
        bestS, bestC = s, C
print(f"  -> best C {bestC}")

print("\nC. recency half-life (select on late-DEV 2013-15):")
tr_in = dev & (seas <= 2012)
tr_val = dev & (seas >= 2013)
bestHL, bestV = None, 9e9
for hl in (None, 12.0, 8.0, 5.0, 3.0):
    w = np.ones(int(tr_in.sum())) if hl is None else \
        0.5 ** ((2012 - seas[tr_in]) / hl)
    fitm = y[tr_in] != 0.5
    m = LogisticRegression(C=bestC, max_iter=5000).fit(
        X14[tr_in][fitm], y[tr_in][fitm], sample_weight=w[fitm])
    v = llv(y[tr_val], m.predict_proba(X14[tr_val])[:, 1]).mean()
    print(f"  half-life {str(hl):<6} val {v:.5f}", flush=True)
    if v < bestV:
        bestV, bestHL = v, hl
print(f"  -> best half-life {bestHL}")

# ---------- TEST evaluations (one reporting pass) ----------
fit_dev = dev & (y != 0.5)
yt = y[test]
def paired(p_new, p_ref):
    d = llv(yt, p_ref) - llv(yt, p_new)
    rng = np.random.default_rng(7)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

p_cur = LogisticRegression(C=1e6, max_iter=5000).fit(
    X14[fit_dev], y[fit_dev]).predict_proba(X14[test])[:, 1]

results = {}
# B: ridge
p_ridge = LogisticRegression(C=bestC, max_iter=5000).fit(
    X14[fit_dev], y[fit_dev]).predict_proba(X14[test])[:, 1]
results[f"ridge C={bestC}"] = p_ridge
# C: recency (fit on full DEV with weights anchored at 2015)
if bestHL is not None:
    w = 0.5 ** ((2015 - seas[dev]) / bestHL)
    fm = y[dev] != 0.5
    p_rec = LogisticRegression(C=bestC, max_iter=5000).fit(
        X14[dev][fm], y[dev][fm], sample_weight=w[fm]).predict_proba(X14[test])[:, 1]
    results[f"recency hl={bestHL}"] = p_rec
# D: walk-forward expanding (plain and recency)
for tag, use_rec in (("walk-forward", False), ("walk-forward+recency", True)):
    p_wf = np.zeros(int(test.sum()))
    tidx = np.where(test)[0]
    pos = {gi: j for j, gi in enumerate(tidx)}
    for s_ in sorted(np.unique(seas[test])):
        tr = (seas < s_) & (y != 0.5)
        te = test & (seas == s_)
        if use_rec and bestHL is not None:
            w_ = 0.5 ** ((s_ - 1 - seas[tr]) / bestHL)
        else:
            w_ = None
        m = LogisticRegression(C=bestC, max_iter=5000).fit(X14[tr], y[tr], sample_weight=w_)
        pp = m.predict_proba(X14[te])[:, 1]
        for k, gi in enumerate(np.where(te)[0]):
            p_wf[pos[gi]] = pp[k]
    results[tag] = p_wf

print(f"\nmain TEST (n={len(yt)}):")
print(f"  current (logistic C=1e6, DEV-fit)   LL {llv(yt, p_cur).mean():.5f}")
best_name, best_ll = None, 9e9
for nm, pp in results.items():
    ll_ = llv(yt, pp).mean()
    d, lo, hi = paired(pp, p_cur)
    tag = "  <-- best" if ll_ < best_ll else ""
    if ll_ < best_ll:
        best_ll, best_name = ll_, nm
    print(f"  {nm:<28} LL {ll_:.5f}  (delta {d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]){tag}")
print(f"\n-> {'ADOPT ' + best_name if best_ll < llv(yt, p_cur).mean() else 'keep current'}")
json.dump({"best": best_name, "best_ll": round(float(best_ll), 5),
           "current_ll": round(float(llv(yt, p_cur).mean()), 5),
           "bestC": bestC, "bestHL": bestHL},
          open("data/nfl_training.json", "w"), indent=1)
