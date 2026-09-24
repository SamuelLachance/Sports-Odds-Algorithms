"""Breakthrough program (NFL, ideas leg) - round 3: decompose the two live leads. DEV ONLY.

Re-uses the feature builders of bt_nfl_ideas_quick.py / _quick2.py (exec of
their builder sections; no harness code duplicated) and asks:
  1. do opponent-adjusted TEAM EPA (replace cols 2/12/13) and opponent-adjusted
     QB (add) stack as one 'schedule-adjusted process block'?
  2. which half of the late-season stakes signal carries it (locked vs
     eliminated), is it week-17 specific, and what sign does each coefficient
     take in every fold?
  3. market-residual anatomy (EVAL ONLY) inside the locked / eliminated games.
Market odds are read ONLY in section 3, for DEV 2006-2015 rows, as an evaluation
benchmark. No odds reach any feature or fit. No season >= 2016 row exists.
Output: data/bt_nfl_ideas_quick3.json
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

_src2 = open("phase0/bt_nfl_ideas_quick2.py", encoding="utf-8").read()
exec(compile(_src2.split("# ------------------------------------------------------------ harness")[0],
             "quick2<builders>", "exec"))  # noqa: S102
_src1 = open("phase0/bt_nfl_ideas_quick.py", encoding="utf-8").read()
_b1 = _src1.split("# ============================================= C2 playoff leverage")[0]
_ns1 = {}
exec(compile(_b1, "quick1<C1 builders>", "exec"), _ns1)  # noqa: S102
ADJ_ALL, ADJ_PASS, ADJ_RUN = _ns1["adj_all"], _ns1["adj_pass"], _ns1["adj_run"]
assert np.corrcoef(_ns1["raw_all"], n_all)[0, 1] > 0.99999 and np.corrcoef(_ns1["raw_pass"], n_pass)[0, 1] > 0.99999

RNG3 = np.random.default_rng(20260926)


def wf(extra=None, replace=None, return_coefs=False):
    X = X14.copy()
    if replace:
        for c, v in replace.items():
            X[:, c] = v
    per, vec, coefs = {}, [], {}
    for s_ in range(DEV_LO, DEV_HI + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        assert SEAS[tr].max() < s_ and SEAS[te].max() < TEST_ERA
        Xs = X
        if extra is not None:
            E = np.column_stack(extra) if isinstance(extra, (list, tuple)) else extra.reshape(-1, 1)
            mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v)
        coefs[s_] = m.coef_[0][14:].round(4).tolist()
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec), coefs


b_ll, b_per, b_vec, _ = wf()
assert abs(b_ll - 0.62292) < 5e-4
OUT = {"baseline_dev_ll": b_ll, "results": {}}


def rep(nm, res):
    ll, per, vec, coefs = res
    d = b_vec - vec
    bs = d[RNG3.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pos = sum(1 for s in per if b_per[s] - per[s] > 0)
    print(f"  {nm:<48} {ll:.5f} gain {b_ll-ll:+.5f} CI[{lo:+.5f},{hi:+.5f}] seasons+ {pos}/10", flush=True)
    OUT["results"][nm] = {"ll": round(ll, 6), "gain": round(b_ll - ll, 6), "ci": [round(lo, 6), round(hi, 6)],
                          "seasons_pos": pos,
                          "per_season_gain": {str(s): round(b_per[s] - per[s], 5) for s in per},
                          "added_coefs_by_fold": {str(k): v for k, v in coefs.items()}}


print("1. schedule-adjusted process block")
rep("C1c team opp-adj replace 2/12/13", wf(replace={2: ADJ_ALL, 12: ADJ_PASS, 13: ADJ_RUN}))
rep("Q  opp-adj QB increment add", wf(FE["Q"]))
rep("C1c + Q (schedule-adjusted block)", wf(FE["Q"], replace={2: ADJ_ALL, 12: ADJ_PASS, 13: ADJ_RUN}))

print("2. stakes decomposition")
LK = LOCK[:, 1] - LOCK[:, 0]; EL = ELIM[:, 1] - ELIM[:, 0]
wk17 = (WK == 17) & (TYP == "REG")
rep("S2 locked + eliminated", wf([LK, EL]))
rep("S3 locked only", wf(LK))
rep("S4 eliminated only", wf(EL))
rep("S5 locked, week 17 only", wf(LK * wk17))
s = OUT["results"]["S3 locked only"]
print("   locked coef by fold:", s["added_coefs_by_fold"])
print("   S2 coefs by fold:", OUT["results"]["S2 locked + eliminated"]["added_coefs_by_fold"])

# 3. market-residual anatomy inside stakes games (EVAL ONLY)
import csv  # noqa: E402
PC = np.full(N, np.nan)
for i, g in enumerate(G):
    if DEV_LO <= g["season"] <= DEV_HI:
        r = RAW[g["gid"]]
        if r["home_moneyline"] and r["away_moneyline"]:
            f = lambda v: (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))  # noqa: E731
            qh_, qa_ = f(float(r["home_moneyline"])), f(float(r["away_moneyline"]))
            PC[i] = qh_ / (qh_ + qa_)
_, _, bvec_full, _ = b_ll, b_per, b_vec, None
# rebuild OOF predictions of the baseline for DEV rows
X = X14.copy(); PRED = np.full(N, np.nan)
for s_ in range(DEV_LO, DEV_HI + 1):
    tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
    w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
    PRED[te] = LogisticRegression(C=C, max_iter=5000).fit(X[tr], Y[tr], sample_weight=w).predict_proba(X[te])[:, 1]
DEVM = (SEAS >= DEV_LO) & ~np.isnan(PC) & (Y != 0.5)
lgt = lambda p: np.log(p / (1 - p))  # noqa: E731
RES = lgt(np.clip(PC, 1e-6, 1 - 1e-6)) - lgt(np.clip(PRED, 1e-6, 1 - 1e-6))
GAPV = llv(Y, np.nan_to_num(PRED, nan=.5)) - llv(Y, np.nan_to_num(PC, nan=.5))
ana = {}
for nm, m_ in (("home locked", LOCK[:, 0] == 1), ("away locked", LOCK[:, 1] == 1),
               ("home eliminated only", (ELIM[:, 0] == 1) & (ELIM[:, 1] == 0)),
               ("away eliminated only", (ELIM[:, 1] == 1) & (ELIM[:, 0] == 0))):
    m = DEVM & m_
    ana[nm] = {"n": int(m.sum()), "mean_resid_logit_home": round(float(RES[m].mean()), 4),
               "se": round(float(RES[m].std(ddof=1) / np.sqrt(m.sum())), 4),
               "gap": round(float(GAPV[m].mean()), 4),
               "home_win_rate": round(float(Y[m].mean()), 3),
               "ours_mean_p": round(float(PRED[m].mean()), 3), "close_mean_p": round(float(PC[m].mean()), 3)}
    print(f"  {nm:<22}", ana[nm])
OUT["stakes_market_anatomy_eval_only"] = ana
json.dump(OUT, open("data/bt_nfl_ideas_quick3.json", "w"), indent=1)
print("wrote data/bt_nfl_ideas_quick3.json")
