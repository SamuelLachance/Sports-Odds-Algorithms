"""Margin-regression-then-threshold head vs direct logistic (MAIN 14-feature model).

Question: fit signed home MARGIN with Ridge, convert to P(home win) via a
Gaussian link p = Phi(pred_margin / sigma_resid). Does that beat the direct
logistic on the same 14 features?

Variants (DEV 5-fold CV, folds IDENTICAL to cv_ll: KFold(5, shuffle=True,
random_state=7) over np.where(dev)[0], gate delta >= 0.0020):
  (a) replace head, Ridge(alpha=1.0), ties KEPT in ridge training (margin 0)
  (b) alpha swept {0.1, 1, 10, 100} -- best reported but marked SWEPT
  (c) hybrid: margin-head probability logit as honest 15th column
      (nested inner 5-fold OOF for the train-side column, ridge refit on the
      full outer train for the va-side column; logistic drops ties as cv_ll)

DEV ONLY. No TEST look.
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
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
assert abs(cur_cv - 0.62300) < 0.0007, f"repro check FAILED: {cur_cv:.5f}"

# ---- signed home margins joined by gid ----
_mg = {}
for _r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")):
    if _r["home_score"] != "":
        _mg[_r["game_id"]] = float(_r["home_score"]) - float(_r["away_score"])
margin = np.array([_mg.get(g["gid"], np.nan) for g in games])
n_missing_all = int(np.isnan(margin).sum())
n_missing_dev = int(np.isnan(margin[dev]).sum())
print(f"margin coverage: {len(games)-n_missing_all}/{len(games)} joined "
      f"({n_missing_dev} missing on DEV)", flush=True)
assert n_missing_dev == 0, "missing margins inside DEV mask"
# sanity: signed margin must agree with the win label where both exist
_ok = ~np.isnan(margin)
_disagree = int(np.sum(_ok & (((margin > 0) & (y == 0.0)) | ((margin < 0) & (y == 1.0)))))
print(f"margin/label disagreements: {_disagree}", flush=True)

IDX = np.where(dev)[0]
FOLDS = list(KFold(5, shuffle=True, random_state=7).split(IDX))  # identical folds

# ---------------- (a)/(b) replace head: ridge margin -> Phi(m/sigma) ----------------
def cv_margin_head(X, alpha):
    """Per-fold ridge on signed margin (ties kept), Gaussian link, llv on va."""
    sc = np.zeros(len(IDX))
    sig = []
    oof_pred = np.full(len(IDX), np.nan)
    for k, (tr, va) in enumerate(FOLDS):
        itr, iva = IDX[tr], IDX[va]
        rg = Ridge(alpha=alpha).fit(X[itr], margin[itr])   # keep ties, drop nothing
        res = margin[itr] - rg.predict(X[itr])
        sigma = res.std()
        pm_va = rg.predict(X[iva])
        p_va = norm.cdf(pm_va / sigma)
        sc[va] = llv(y[iva], p_va)
        sig.append(float(sigma))
        oof_pred[va] = pm_va
    return sc.mean(), sig, oof_pred

results = {}
sigmas_a = None
oof_a = None
print("\nMargin head, REPLACE (DEV 5-fold, gate delta >= 0.0020):", flush=True)
for alpha in (0.1, 1.0, 10.0, 100.0):
    s, sig, oof = cv_margin_head(X14, alpha)
    tag = "  <- (a) headline" if alpha == 1.0 else "  (swept)"
    results[f"replace_a{alpha:g}"] = round(float(s), 5)
    print(f"  Ridge a={alpha:<5g} CV {s:.5f}  (delta {cur_cv-s:+.5f}){tag}", flush=True)
    if alpha == 1.0:
        sigmas_a, oof_a = sig, oof

# diagnostics: OOF predicted margin vs elo logit col 0, and sigma
corr_elo = float(np.corrcoef(oof_a, X14[IDX, 0])[0, 1])
corr_margin = float(np.corrcoef(oof_a, margin[IDX])[0, 1])
print(f"\ndiagnostics (alpha=1.0):", flush=True)
print(f"  per-fold sigma: {[round(s_,2) for s_ in sigmas_a]}  mean {np.mean(sigmas_a):.2f}", flush=True)
print(f"  corr(OOF pred margin, elo logit X14[:,0]) on DEV = {corr_elo:.4f}", flush=True)
print(f"  corr(OOF pred margin, actual margin)      on DEV = {corr_margin:.4f}", flush=True)

# ---------------- (c) hybrid: honest 15th column = margin-head prob logit ----------------
def cv_hybrid(X, alpha=1.0):
    sc = np.zeros(len(IDX))
    for k, (tr, va) in enumerate(FOLDS):
        itr, iva = IDX[tr], IDX[va]
        # inner 5-fold OOF margin predictions for the TRAIN side (honest column)
        oin = np.zeros(len(itr))
        for tr2, va2 in KFold(5, shuffle=True, random_state=11).split(itr):
            rg2 = Ridge(alpha=alpha).fit(X[itr[tr2]], margin[itr[tr2]])
            oin[va2] = rg2.predict(X[itr[va2]])
        sigma_in = (margin[itr] - oin).std()          # honest OOF residual scale
        p_tr = np.clip(norm.cdf(oin / sigma_in), eps, 1 - eps)
        lg_tr = np.log(p_tr / (1 - p_tr))
        # va side: ridge refit on full outer train
        rg = Ridge(alpha=alpha).fit(X[itr], margin[itr])
        p_va = np.clip(norm.cdf(rg.predict(X[iva]) / sigma_in), eps, 1 - eps)
        lg_va = np.log(p_va / (1 - p_va))
        Xtr = np.column_stack([X[itr], lg_tr])
        Xva = np.column_stack([X[iva], lg_va])
        keep = y[itr] != 0.5                          # logistic drops ties like cv_ll
        clf = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr[keep], y[itr][keep])
        sc[va] = llv(y[iva], clf.predict_proba(Xva)[:, 1])
    return sc.mean(), clf.coef_[0]

s_hyb, last_coef = cv_hybrid(X14, 1.0)
results["hybrid_15col_a1"] = round(float(s_hyb), 5)
print(f"\n  (c) hybrid 15th col: CV {s_hyb:.5f}  (delta {cur_cv-s_hyb:+.5f})", flush=True)
print(f"      last-fold logistic coef on margin-logit col: {last_coef[-1]:+.4f}", flush=True)

# ---------------- summary table ----------------
print("\n==== SUMMARY (DEV 5-fold, base {:.5f}, gate delta >= 0.0020) ====".format(cur_cv), flush=True)
print(f"  {'variant':<20} {'CV':>8} {'delta':>9}  gate", flush=True)
best_name, best_s = None, None
for name, s in results.items():
    d = cur_cv - s
    hit = "PASS" if d >= 0.0020 else "-"
    print(f"  {name:<20} {s:>8.5f} {d:>+9.5f}  {hit}", flush=True)
    if best_s is None or s < best_s:
        best_name, best_s = name, s
print(f"  best: {best_name} (delta {cur_cv-best_s:+.5f})", flush=True)

out = {
    "base_cv": round(float(cur_cv), 5),
    "variants": {k: v for k, v in results.items()},
    "deltas": {k: round(float(cur_cv - v), 5) for k, v in results.items()},
    "gate": 0.0020,
    "gate_met": bool((cur_cv - best_s) >= 0.0020),
    "best": best_name,
    "sigma_per_fold_a1": [round(s_, 3) for s_ in sigmas_a],
    "sigma_mean_a1": round(float(np.mean(sigmas_a)), 3),
    "corr_oof_pred_margin_vs_elo_logit": round(corr_elo, 4),
    "corr_oof_pred_margin_vs_actual_margin": round(corr_margin, 4),
    "margin_coverage": {"joined": len(games) - n_missing_all,
                        "total": len(games), "missing_dev": n_missing_dev,
                        "label_disagreements": _disagree},
    "notes": "alpha=1.0 is the pre-registered (a); other alphas swept (b). "
             "hybrid (c) uses nested inner 5-fold OOF for the train column.",
}
json.dump(out, open("data/nfl_margin_head.json", "w"), indent=1)
print(f"\n[{time.time()-T0:.0f}s] wrote data/nfl_margin_head.json", flush=True)
