"""Breakthrough program 2026-09-24 -- NFL gap anatomy, step 5 (market-blind learnability).

DEV ONLY. For each mechanism the anatomy located, ask the OUTCOMES (no odds) whether
the information is recoverable: leave-one-season-out logistic
    base : y ~ a + b*logit(p_model)                (recalibration only)
    plus : y ~ a + b*logit(p_model) + c*v          (candidate signal)
gain = LL(base) - LL(plus), paired bootstrap CI, per variable. This is a
diagnostic ranking (LOSO uses future DEV seasons to fit c), NOT a screen against
the pre-registered bar. Odds appear only in the 'market_gap_in_support' column.
Output data/bt_nfl_anatomy4.json.
"""
from __future__ import annotations

import json

import numpy as np

src2 = open("phase0/bt_nfl_anatomy2.py", encoding="utf-8").read()
exec(src2.split("# ---------------------------------------------------------------- E. gap budget")[0])  # noqa: S102

LP = logit(p)


def fit_logit(X, yy, iters=60, ridge=1e-6):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ w)))
        H_ = (X * (mu * (1 - mu))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        st_ = np.linalg.solve(H_, X.T @ (yy - mu) - ridge * w)
        w += st_
        if np.abs(st_).max() < 1e-10:
            break
    return w


def loso_free(v):
    v = np.nan_to_num(v)
    Xb = np.column_stack([np.ones(n), LP]); Xp = np.column_stack([np.ones(n), LP, v])
    lb = np.zeros(n); lp_ = np.zeros(n)
    for s in range(2006, 2016):
        te = SEA == s; tr = ~te
        wb = fit_logit(Xb[tr], y[tr]); wp = fit_logit(Xp[tr], y[tr])
        lb[te] = llv(y[te], 1 / (1 + np.exp(-(Xb[te] @ wb))))
        lp_[te] = llv(y[te], 1 / (1 + np.exp(-(Xp[te] @ wp))))
    g = lb - lp_
    wall = fit_logit(Xp, y)
    return g, wall


def sig(v, cond):
    return np.where(cond, v, 0.0)


lock_bye_h = np.array([ps(i, "h", "bye") for i in range(n)], float)
lock_bye_a = np.array([ps(i, "a", "bye") for i in range(n)], float)
pin_h = np.array([ps(i, "h", "playoff_in") for i in range(n)], float)
pin_a = np.array([ps(i, "a", "playoff_in") for i in range(n)], float)
cdiv_h = np.array([ps(i, "h", "clinch_div") for i in range(n)], float)
cdiv_a = np.array([ps(i, "a", "clinch_div") for i in range(n)], float)
late = REG & (W >= 16)
wk17 = REG & (W == 17)
pdd0 = np.where(okpd, pdd, 0.0)
qb_nonest = np.array([int(g["h"]["is_estab"] == 0) - int(g["a"]["is_estab"] == 0) for g in G], float)
V = {
    "late: seed-locked (wk16-17) diff": sig(lock_h.astype(float) - lock_a.astype(float), late),
    "late: clinched div + bye-safe (wk16-17) diff": sig(lock_bye_h - lock_bye_a, late),
    "late: clinched division (wk17) diff": sig(cdiv_h - cdiv_a, wk17),
    "late: clinched playoff spot (wk17) diff": sig(pin_h - pin_a, wk17),
    "late: eliminated (wk14+) diff": sig(elim_h.astype(float) - elim_a.astype(float), REG & (W >= 14)),
    "margin: season PD/g diff (>=3 games, else 0)": pdd0,
    "margin: season PD/g diff, weeks 9+ only": np.where(okpd & (W >= 9), pdd, 0.0),
    "offseason: prior PD/g beyond record (wk1-4)": np.where(np.isnan(pyth_d), 0.0, pyth_d),
    "offseason: prior-season PD/g (wk1-4)": np.where(wk14 & ~np.isnan(ppd_d), ppd_d, 0.0),
    "offseason: QB != last season primary (wk1-4)": np.where(wk14, OFFV["QB != last season primary (diff)"], 0.0),
    "qb: non-established starter diff": qb_nonest,
    "inj: Out+Doubtful count diff (2009+)": np.nan_to_num(od_diff),
    "inj: snap-weighted O+D diff (2012+)": np.nan_to_num(wod),
    "coach: interim diff": np.array([g["h"]["interim"] - g["a"]["interim"] for g in G], float),
}
rows_ = []
for name, v in V.items():
    g, wall = loso_free(v)
    nz = v != 0
    rows_.append({"var": name, "n_nonzero": int(nz.sum()), "coef_all_dev": round(float(wall[2]), 4),
                  "slope_on_logit_p": round(float(wall[1]), 4),
                  "loso_gain": round(float(g.mean()), 5), "loso_ci": boot_ci(g),
                  "loso_gain_on_nonzero_games": round(float(g[nz].mean()), 5) if nz.any() else None,
                  "market_gap_in_support": round(float(d[nz].mean()), 5) if nz.any() else None,
                  "market_gap_share_in_support": round(float(d[nz].sum() / d.sum()), 4) if nz.any() else None})
    print(f"{name[:50]:<50} nz={int(nz.sum()):>4} coef={wall[2]:+.4f} loso={g.mean():+.5f} "
          f"ci={rows_[-1]['loso_ci']} gap_in_support={rows_[-1]['market_gap_in_support']} "
          f"share={rows_[-1]['market_gap_share_in_support']}")
# recalibration alone (for context)
Xb = np.column_stack([np.ones(n), LP])
lb = np.zeros(n)
for s in range(2006, 2016):
    te = SEA == s
    wb = fit_logit(Xb[~te], y[~te]); lb[te] = llv(y[te], 1 / (1 + np.exp(-(Xb[te] @ wb))))
rec = {"loso_recal_gain_vs_shipped": round(float((llm - lb).mean()), 5), "ci": boot_ci(llm - lb),
       "slope_all_dev": [round(float(v), 4) for v in fit_logit(Xb, y)]}
print("recalibration:", rec)
json.dump({"recalibration": rec, "vars": rows_}, open("data/bt_nfl_anatomy4.json", "w"), indent=1)
print("wrote data/bt_nfl_anatomy4.json")
