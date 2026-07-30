"""Formalize the NFL model spec.

V2 = elo_logit + qb_diff(pedigree prior) + epa_net + early_kick + team_hfa
     + rest_diff + fumble_luck + ts_edge   (the V4 stack from nfl_big_test.py)

Outputs data/nfl_model.json with the complete reproducible spec:
component params, DEV-fit coefficients (the TEST-evaluated model, LL 0.62743),
and production coefficients refit walk-forward on ALL completed seasons.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

# reuse the exact feature construction from the big test
src = open("phase0/nfl_big_test.py").read()
exec(src.split("# ---------- variants, DEV CV selection ----------")[0])  # noqa: S102

FEATS = ["elo_logit", "qb_pedigree", "epa_net", "early_kick", "team_hfa",
         "rest_diff", "fumble_luck", "ts_edge"]
X = np.column_stack([lgt, qd_ped, f_epa, f_early, f_thfa, f_rest, f_luck, f_ts])


def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


# ---- 1. verification: DEV-fit -> TEST (must reproduce the big-test numbers) ----
fit = dev & (y != 0.5)
clf_dev = LogisticRegression(C=1e6, max_iter=5000).fit(X[fit], y[fit])
pw = clf_dev.predict_proba(X[test])[:, 1]
BASE = np.column_stack([lgt, qd])
cb = LogisticRegression(C=1e6, max_iter=5000).fit(BASE[fit], y[fit])
pb = cb.predict_proba(BASE[test])[:, 1]
yt = y[test]
ll_v2, ll_v1 = llv(yt, pw).mean(), llv(yt, pb).mean()
print(f"TEST 2016-2025: v1 (elo+qb) {ll_v1:.5f}  ->  V2 {ll_v2:.5f}  "
      f"(delta {ll_v1-ll_v2:+.5f})")

# ---- 2. production coefficients: walk-forward fit on ALL completed seasons ----
allfit = (seasons >= DEV_SCORE_FROM) & (y != 0.5)
clf_all = LogisticRegression(C=1e6, max_iter=5000).fit(X[allfit], y[allfit])
print("\nproduction blend (fit 2001-2025):")
for name, c in zip(FEATS, clf_all.coef_[0]):
    print(f"  {name:<12} {c:+.5f}")
print(f"  intercept    {clf_all.intercept_[0]:+.5f}")

spec = {
    "model": "NFL GLASSBOX", "adopted": "2026-07-23",
    "decision": "Samuel: better TEST point estimate wins (v1 kept as reference)",
    "test_ll_v2": round(float(ll_v2), 5), "test_ll_v1": round(float(ll_v1), 5),
    "features": FEATS,
    "component_params": {
        "elo": base["params"],
        "qb": {**{k: SP[k] for k in ("decay", "prior_db", "season_decay")},
               "delta": pedP["delta"], "pedigree_c": pedP["c"],
               "pedigree_scale": pedP["scale"], "lg_rate": lg_qb},
        "epa": {**epaP, "lg_epa": round(LG_EPA, 5)},
        "ts": tsP,
        "team_hfa": {"decay": 0.991, "prior_n": 180.0},
        "fumble_luck": luckP,
        "rest": {"clip": 7}, "early_kick": {"tz_gap_min": 2, "latest_et": 14.0},
    },
    "blend_dev_fit": {"coef": [round(float(c), 6) for c in clf_dev.coef_[0]],
                      "intercept": round(float(clf_dev.intercept_[0]), 6)},
    "blend_production": {"coef": [round(float(c), 6) for c in clf_all.coef_[0]],
                         "intercept": round(float(clf_all.intercept_[0]), 6),
                         "fit": "2001-2025 all completed, ties dropped"},
}
json.dump(spec, open("data/nfl_model.json", "w"), indent=1)
print("\nwrote data/nfl_model.json  (NFL model = V2)")
