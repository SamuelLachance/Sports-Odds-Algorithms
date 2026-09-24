"""Breakthrough program 2026-09-24 -- NFL gap anatomy, step 4 (QB-in-flux mechanism).

DEV ONLY. Joins data/bt_nfl_anatomy_qb.csv (as-of shipped QB ratings of current
and established starters) onto the anatomy frame and asks which mechanism the
market exploits in QB-in-flux games:
  H1 stale team rating  -> r and the outcome residual load on the oriented
                           rating CHANGE (q_cur - q_estab) of the side in flux
  H2 over-shrunk starter -> r loads on (raw EPA/db - shrunk rating) of low-
                           evidence current starters
Odds = evaluation only. Output data/bt_nfl_anatomy3.json.
"""
from __future__ import annotations

import csv
import json

import numpy as np

src = open("phase0/bt_nfl_anatomy.py", encoding="utf-8").read()
exec(src.split("def team_diff(key")[0])  # noqa: S102

QB = {r["gid"]: r for r in csv.DictReader(open("data/bt_nfl_anatomy_qb.csv", encoding="utf-8"))}
assert max(int(r["season"]) for r in QB.values()) <= 2015
f = lambda i, k: float(QB[G[i]["gid"]][k])
qcur_h = np.array([f(i, "h_q_cur") for i in range(n)]); qcur_a = np.array([f(i, "a_q_cur") for i in range(n)])
qest_h = np.array([f(i, "h_q_estab") for i in range(n)]); qest_a = np.array([f(i, "a_q_estab") for i in range(n)])
est_h = np.array([f(i, "h_is_estab") for i in range(n)]) == 1; est_a = np.array([f(i, "a_is_estab") for i in range(n)]) == 1
db_h = np.array([f(i, "h_db") for i in range(n)]); db_a = np.array([f(i, "a_db") for i in range(n)])
raw_h = np.array([f(i, "h_raw") for i in range(n)]); raw_a = np.array([f(i, "a_raw") for i in range(n)])
stint_h = np.array([f(i, "h_stint") for i in range(n)]); stint_a = np.array([f(i, "a_stint") for i in range(n)])
qd_x = np.array([g["X"]["qd"] for g in G])
chk = float(np.abs((qcur_h - qcur_a) - qd_x).max())
print("max |(q_cur_h - q_cur_a) - shipped qd| =", chk)

OUT = {"qd_reproduction_max_abs_err": chk}
# ---------------- H1: oriented rating change of the side in flux
one = est_h ^ est_a                                   # exactly one side in flux
orient = np.where(~est_h, 1.0, -1.0)                  # +1 if HOME is the side in flux
dq = np.where(~est_h, qcur_h - qest_h, qcur_a - qest_a)   # new minus established (<0 downgrade)
r_o = orient * r_dis
res_o = orient * (y - p)
def t_of(x, yy):
    a = ols_t(x, yy)
    return None if a is None else {"slope": round(a[0], 4), "t": round(a[1], 2), "corr": round(a[3], 4), "n": a[2]}
OUT["H1_one_side_in_flux"] = {
    "n": int(one.sum()),
    "r_oriented_on_dq": t_of(np.where(one, dq, np.nan), r_o),
    "outcome_resid_oriented_on_dq": t_of(np.where(one, dq, np.nan), res_o),
    "mean_dq": round(float(dq[one].mean()), 4), "sd_dq": round(float(dq[one].std()), 4)}
S = []
qs = np.nanpercentile(np.where(one, dq, np.nan), [25, 50, 75])
for lab, msk in (("big downgrade (dq <= q25)", one & (dq <= qs[0])),
                 ("mild downgrade (q25,q50]", one & (dq > qs[0]) & (dq <= qs[1])),
                 ("mild change (q50,q75]", one & (dq > qs[1]) & (dq <= qs[2])),
                 ("upgrade/near-equal (dq > q75)", one & (dq > qs[2]))):
    S.append(summarize(msk, lab, orient))
OUT["H1_slices_by_dq"] = {"quartiles": [round(float(v), 4) for v in qs], "slices": S}
# stint phase: 1st start of stint vs 2-4 vs 5+
S2 = []
stint = np.where(~est_h, stint_h, stint_a)
for lab, msk in (("first start of stint", one & (stint == 0)), ("starts 2-4 of stint", one & (stint >= 1) & (stint <= 3)),
                 ("starts 5+ of stint (still non-estab)", one & (stint >= 4))):
    S2.append(summarize(msk, lab, orient))
OUT["H1_slices_by_stint"] = S2

# ---------------- H2: over-shrinkage of low-evidence current starters
dbo = np.where(~est_h, db_h, db_a)
rawo = np.where(~est_h, raw_h, raw_a)
qcuro = np.where(~est_h, qcur_h, qcur_a)
gapq = rawo - qcuro                                    # raw rate minus shrunk rating
low = one & (dbo > 20) & (dbo < 400) & ~np.isnan(rawo)
OUT["H2_low_evidence_in_flux"] = {
    "n": int(low.sum()),
    "r_oriented_on_raw_minus_shrunk": t_of(np.where(low, gapq, np.nan), r_o),
    "outcome_resid_on_raw_minus_shrunk": t_of(np.where(low, gapq, np.nan), res_o),
    "r_oriented_on_raw_rate": t_of(np.where(low, rawo, np.nan), r_o),
    "outcome_resid_on_raw_rate": t_of(np.where(low, rawo, np.nan), res_o)}
# all starters (not only in flux): shrinkage association over evidence bins, home-away diff
gq_h = np.where(np.isnan(raw_h), 0.0, raw_h - qcur_h); gq_a = np.where(np.isnan(raw_a), 0.0, raw_a - qcur_a)
OUT["H2_all_games_diff"] = {"r_on_(raw-shrunk)_diff": t_of(gq_h - gq_a, r_dis),
                            "outcome_resid_on_(raw-shrunk)_diff": t_of(gq_h - gq_a, y - p)}

# ---------------- H1 generalized: every game, the Elo-embedded QB mismatch (both sides)
# the team Elo is QB-agnostic; its embedded QB is ~ the established starter. Mismatch per side:
mm_h = qcur_h - qest_h; mm_a = qcur_a - qest_a
OUT["H1_all_games"] = {"r_on_mismatch_diff": t_of(mm_h - mm_a, r_dis),
                       "outcome_resid_on_mismatch_diff": t_of(mm_h - mm_a, y - p),
                       "outcome_offset_logit": None}
xm = mm_h - mm_a
w_, se_ = offset_logit(xm, logit(p), y)
lg_ = loso_gain(xm)
OUT["H1_all_games"]["outcome_offset_logit"] = {"coef": round(float(w_[1]), 3), "t": round(float(w_[1] / se_), 2),
                                               "loso_gain": round(lg_[0], 5), "loso_ci": lg_[1],
                                               "blend_qd_coef_note": "compare with qd's own blend coefficient (data/bt_nfl_anatomy_coefs.json col qd)"}
# market-implied weight on the mismatch vs on qd itself (both regressions on r, joint)
Aj = np.column_stack([np.ones(n), xm, qd_x, logit(p)])
bj = np.linalg.lstsq(Aj, logit(pm), rcond=None)[0]
OUT["market_projection_logit_close_on[1,mismatch,qd,logit_p]"] = [round(float(v), 4) for v in bj]
Ao = np.column_stack([xm, qd_x])
w2 = np.zeros(3)
X_ = np.column_stack([np.ones(n), Ao])
for _ in range(50):
    mu = 1 / (1 + np.exp(-(logit(p) + X_ @ w2)))
    H_ = (X_ * (mu * (1 - mu))[:, None]).T @ X_ + 1e-9 * np.eye(3)
    st_ = np.linalg.solve(H_, X_.T @ (y - mu)); w2 += st_
    if np.abs(st_).max() < 1e-10:
        break
OUT["outcome_offset_on[1,mismatch,qd]"] = [round(float(v), 4) for v in w2]
coefs = json.load(open("data/bt_nfl_anatomy_coefs.json"))
OUT["shipped_qd_coef_by_season"] = {s: c[1] for s, c in coefs["coefs_by_season"].items()}
OUT["shipped_lgt_coef_by_season"] = {s: c[0] for s, c in coefs["coefs_by_season"].items()}
json.dump(OUT, open("data/bt_nfl_anatomy3.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))

# ---------------- interplay: QB flux x margin extreme x early season (2x2x) with contrasts
qb_flux = np.array([(g["h"]["is_estab"] == 0) or (g["a"]["is_estab"] == 0) or (g["h"]["qb_changed"] == 1)
                    or (g["a"]["qb_changed"] == 1) for g in G])
ext_pd = okpd & ((pd_res <= np.nanpercentile(pd_res, 20)) | (pd_res >= np.nanpercentile(pd_res, 80)))
early = REG & (W <= 2)
X2 = []
for lab, msk in (("flux & PD-extreme", qb_flux & ext_pd), ("flux & not PD-extreme", qb_flux & ~ext_pd),
                 ("stable & PD-extreme", ~qb_flux & ext_pd), ("stable & not PD-extreme", ~qb_flux & ~ext_pd)):
    X2.append(summarize(msk, lab))
def contrast(m1, m2):
    d1, d2 = d[m1], d[m2]
    i1 = RNG.integers(0, len(d1), size=(NB, len(d1))); i2 = RNG.integers(0, len(d2), size=(NB, len(d2)))
    bs = d1[i1].mean(1) - d2[i2].mean(1)
    return {"diff": round(float(d1.mean() - d2.mean()), 5),
            "ci": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]}
OUT2 = {"cells": X2,
        "contrast_flux_vs_stable": contrast(qb_flux, ~qb_flux),
        "contrast_flux_vs_stable_within_notPD": contrast(qb_flux & ~ext_pd, ~qb_flux & ~ext_pd),
        "contrast_PDext_vs_not": contrast(ext_pd, okpd & ~ext_pd),
        "contrast_PDext_vs_not_within_stable": contrast(~qb_flux & ext_pd, ~qb_flux & okpd & ~ext_pd),
        "contrast_wk1_2_vs_rest": contrast(early, ~early)}
# budget in the other order (margin before QB)
taken = np.zeros(n, bool); B2 = []
hr_ = np.array([g["hrest"] for g in G]); ar_ = np.array([g["arest"] for g in G])
for lab, msk in (("margin extreme 40%", ext_pd), ("QB flux", qb_flux), ("weeks 1-2", early), ("playoffs", ~REG)):
    m_ = msk & ~taken; B2.append(summarize(m_, lab)); taken |= m_
B2.append(summarize(~taken, "remainder"))
OUT2["budget_margin_first"] = B2
OUT["interplay"] = OUT2
json.dump(OUT, open("data/bt_nfl_anatomy3.json", "w"), indent=1)
for r in X2 + B2:
    print(f"  {r['slice']:<28} n={r['n']:>4} gap={r['gap']:+.4f} ci={r['gap_ci']} share={r['share_of_gap']:+.3f} nsh={r['n_share']:.3f}")
for k, v in OUT2.items():
    if k.startswith("contrast"):
        print(k, v)
