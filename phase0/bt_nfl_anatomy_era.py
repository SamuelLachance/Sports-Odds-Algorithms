"""Breakthrough program 2026-09-24 -- NFL gap anatomy, era split (DEV only).

The shipped model's player-availability columns (absence OL/DEF/skill + roster
quality) are live on DEV only from 2013 (snap counts); the v7 participation
ratings are zero on all of DEV. 2013-2015 is therefore the DEV window whose
information set is closest to the shipped TEST-era model. This script splits
every main mechanism by era (2006-2012 no player columns / 2013-2015 live) and
gives a 2013-2015 gap budget. Odds = evaluation only.
Output data/bt_nfl_anatomy_era.json.
"""
from __future__ import annotations

import json

import numpy as np

src2 = open("phase0/bt_nfl_anatomy2.py", encoding="utf-8").read()
exec(src2.split("# ---------------------------------------------------------------- E. gap budget")[0])  # noqa: S102

qb_flux = np.array([(g["h"]["is_estab"] == 0) or (g["a"]["is_estab"] == 0) or (g["h"]["qb_changed"] == 1)
                    or (g["a"]["qb_changed"] == 1) for g in G])
ext_pd = okpd & ((pd_res <= np.nanpercentile(pd_res, 20)) | (pd_res >= np.nanpercentile(pd_res, 80)))
pin_h = np.array([ps(i, "h", "playoff_in") for i in range(n)], bool)
pin_a = np.array([ps(i, "a", "playoff_in") for i in range(n)], bool)
absx = np.array([g["X"]["ol"] + g["X"]["de"] + g["X"]["sk"] for g in G])
OUT = {}
for lab, era in (("2006-2012", SEA <= 2012), ("2013-2015", SEA >= 2013)):
    rows = []
    for nm, m in (("all", np.ones(n, bool)), ("QB flux (either side non-estab or changed)", qb_flux),
                  ("QB stable both sides", ~qb_flux), ("PD-residual extreme 40%", ext_pd),
                  ("PD-residual middle 60%", okpd & ~ext_pd), ("weeks 1-2", REG & (W <= 2)),
                  ("weeks 3-8", REG & (W >= 3) & (W <= 8)), ("weeks 9-16", REG & (W >= 9) & (W <= 16)),
                  ("week 17", REG & (W == 17)),
                  ("wk17 one team clinched playoff", REG & (W == 17) & (pin_h ^ pin_a)),
                  ("playoffs", ~REG), ("primetime", kick >= 19), ("primetime & PD-extreme", (kick >= 19) & ext_pd),
                  ("primetime & not PD-extreme", (kick >= 19) & ~ext_pd),
                  ("|O+D injury diff|>=2", has_inj & (np.abs(np.nan_to_num(od_diff)) >= 2)),
                  ("home market favourite", pm >= 0.5), ("away market favourite", pm < 0.5),
                  ("different favourite (model vs close)", np.sign(logit(p)) != np.sign(logit(pm)))):
        r = summarize(m & era, f"{lab}: {nm}")
        r["share_of_era_gap"] = round(float(d[m & era].sum() / d[era].sum()), 4) if (m & era).any() else None
        rows.append(r)
        print(f"  {r['slice'][:60]:<60} n={r['n']:>4} gap={r['gap']:+.4f} ci={r['gap_ci']} era_share={r['share_of_era_gap']}")
    OUT[lab] = rows
m2 = REG & (W == 17) & (pin_h ^ pin_a)
OUT["wk17_clinched_detail"] = {}
for lab, era in (("2006-2012", SEA <= 2012), ("2013-2015", SEA >= 2013)):
    mm = m2 & era
    OUT["wk17_clinched_detail"][lab] = {
        "n": int(mm.sum()),
        "model_p_clinched": round(float(np.mean(np.where(pin_h, p, 1 - p)[mm])), 3),
        "close_p_clinched": round(float(np.mean(np.where(pin_h, pm, 1 - pm)[mm])), 3),
        "win_rate_clinched": round(float(np.mean(np.where(pin_h, y, 1 - y)[mm])), 3),
        "mean_oriented_absence_feature": round(float(np.mean(np.where(pin_h, absx, -absx)[mm])), 3)}
print(OUT["wk17_clinched_detail"])
json.dump(OUT, open("data/bt_nfl_anatomy_era.json", "w"), indent=1)
print("wrote data/bt_nfl_anatomy_era.json")

# ---------------- per-season stability of the main carriers
PS = []
for s in range(2006, 2016):
    e = SEA == s
    row = {"season": s}
    for nm, m in (("qb_flux", qb_flux), ("qb_stable", ~qb_flux), ("pd_extreme", ext_pd), ("pd_middle", okpd & ~ext_pd),
                  ("week17", REG & (W == 17)), ("weeks1_2", REG & (W <= 2))):
        mm = m & e
        row[nm] = {"n": int(mm.sum()), "gap": round(float(d[mm].mean()), 4) if mm.any() else None}
    PS.append(row)
    print(s, {k: (v["n"], v["gap"]) for k, v in row.items() if k != "season"})
OUT["per_season"] = PS
json.dump(OUT, open("data/bt_nfl_anatomy_era.json", "w"), indent=1)
