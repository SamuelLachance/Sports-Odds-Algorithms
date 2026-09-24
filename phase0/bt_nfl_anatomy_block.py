"""Breakthrough program 2026-09-24 -- NFL anatomy: is the 2013-2015 closure CAUSED by the
player-availability block? (DEV only)

Era split showed the QB-flux and week-17 resting gaps vanish in 2013-2015, the
seasons where absence OL/DEF/skill + roster_quality are live. Counterfactual:
re-run the shipped walk-forward protocol WITHOUT that block and re-slice
2013-2015. If the gaps reappear, the block is what closes them. Odds = evaluation.
Output data/bt_nfl_anatomy_block.json.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

src2 = open("phase0/bt_nfl_anatomy2.py", encoding="utf-8").read()
exec(src2.split("# ---------------------------------------------------------------- E. gap budget")[0])  # noqa: S102

rows_all = list(csv.DictReader(open("data/bt_nfl_anatomy_dev.csv", encoding="utf-8")))
assert max(int(r["season"]) for r in rows_all) <= 2015
COLS = ["lgt", "qd", "epa", "early", "thfa", "rest", "luck", "v7", "ol", "de", "sk", "rq", "pass", "run"]
XA = np.array([[float(r[c]) for c in COLS] for r in rows_all])
yA = np.array([float(r["y"]) for r in rows_all]); sA = np.array([int(r["season"]) for r in rows_all])
gA = [r["gid"] for r in rows_all]


def wf(cols):
    idx = [COLS.index(c) for c in cols]
    pp = np.full(len(yA), np.nan)
    for s_ in range(2006, 2016):
        tr = (sA < s_) & (yA != 0.5); te = sA == s_
        w = 0.5 ** ((s_ - 1 - sA[tr]) / 3.0)
        m = LogisticRegression(C=100.0, max_iter=5000).fit(XA[tr][:, idx], yA[tr], sample_weight=w)
        pp[te] = m.predict_proba(XA[te][:, idx])[:, 1]
    return dict(zip(gA, pp))


qb_flux = np.array([(g["h"]["is_estab"] == 0) or (g["a"]["is_estab"] == 0) or (g["h"]["qb_changed"] == 1)
                    or (g["a"]["qb_changed"] == 1) for g in G])
pin_h = np.array([ps(i, "h", "playoff_in") for i in range(n)], bool)
pin_a = np.array([ps(i, "a", "playoff_in") for i in range(n)], bool)
ext_pd = okpd & ((pd_res <= np.nanpercentile(pd_res, 20)) | (pd_res >= np.nanpercentile(pd_res, 80)))
late = SEA >= 2013
OUT = {}
for arm, cols in (("with block (csv refit)", COLS), ("WITHOUT absence+rq block", [c for c in COLS if c not in ("ol", "de", "sk", "rq")])):
    pmap = wf(cols)
    p = np.array([pmap[g["gid"]] for g in G])
    llm = llv(y, p); d = llm - llk; TOT = d.sum(); r_dis = logit(pm) - logit(p)
    rows = []
    for nm, m in (("2013-15 all", late), ("2013-15 QB flux", late & qb_flux), ("2013-15 QB stable", late & ~qb_flux),
                  ("2013-15 week 17", late & REG & (W == 17)),
                  ("2013-15 wk17 one team clinched playoff", late & REG & (W == 17) & (pin_h ^ pin_a)),
                  ("2013-15 PD-extreme", late & ext_pd), ("2013-15 PD-middle", late & okpd & ~ext_pd)):
        r = summarize(m, nm)
        rows.append(r)
        print(f"  [{arm}] {nm:<42} n={r['n']:>4} model_ll={r['model_ll']:.4f} gap={r['gap']:+.4f} ci={r['gap_ci']}")
    OUT[arm] = rows
json.dump(OUT, open("data/bt_nfl_anatomy_block.json", "w"), indent=1)
print("wrote data/bt_nfl_anatomy_block.json")
