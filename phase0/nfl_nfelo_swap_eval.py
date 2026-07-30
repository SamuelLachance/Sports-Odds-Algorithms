"""MEASUREMENT (phase0, not shippable without ending market-blindness): replace our
Elo logit with nfelo's base dif (market-tinted via resist) inside the 12-feature model.

nlgt = logit(1/(1+10^(-dif_base/400))) where nfelo covers the game; our lgt elsewhere
(pre-2009 + unjoined). Variants: replace lgt / add alongside. DEV CV selects; ONE TEST
look with CI. Adoption is a constitutional decision, not an automatic rule application.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102

NF = r"C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Projects-Sports-Odds-Algorithms/c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad/nfelo_games.csv"
dif = {}
with open(NF, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        gid = r.get("game_id") or ""
        try:
            dif[gid] = float(r["nfelo_dif_base"])
        except (KeyError, TypeError, ValueError):
            continue

nlgt = np.array(F["lgt"], copy=True)
hit = 0
for i, g in enumerate(games):
    d = dif.get(g["gid"])
    if d is not None:
        p = 1.0 / (1.0 + 10 ** (-d / 400.0))
        nlgt[i] = np.log(p / (1 - p))
        hit += 1
print(f"nfelo base joined: {hit}/{len(games)} games "
      f"({hit / len(games):.0%}; our lgt fills the rest)")

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

X_cur = X_of(F)
F_rep = dict(F); F_rep["lgt"] = nlgt
X_rep = X_of(F_rep)
X_add = np.column_stack([X_cur, nlgt])
base_cv = cv_ll(X_cur)
cvs = {"replace": cv_ll(X_rep), "add": cv_ll(X_add)}
print(f"DEV CV: current {base_cv:.5f}  replace {cvs['replace']:.5f}  add {cvs['add']:.5f}")
name = min(cvs, key=cvs.get)
Xn = X_rep if name == "replace" else X_add
fitm = dev & (y != 0.5)
c0 = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[fitm], y[fitm])
c1 = LogisticRegression(C=1e6, max_iter=5000).fit(Xn[fitm], y[fitm])
p0 = c0.predict_proba(X_cur[test])[:, 1]
p1 = c1.predict_proba(Xn[test])[:, 1]
yt = y[test]
dd = llv(yt, p0) - llv(yt, p1)
rng = np.random.default_rng(7)
n = len(dd)
bs = dd[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nmain TEST ({name}): current {llv(yt, p0).mean():.5f} -> {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])")
print("NOTE: adoption = ending market-blindness (nfelo base carries market bleed).")
json.dump({"variant": name, "dev_cv": {k: round(v, 5) for k, v in cvs.items()},
           "dev_cv_base": round(base_cv, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)]},
          open("data/nfl_nfelo_swap.json", "w"), indent=1)
