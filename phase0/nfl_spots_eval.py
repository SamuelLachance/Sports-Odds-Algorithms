"""Conditional 'spots' screen: the four untested flags from the heuristics list.

  FATIGUE  away team on short week (<=5d) AND crossing >=1 timezone AND night kick
  HFAWX    outdoor game in bad weather (temp<=32F or wind>=15) — HFA amplifier
  DOMECOLD away team is a dome-home team, game outdoors and cold (<=40F)
  FINALWK  final regular-season week (motivation asymmetries)
DEV tryout together and individually; ONE TEST look only if something clears.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102

raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
for g, r in zip(games, raw):
    g["roof"] = r["roof"]
    g["temp"] = float(r["temp"]) if r["temp"] else None
    g["wind"] = float(r["wind"]) if r["wind"] else None
    g["gtype"] = r["game_type"]

TZ = {"BUF": 0, "MIA": 0, "NE": 0, "NYJ": 0, "BAL": 0, "CIN": 0, "CLE": 0, "PIT": 0,
      "IND": 0, "JAX": 0, "ATL": 0, "CAR": 0, "TB": 0, "WAS": 0, "NYG": 0, "PHI": 0,
      "DET": 0, "CHI": 1, "GB": 1, "MIN": 1, "DAL": 1, "HOU": 1, "TEN": 1, "KC": 1,
      "NO": 1, "DEN": 2, "ARI": 2, "SEA": 3, "SF": 3, "LAC": 3, "LV": 3, "LA": 3}
def tz_of(t, s): return 1 if (t == "LA" and s < 2016) else TZ[t]

# as-of dome-home map: majority roof of the team's home games in the PREVIOUS season
home_roofs = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # season -> team -> [dome, tot]
for g in games:
    hr = home_roofs[g["season"]][g["home"]]
    hr[1] += 1
    if g["roof"] in ("dome", "closed"):
        hr[0] += 1
def is_dome_team(team, season):
    hr = home_roofs.get(season - 1, {}).get(team)
    return bool(hr and hr[1] > 0 and hr[0] / hr[1] > 0.5)

f_fat = np.zeros(len(games)); f_wx = np.zeros(len(games))
f_dc = np.zeros(len(games)); f_fw = np.zeros(len(games))
for i, g in enumerate(games):
    s = g["season"]
    if (g["arest"] <= 5 and tz_of(g["away"], s) != tz_of(g["home"], s)
            and g["kick"] >= 19.0):
        f_fat[i] = 1.0
    if g["roof"] == "outdoors" and ((g["temp"] is not None and g["temp"] <= 32)
                                    or (g["wind"] or 0) >= 15):
        f_wx[i] = 1.0
    if (g["roof"] == "outdoors" and g["temp"] is not None and g["temp"] <= 40
            and is_dome_team(g["away"], s)):
        f_dc[i] = 1.0
    if (s <= 2020 and g["week"] == 17 and g["gtype"] == "REG") or (s >= 2021 and g["week"] == 18):
        f_fw[i] = 1.0
for nm, f_ in (("fatigue", f_fat), ("hfa-wx", f_wx), ("dome-cold", f_dc), ("final-wk", f_fw)):
    print(f"  {nm}: {int(f_.sum())} flagged games")

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

X_cur = X_of(F)
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}")
CAND = {"fatigue": [f_fat], "hfa-wx": [f_wx], "dome-cold": [f_dc], "final-wk": [f_fw],
        "all four": [f_fat, f_wx, f_dc, f_fw]}
best = None
for name, cols in CAND.items():
    X = np.column_stack([X_cur] + [c.reshape(-1, 1) for c in cols])
    s = cv_ll(X)
    tag = ""
    if best is None or s < best[0]:
        best = (s, name, X); tag = "  <-- best"
    print(f"  + {name:<10} DEV CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, name, Xn = best
if s >= base_cv:
    print("\nno DEV improvement -> conditional spots rejected at tryouts, no TEST look")
    json.dump({"verdict": "dev_rejected", "dev_cv_base": round(base_cv, 5),
               "dev_cv": round(s, 5), "variant": name},
              open("data/nfl_spots.json", "w"), indent=1)
    raise SystemExit(0)

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
better = llv(yt, p1).mean() < llv(yt, p0).mean()
print(f"\nmain TEST ({name}): current {llv(yt, p0).mean():.5f} -> {llv(yt, p1).mean():.5f}"
      f"  (delta {dd.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}])  -> "
      f"{'ADOPT' if better else 'keep current'}")
json.dump({"variant": name, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_spots.json", "w"), indent=1)
