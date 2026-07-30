"""NFL luck features (the MLB-style regression correctors), DEV screen -> one TEST look.

  CLOSE  one-score-game luck: EWMA of (result - 0.5) over past games decided by <=T
         points — close games ~ coin flips, Elo credits them fully -> regression.
  FG     kicker luck: EWMA of (made - p_expect(distance)) — distance curve fit on DEV.
  FUM    fumble-recovery luck (revisit, tuned params from earlier screen).
Variants: each alone + composite. Expected blend signs NEGATIVE (luck regresses).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102

# ---- margins (from games.csv scores) ----
raw = [r for r in csv.DictReader(open("data/nfl_games.csv")) if r["home_score"] != ""]
raw.sort(key=lambda r: (r["gameday"],))
margins = np.array([int(r["home_score"]) - int(r["away_score"]) for r in raw])

# ---- A. close-game luck ----
def close_luck(T, decay):
    st = defaultdict(lambda: [0.0, 0.0])
    f = np.zeros(len(games))
    for i, g in enumerate(games):
        h, a = g["home"], g["away"]
        f[i] = (st[h][0] / (st[h][1] + 4.0)) - (st[a][0] / (st[a][1] + 4.0))
        if abs(margins[i]) <= T and g["y"] != 0.5:
            for team, res in ((h, g["y"]), (a, 1.0 - g["y"])):
                s_ = st[team]
                s_[0] = decay * s_[0] + (res - 0.5)
                s_[1] = decay * s_[1] + 1.0
    return f

# ---- B. FG/kicker luck ----
fg_rows = []
with open("data/nfl_fgs.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    next(rd)
    for r in rd:
        fg_rows.append((r[0], PBP_FIX.get(r[1], r[1]), float(r[2]), int(r[3])))
fg_by_game = defaultdict(list)
for gid, t_, d_, m_ in fg_rows:
    fg_by_game[gid].append((t_, d_, m_))
# distance curve on DEV-era kicks
Xd = np.array([[d_] for gid, t_, d_, m_ in fg_rows if int(gid[:4]) <= 2015])
Yd = np.array([m_ for gid, t_, d_, m_ in fg_rows if int(gid[:4]) <= 2015])
fgcurve = LogisticRegression(C=1e6, max_iter=2000).fit(Xd, Yd)
print(f"FG curve (DEV): 40yd {fgcurve.predict_proba([[40]])[0, 1]:.3f}  "
      f"50yd {fgcurve.predict_proba([[50]])[0, 1]:.3f}  n={len(Yd):,}")

def fg_luck(decay):
    st = defaultdict(lambda: [0.0, 0.0])
    f = np.zeros(len(games))
    for i, g in enumerate(games):
        h, a = g["home"], g["away"]
        f[i] = (st[h][0] / (st[h][1] + 8.0)) - (st[a][0] / (st[a][1] + 8.0))
        for (t_, d_, m_) in fg_by_game.get(g["gid"], ()):
            pexp = fgcurve.predict_proba([[d_]])[0, 1]
            s_ = st[t_]
            s_[0] = decay * s_[0] + (m_ - pexp)
            s_[1] = decay * s_[1] + 1.0
    return f

def llv(yy, p):
    p = np.clip(p, eps, 1 - eps)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

X_cur = X_of(F)
base_cv = cv_ll(X_cur)
print(f"current model DEV CV {base_cv:.5f}")

f_close3 = close_luck(3, 0.95)
f_close8 = close_luck(8, 0.95)
f_fgl = fg_luck(0.95)
CAND = {
    "close<=3": [f_close3], "close<=8": [f_close8], "fg-luck": [f_fgl],
    "close8+fg": [f_close8, f_fgl],
    "composite (close8+fg+fum)": [f_close8, f_fgl, F["luck"]],
}
best = None
for name, cols in CAND.items():
    X = np.column_stack([X_cur] + cols)
    s = cv_ll(X)
    tag = ""
    if best is None or s < best[0]:
        best = (s, name, X); tag = "  <-- best"
    print(f"  + {name:<26} DEV CV {s:.5f} ({s-base_cv:+.5f}){tag}")

s, name, Xn = best
if s >= base_cv:
    print("\nno DEV improvement -> luck features rejected at tryouts, no TEST look")
    json.dump({"verdict": "dev_rejected", "dev_cv_base": round(base_cv, 5),
               "dev_cv": round(s, 5), "variant": name},
              open("data/nfl_luck.json", "w"), indent=1)
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
print(f"  luck coefs: {[round(float(c), 4) for c in c1.coef_[0][12:]]}")
json.dump({"variant": name, "dev_cv_base": round(base_cv, 5), "dev_cv": round(s, 5),
           "test_cur": round(float(llv(yt, p0).mean()), 5),
           "test_new": round(float(llv(yt, p1).mean()), 5),
           "delta": round(float(dd.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
           "adopted": bool(better)},
          open("data/nfl_luck.json", "w"), indent=1)
