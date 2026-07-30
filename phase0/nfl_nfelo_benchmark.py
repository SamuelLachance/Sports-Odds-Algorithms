"""BENCHMARK ONLY (market data never enters our model): our market-blind model vs
nfelo — the most respected public NFL model, which is market-ANCHORED by design
(config: market_resist_factor/market_weight; MarketRegression in the update loop).

Compares on our main TEST years (2016-2025), joined by game: our model, nfelo's
open- and close-anchored probabilities, its 'base' elo dif (converted 10^(d/400) —
still market-tinted via resist, labeled as such), and the market itself.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # tuned F dict, cv_ll/llv, games  # noqa: S102

NF = r"C:/Users/Admin/AppData/Local/Temp/claude/C--Users-Admin-Projects-Sports-Odds-Algorithms/c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad/nfelo_games.csv"
nf = {}
with open(NF, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        gid = r.get("game_id") or ""
        if gid:
            nf[gid] = r

def f2(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

X_cur = X_of(F)
fitm = dev & (y != 0.5)
clf = LogisticRegression(C=1e6, max_iter=5000).fit(X_cur[fitm], y[fitm])
p_ours = clf.predict_proba(X_cur)[:, 1]

rows = []
for i, g in enumerate(games):
    if not (test[i]):
        continue
    r = nf.get(g["gid"])
    if r is None:
        continue
    po = f2(r.get("nfelo_home_probability_open"))
    pc = f2(r.get("nfelo_home_probability_close"))
    db = f2(r.get("nfelo_dif_base"))
    mo = f2(r.get("home_implied_win_probability_open"))
    mc = f2(r.get("home_implied_win_probability_close"))
    if po is None or pc is None:
        continue
    pb = 1.0 / (1.0 + 10 ** (-(db) / 400.0)) if db is not None else None
    rows.append((y[i], p_ours[i], po, pc, pb, mo, mc))

arr = np.array([[a, b, c, d, e if e is not None else np.nan,
                 f if f is not None else np.nan, g2 if g2 is not None else np.nan]
                for (a, b, c, d, e, f, g2) in rows])
yt, ours, po, pc, pb, mo, mc = arr.T
print(f"joined TEST games: {len(yt)}")

def LL(p, mask=None):
    m = ~np.isnan(p) if mask is None else (mask & ~np.isnan(p))
    pp = np.clip(p[m], 1e-12, 1 - 1e-12)
    return float(-(yt[m] * np.log(pp) + (1 - yt[m]) * np.log(1 - pp)).mean()), int(m.sum())

for name, p in [("OUR model (market-blind)", ours),
                ("nfelo base elo (10^(d/400); resist-tinted)", pb),
                ("nfelo open-anchored", po),
                ("nfelo close-anchored", pc),
                ("market implied open (raw)", mo),
                ("market implied close (raw)", mc)]:
    ll, n = LL(p)
    print(f"  {name:<44} LL {ll:.5f}  (n={n})")

# paired: ours vs nfelo close-anchored (their flagship)
def v(p):
    pp = np.clip(p, 1e-12, 1 - 1e-12)
    return -(yt * np.log(pp) + (1 - yt) * np.log(1 - pp))
d = v(pc) - v(ours)                      # >0 => ours better
rng = np.random.default_rng(7)
n = len(d)
bs = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\npaired OURS vs nfelo-close: delta {d.mean():+.5f}  CI [{lo:+.5f}, {hi:+.5f}]"
      f"  ({'OURS better' if d.mean() > 0 else 'nfelo better'}"
      f"{', SIG' if (lo > 0 or hi < 0) else ', n.s.'})")
d2 = v(po) - v(ours)
bs2 = d2[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
print(f"paired OURS vs nfelo-open:  delta {d2.mean():+.5f}  CI [{lo2:+.5f}, {hi2:+.5f}]"
      f"  ({'OURS better' if d2.mean() > 0 else 'nfelo better'}"
      f"{', SIG' if (lo2 > 0 or hi2 < 0) else ', n.s.'})")
print(f"corr(ours, nfelo_close) = {np.corrcoef(ours, pc)[0, 1]:.3f}")
