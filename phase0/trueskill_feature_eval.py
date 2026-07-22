"""Does the per-PA TrueSkill lineup feature add anything over the FIP-Elo model?

Honest incremental test, same locked split:
  DEV  2000-2015 (from 2002)  -> fit the blend
  TEST 2016-2024 ex 2020      -> score once

Baseline is the working FIP-Elo model's probability, RECALIBRATED through a
logistic on DEV. The contender adds one feature, ts_edge, to that same logistic.
So the only difference between the two is the TrueSkill feature — a clean read on
its incremental value. Paired bootstrap for significance.

The TrueSkill ratings are leak-safe (as-of before each game). Lineup COMPOSITION
uses the actual lineups from the event files, which is mildly optimistic vs the
projected lineups you'd have pre-game — so this is a ceiling test: if it does not
help with actual lineups, it will not help with projected ones.
"""

from __future__ import annotations

import csv
import os

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, ".")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402

EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p, y):
    return float(ll_vec(p, y).mean())


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    # 1. FIP-Elo per-game probabilities (walk all games, record with game_id)
    games = load_games()
    lg = league_fip_core(games)
    m = FipPitcherElo(lg_fip=lg, **PARAMS)
    rec = {}
    last = None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        rec[g["game_id"]] = (g["season"], p, g["y"])
        m.update(g)
    print(f"FIP-Elo scored {len(rec):,} games")

    # 2. join the TrueSkill feature
    ts = {}
    for r in csv.reader(open("data/retro_events/ts_feature.csv")):
        if r[0] == "game_id":
            continue
        ts[r[0]] = float(r[3])
    joined = [(s, p, y, ts[gid]) for gid, (s, p, y) in rec.items() if gid in ts]
    print(f"games with both FIP-Elo and TrueSkill feature: {len(joined):,}")

    seasons = np.array([r[0] for r in joined])
    pf = np.array([r[1] for r in joined])
    y = np.array([r[2] for r in joined], float)
    tse = np.array([r[3] for r in joined])
    lf = logit(pf)
    # standardise ts_edge on DEV only
    dmask = np.isin(seasons, list(DEV))
    tmask = np.isin(seasons, TEST)
    mu, sd = tse[dmask].mean(), tse[dmask].std()
    tsz = (tse - mu) / sd

    # 3. fit both models on DEV
    Xb = lf[dmask].reshape(-1, 1)                       # baseline: fip only
    Xf = np.column_stack([lf[dmask], tsz[dmask]])       # fip + trueskill
    yb = y[dmask]
    base = LogisticRegression(C=1e6, max_iter=1000).fit(Xb, yb)
    full = LogisticRegression(C=1e6, max_iter=1000).fit(Xf, yb)
    print(f"\nDEV fit: baseline coef(logit_fip)={base.coef_[0][0]:.3f}")
    print(f"DEV fit: full coef(logit_fip)={full.coef_[0][0]:.3f}  coef(ts_edge)={full.coef_[0][1]:.4f}")

    # 4. score TEST
    Xbt = lf[tmask].reshape(-1, 1)
    Xft = np.column_stack([lf[tmask], tsz[tmask]])
    yt = y[tmask]
    p_base = base.predict_proba(Xbt)[:, 1]
    p_full = full.predict_proba(Xft)[:, 1]
    p_raw = pf[tmask]

    print(f"\n=== LOCKED TEST  n={len(yt):,}  home win rate={yt.mean():.4f} ===")
    print(f"  FIP-Elo (raw prob)            LL={ll(p_raw, yt):.5f}")
    print(f"  FIP-Elo (recalibrated)        LL={ll(p_base, yt):.5f}   <- baseline")
    print(f"  FIP-Elo + TrueSkill feature   LL={ll(p_full, yt):.5f}")

    d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_full, yt))
    print(f"\nrecalibrated FIP-Elo  minus  +TrueSkill: {d:+.5f} nats (positive = TrueSkill helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT — the TrueSkill lineup feature improves the model' if lo > 0 else ('SIGNIFICANT — it HURTS' if hi < 0 else 'NOT significant — no measurable gain over the working model')}")


if __name__ == "__main__":
    main()
