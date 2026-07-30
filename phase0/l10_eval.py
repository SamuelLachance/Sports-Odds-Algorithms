"""Does recent form (L10) add over the 0.67509 model? Locked-split test.

Team Elo already encodes recent form dynamically (it updates every game and
weights by opponent), so L10 is expected to be largely redundant. Tested anyway:
a group of form encodings (L10, L20, current streak differentials), leak-safe and
within-season, added to the DEV blend and scored once on TEST. Paired bootstrap.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict, deque

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import sys  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

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


def streak(res: deque) -> float:
    """Signed current streak from most-recent results (1=W,0=L)."""
    if not res:
        return 0.0
    last = res[-1]
    s = 0
    for r in reversed(res):
        if r == last:
            s += 1
        else:
            break
    return s if last == 1 else -s


def main():
    games = load_games()
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    hist = defaultdict(lambda: deque(maxlen=20))    # (season,team) -> recent W/L
    rows = []
    last = None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        hk = (g["season"], g["home"])
        ak = (g["season"], g["away"])
        hh, ah = hist[hk], hist[ak]

        def w(dq, n):
            s = list(dq)[-n:]
            return sum(s) / len(s) if s else 0.5
        l10 = w(hh, 10) - w(ah, 10)
        l20 = w(hh, 20) - w(ah, 20)
        strk = (streak(hh) - streak(ah)) / 10.0
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]], l10, l20, strk))

        m.update(g)
        hh.append(int(g["y"] == 1.0))
        ah.append(int(g["y"] == 0.0))

    seasons = np.array([r[0] for r in rows])
    lf = logit(np.array([r[1] for r in rows]))
    y = np.array([r[2] for r in rows], float)
    tse = np.array([r[3] for r in rows])
    l10 = np.array([r[4] for r in rows])
    l20 = np.array([r[5] for r in rows])
    strk = np.array([r[6] for r in rows])
    dmask = np.isin(seasons, list(DEV))
    tmask = np.isin(seasons, TEST)

    def z(v):
        return (v - v[dmask].mean()) / (v[dmask].std() or 1.0)
    tsz, l10z, l20z, sz = z(tse), z(l10), z(l20), z(strk)

    base = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask]]), y[dmask])
    Xd = np.column_stack([lf[dmask], tsz[dmask], l10z[dmask], l20z[dmask], sz[dmask]])
    Xt = np.column_stack([lf[tmask], tsz[tmask], l10z[tmask], l20z[tmask], sz[tmask]])
    plus = LogisticRegression(C=1e6, max_iter=1000).fit(Xd, y[dmask])
    print(f"games: {len(y):,}")
    print("DEV form coeffs:", {"L10": round(plus.coef_[0][2], 4),
                               "L20": round(plus.coef_[0][3], 4),
                               "streak": round(plus.coef_[0][4], 4)})

    yt = y[tmask]
    p_base = base.predict_proba(np.column_stack([lf[tmask], tsz[tmask]]))[:, 1]
    p_plus = plus.predict_proba(Xt)[:, 1]
    print(f"\n=== LOCKED TEST  n={len(yt):,} ===")
    print(f"  FIP-Elo + TrueSkill        LL={ll(p_base, yt):.5f}   <- current model")
    print(f"  + recent form (L10/L20/streak)  LL={ll(p_plus, yt):.5f}")
    d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
    print(f"\ncurrent  minus  +form: {d:+.5f} nats (positive = form helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT - form helps' if lo > 0 else ('SIGNIFICANT - it HURTS' if hi < 0 else 'NOT significant - no measurable gain')}")


if __name__ == "__main__":
    main()
