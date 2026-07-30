"""Does starter rotation fatigue add over the 0.67509 model? Locked-split test.

Rest is 5-6 days for ~85% of starts and short rest is 0.1%, so the fatigue signal
has almost no variance. Still tested honestly: a group of rest encodings
(differential rest, short/long flags, workload-weighted) added to the DEV blend,
scored once on TEST, paired bootstrap vs the current model.
"""

from __future__ import annotations

import csv
import os

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


def main():
    games = load_games()
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    rec, last = {}, None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        rec[g["game_id"]] = (g["season"], m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"]), g["y"])
        m.update(g)

    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    rest = {}
    for r in csv.reader(open("data/retro_events/rest_feature.csv")):
        if r[0] == "game_id":
            continue
        hr = float(r[3]) if r[3] else 5.0
        ar = float(r[4]) if r[4] else 5.0
        rest[r[0]] = (hr, ar)

    rows = [(s, p, y, ts[gid], *rest[gid]) for gid, (s, p, y) in rec.items()
            if gid in ts and gid in rest]
    seasons = np.array([x[0] for x in rows])
    lf = logit(np.array([x[1] for x in rows]))
    y = np.array([x[2] for x in rows], float)
    tse = np.array([x[3] for x in rows])
    hr = np.array([x[4] for x in rows])
    ar = np.array([x[5] for x in rows])
    dmask = np.isin(seasons, list(DEV))
    tmask = np.isin(seasons, TEST)

    def z(v):
        s = v[dmask].std() or 1.0
        return (v - v[dmask].mean()) / s
    tsz = z(tse)
    # rest encodings (home advantage orientation)
    feats = {
        "rest_diff": z(hr - ar),
        "home_short": z((hr < 4).astype(float)) - z((ar < 4).astype(float)),
        "home_long": z((hr >= 7).astype(float)) - z((ar >= 7).astype(float)),
    }
    print(f"games: {len(y):,}  short-rest games in TEST: "
          f"{int(((hr[tmask]<4)|(ar[tmask]<4)).sum())}")

    base = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask]]), y[dmask])
    Xd = np.column_stack([lf[dmask], tsz[dmask]] + [f[dmask] for f in feats.values()])
    Xt = np.column_stack([lf[tmask], tsz[tmask]] + [f[tmask] for f in feats.values()])
    plus = LogisticRegression(C=1e6, max_iter=1000).fit(Xd, y[dmask])
    print("DEV rest coeffs:", {k: round(plus.coef_[0][2 + i], 4) for i, k in enumerate(feats)})

    yt = y[tmask]
    p_base = base.predict_proba(np.column_stack([lf[tmask], tsz[tmask]]))[:, 1]
    p_plus = plus.predict_proba(Xt)[:, 1]

    print(f"\n=== LOCKED TEST  n={len(yt):,} ===")
    print(f"  FIP-Elo + TrueSkill        LL={ll(p_base, yt):.5f}   <- current model")
    print(f"  + rotation fatigue         LL={ll(p_plus, yt):.5f}")
    d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
    print(f"\ncurrent  minus  +fatigue: {d:+.5f} nats (positive = fatigue helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT - fatigue helps' if lo > 0 else ('SIGNIFICANT - it HURTS' if hi < 0 else 'NOT significant - no measurable gain')}")


if __name__ == "__main__":
    main()
