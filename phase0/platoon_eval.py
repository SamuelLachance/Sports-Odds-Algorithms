"""Does the platoon feature add anything over the 0.67509 model (FIP-Elo + TrueSkill)?

Same locked split. Baseline = the current full model (logit_fip + ts_edge_z),
recalibrated on DEV. Contender adds platoon_z. Only difference is the platoon
feature. Paired bootstrap for significance.
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


def col(path, idx):
    return {r[0]: float(r[idx]) for r in csv.reader(open(path)) if r[0] != "game_id"}


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

    ts = col("data/retro_events/ts_feature.csv", 3)
    pl = col("data/retro_events/platoon_feature.csv", 3)
    j = [(s, p, y, ts[gid], pl[gid]) for gid, (s, p, y) in rec.items() if gid in ts and gid in pl]
    print(f"games with all three: {len(j):,}")

    seasons = np.array([x[0] for x in j])
    lf = logit(np.array([x[1] for x in j]))
    y = np.array([x[2] for x in j], float)
    tse = np.array([x[3] for x in j])
    ple = np.array([x[4] for x in j])
    dmask = np.isin(seasons, list(DEV))
    tmask = np.isin(seasons, TEST)

    def z(v):
        return (v - v[dmask].mean()) / v[dmask].std()
    tsz, plz = z(tse), z(ple)

    base = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask]]), y[dmask])
    plus = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask], plz[dmask]]), y[dmask])
    print(f"DEV coef(platoon) = {plus.coef_[0][2]:.4f}")

    yt = y[tmask]
    p_base = base.predict_proba(np.column_stack([lf[tmask], tsz[tmask]]))[:, 1]
    p_plus = plus.predict_proba(np.column_stack([lf[tmask], tsz[tmask], plz[tmask]]))[:, 1]

    print(f"\n=== LOCKED TEST  n={len(yt):,} ===")
    print(f"  FIP-Elo + TrueSkill        LL={ll(p_base, yt):.5f}   <- current model (0.67509)")
    print(f"  + platoon (handedness)     LL={ll(p_plus, yt):.5f}")
    d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
    print(f"\ncurrent  minus  +platoon: {d:+.5f} nats (positive = platoon helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT - platoon helps' if lo > 0 else ('SIGNIFICANT - it HURTS' if hi < 0 else 'NOT significant - no measurable gain')}")


if __name__ == "__main__":
    main()
