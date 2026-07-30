"""Does pitcher DEPTH / times-through-the-order add over the 0.67431 model?

Two leak-safe, as-of starter features from the parsed data:
  depth   rolling outs-per-start EWMA (how deep the starter goes -> bullpen exposure)
  tto     the starter's 3rd-time-through minus 1st-time-through on-base rate
          allowed (the "third time through the order penalty"): each start's PAs
          are bucketed by the starter's cumulative batters faced (1-9=1st, 10-18=2nd,
          19+=3rd), accumulated per pitcher, EB-shrunk.
Home-minus-away differentials added to the current full model
[logit_fip + ts_edge_z + bullpen_z + power_z] (=0.67431). Paired bootstrap.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402
import freeze_trueskill as FZ  # noqa: E402

EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]
DECAY = 0.95            # per-start EWMA for depth
LG_OB = 0.33           # league on-base rate (per-PA) prior for TTO buckets
TTO_PRIOR = 60.0       # EB shrink strength (PAs) per TTO bucket


def starter_outs():
    """(game_id, pitcher) -> outs, for starters only."""
    out = {}
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        if r["is_starter"] == "1":
            out[(r["game_id"], r["pitcher"])] = int(r["outs"])
    return out


def pa_by_game():
    """game_id -> [(pitcher, on_base), ...] in order."""
    g = defaultdict(list)
    for r in csv.reader(open("data/retro_events/pa.csv")):
        if r[0] == "game_id":
            continue
        g[r[0]].append((r[5], int(r[6])))
    return g


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
    so = starter_outs()
    pas = pa_by_game()
    pgbp, bplg = FZ.reliever_aggs()
    bpd = FZ.bullpen_diffs(games, pgbp, bplg)
    power, lg_iso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, FZ.load_lineups(), lg_iso)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    dep_o = defaultdict(float)   # pitcher -> decayed outs sum
    dep_n = defaultdict(float)   # pitcher -> decayed start count
    tto_ob = defaultdict(lambda: [0.0, 0.0, 0.0])   # pitcher -> onbase per bucket
    tto_n = defaultdict(lambda: [0.0, 0.0, 0.0])    # pitcher -> PA count per bucket
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]

        def depth(pid):
            return dep_o[pid] / dep_n[pid] if dep_n[pid] > 0 else 15.0

        def tto_pen(pid):
            ob, n = tto_ob[pid], tto_n[pid]
            r1 = (ob[0] + TTO_PRIOR * LG_OB) / (n[0] + TTO_PRIOR)
            r3 = (ob[2] + TTO_PRIOR * LG_OB) / (n[2] + TTO_PRIOR)
            return r3 - r1
        dep_d = depth(g["home_sp"]) - depth(g["away_sp"])
        tto_d = tto_pen(g["home_sp"]) - tto_pen(g["away_sp"])
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts and pdiff.get(g["game_id"]) is not None:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]], bpd[g["game_id"]],
                         pdiff[g["game_id"]], dep_d, tto_d))

        m.update(g)
        # update depth (starter outs this game)
        for sp in (g["home_sp"], g["away_sp"]):
            o = so.get((g["game_id"], sp))
            if o is not None:
                dep_o[sp] = DECAY * dep_o[sp] + o
                dep_n[sp] = DECAY * dep_n[sp] + 1
        # update TTO buckets from this game's PAs (starters only)
        starters = {g["home_sp"], g["away_sp"]}
        bf = defaultdict(int)
        for pid, ob in pas.get(g["game_id"], []):
            bf[pid] += 1
            if pid in starters:
                bkt = min((bf[pid] - 1) // 9, 2)
                tto_ob[pid][bkt] += ob
                tto_n[pid][bkt] += 1

    arr = np.array(rows, dtype=float)
    seasons = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]
    tse, bpe, pwe, depe, ttoe = arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6], arr[:, 7]
    dm = np.isin(seasons, list(DEV)); tm = np.isin(seasons, TEST)

    def z(v):
        return (v - v[dm].mean()) / (v[dm].std() or 1.0)
    tsz, bpz, pwz, depz, ttoz = z(tse), z(bpe), z(pwe), z(depe), z(ttoe)

    yt = y[tm]
    base_cols = [lf, tsz, bpz, pwz]
    base = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(base_cols)[dm], y[dm])
    p_base = base.predict_proba(np.column_stack(base_cols)[tm])[:, 1]
    print(f"games: {len(y):,}  depth diff sd(outs)={depe.std():.2f}  tto diff sd={ttoe.std():.4f}")
    print(f"\n=== LOCKED TEST  n={len(yt):,}  base [fip+ts+bullpen+power] LL={ll(p_base, yt):.5f} (=0.67431) ===")

    for name, feats in [("pitcher depth (outs/start)", [depz]),
                        ("TTO penalty (3rd-1st on-base)", [ttoz]),
                        ("depth + TTO", [depz, ttoz])]:
        cols = base_cols + feats
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        p_plus = mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]
        d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
        sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
        coefs = [round(c, 4) for c in mdl.coef_[0][4:]]
        print(f"  + {name:32s} LL={ll(p_plus, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig:5s} coef={coefs}")


if __name__ == "__main__":
    main()
