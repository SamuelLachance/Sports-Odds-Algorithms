"""Does PLAYER HEALTH (rust / return-from-absence) add over the 0.67431 model?

The lineup says WHO plays, not how healthy. A hitter or starter returning from a
multi-day gap (IL stint or day-to-day injury) is likely below his rating for a
game or two. Leak-safe, as-of, within-season features from the parsed data:
  bat_rust   mean days-since-last-game over the starting lineup (capped)
  returnees  count of lineup hitters back from a >=10-day gap (fresh IL returns)
  sp_rust    the starter's days since his last start (capped)
Home-minus-away differentials added to the current full model
[logit_fip + ts_edge_z + bullpen_z + power_z] (=0.67431). Paired bootstrap.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

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
CAP = 20        # cap a gap in days (offseason / debut noise)
RET = 10        # a gap this long = a fresh return from injury


def days(d1, d2):
    return (datetime.strptime(d1, "%Y%m%d") - datetime.strptime(d2, "%Y%m%d")).days


def batters_by_game():
    g = defaultdict(set)
    for r in csv.reader(open("data/retro_events/pa.csv")):
        if r[0] != "game_id":
            g[r[0]].add(r[4])
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
    lu = FZ.load_lineups()
    bat_games = batters_by_game()
    pgbp, bplg = FZ.reliever_aggs()
    bpd = FZ.bullpen_diffs(games, pgbp, bplg)
    power, lg_iso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, lu, lg_iso)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    last_bat: dict[str, str] = {}      # batter -> last game date (within season)
    last_start: dict[str, str] = {}    # pitcher -> last start date
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
            last_bat.clear(); last_start.clear()
        prev = g["season"]
        d0 = g["date"]

        def bat_feats(side):
            bs = lu.get((g["game_id"], side), [])
            gaps = [min(days(d0, last_bat[b]), CAP) for b in bs if b in last_bat]
            if not gaps:
                return 0.0, 0.0
            return sum(gaps) / len(gaps), float(sum(1 for x in gaps if x >= RET))

        def sp_gap(sp):
            return min(days(d0, last_start[sp]), CAP) if sp in last_start else 0.0
        h_rust, h_ret = bat_feats(1)
        a_rust, a_ret = bat_feats(0)
        rust_d = h_rust - a_rust
        ret_d = h_ret - a_ret
        sp_d = sp_gap(g["home_sp"]) - sp_gap(g["away_sp"])

        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts and pdiff.get(g["game_id"]) is not None:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]], bpd[g["game_id"]],
                         pdiff[g["game_id"]], rust_d, ret_d, sp_d))

        m.update(g)
        for b in bat_games.get(g["game_id"], ()):
            last_bat[b] = d0
        for sp in (g["home_sp"], g["away_sp"]):
            last_start[sp] = d0

    arr = np.array(rows, dtype=float)
    s = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]
    tse, bpe, pwe = arr[:, 3], arr[:, 4], arr[:, 5]
    rust, ret, spg = arr[:, 6], arr[:, 7], arr[:, 8]
    dm = np.isin(s, list(DEV)); tm = np.isin(s, TEST)

    def z(v):
        return (v - v[dm].mean()) / (v[dm].std() or 1.0)
    tsz, bpz, pwz = z(tse), z(bpe), z(pwe)
    rustz, retz, spz = z(rust), z(ret), z(spg)

    yt = y[tm]
    base_cols = [lf, tsz, bpz, pwz]
    base = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(base_cols)[dm], y[dm])
    p_base = base.predict_proba(np.column_stack(base_cols)[tm])[:, 1]
    print(f"games: {len(y):,}   lineup rust sd={rust.std():.2f}d   returnees sd={ret.std():.2f}   sp-gap sd={spg.std():.2f}d")
    print(f"\n=== LOCKED TEST  n={len(yt):,}  base [fip+ts+bullpen+power] LL={ll(p_base, yt):.5f} (=0.67431) ===")

    for name, feats in [("lineup rust (days-since-game)", [rustz]),
                        ("fresh returnees (>=10d gap)", [retz]),
                        ("starter rust (days-since-start)", [spz]),
                        ("all health signals", [rustz, retz, spz])]:
        cols = base_cols + feats
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        p_plus = mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]
        d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
        sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
        coefs = [round(c, 4) for c in mdl.coef_[0][4:]]
        print(f"  + {name:33s} LL={ll(p_plus, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig:5s} coef={coefs}")


if __name__ == "__main__":
    main()
