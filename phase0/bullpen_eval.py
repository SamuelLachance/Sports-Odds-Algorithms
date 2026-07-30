"""Does BULLPEN STATE add over the 0.67509 model? Locked-split test.

The model rates the STARTER (FIP) but has no bullpen model at all -- the last
untested real gap. Three leak-safe, season-to-date team features built from the
relievers only (is_starter==0 rows of appearances.csv):

  bp_quality  season-to-date reliever FIP-core (13*HR+3*(BB+HBP)-2*SO)/IP,
              empirical-Bayes shrunk to the league bullpen core. Lower = better.
  bp_fat_3d   reliever outs thrown by the bullpen in the last 3 days (fatigue).
  bp_fat_1d   reliever outs thrown yesterday (back-to-back fatigue).

Home-minus-away differentials added to the DEV blend, scored once on TEST.
Paired bootstrap. (bp_quality overlaps team Elo -- a good pen already wins games;
the genuinely new part is game-specific fatigue.)
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict, deque
from datetime import datetime

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
PRIOR_OUTS = 200.0   # empirical-Bayes shrink strength (~67 IP) for bullpen FIP


def reliever_games():
    """(game_id, team) -> (fip_num, outs, bf), relievers only; plus a per-team,
    date-sorted list of (date, outs) reliever workloads for fatigue windows.
    Also returns the league reliever FIP-core for shrinkage."""
    per_game = defaultdict(lambda: [0.0, 0, 0])   # fip_num, outs, bf
    lg_num = 0.0
    lg_outs = 0
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        if r["is_starter"] == "1":
            continue
        outs = int(r["outs"])
        hr, bb, hbp, so = (int(r[k]) for k in ("HR", "BB", "HBP", "SO"))
        num = 13 * hr + 3 * (bb + hbp) - 2 * so
        a = per_game[(r["game_id"], r["team"])]
        a[0] += num
        a[1] += outs
        a[2] += int(r["bf"])
        lg_num += num
        lg_outs += outs
    lg_core = lg_num / lg_outs * 3.0 if lg_outs else 1.0
    return per_game, lg_core


def days(d1, d2):
    return (datetime.strptime(d1, "%Y%m%d") - datetime.strptime(d2, "%Y%m%d")).days


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
    per_game, lg_core = reliever_games()
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    cum = defaultdict(lambda: [0.0, 0])                 # (season,team) -> cum fip_num, outs
    work = defaultdict(lambda: deque(maxlen=40))        # (season,team) -> deque[(date, outs)]
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
            cum.clear(); work.clear()
        prev = g["season"]

        def bp_fip(team):
            c = cum[(g["season"], team)]
            return (c[0] + PRIOR_OUTS * lg_core / 3.0) / (c[1] + PRIOR_OUTS) * 3.0

        def fatigue(team, win):
            return sum(o for d, o in work[(g["season"], team)] if 0 < days(g["date"], d) <= win)

        q_d = bp_fip(g["away"]) - bp_fip(g["home"])     # +: home pen better (lower FIP)
        f3_d = fatigue(g["away"], 3) - fatigue(g["home"], 3)   # +: away more tired
        f1_d = fatigue(g["away"], 1) - fatigue(g["home"], 1)
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]], q_d, f3_d, f1_d))

        m.update(g)
        for team in (g["home"], g["away"]):
            num, outs, _bf = per_game.get((g["game_id"], team), (0.0, 0, 0))
            c = cum[(g["season"], team)]
            c[0] += num; c[1] += outs
            work[(g["season"], team)].append((g["date"], outs))

    arr = np.array(rows, dtype=float)
    seasons = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]; tse = arr[:, 3]
    q_d, f3_d, f1_d = arr[:, 4], arr[:, 5], arr[:, 6]
    dmask = np.isin(seasons, list(DEV)); tmask = np.isin(seasons, TEST)

    def z(v):
        return (v - v[dmask].mean()) / (v[dmask].std() or 1.0)
    tsz, qz, f3z, f1z = z(tse), z(q_d), z(f3_d), z(f1_d)

    yt = y[tmask]
    base = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask]]), y[dmask])
    p_base = base.predict_proba(np.column_stack([lf[tmask], tsz[tmask]]))[:, 1]
    print(f"games: {len(y):,}   league reliever FIP-core: {lg_core:.3f}")
    print(f"bp_quality diff sd (raw FIP pts): {q_d.std():.3f}")
    print(f"\n=== LOCKED TEST  n={len(yt):,}   base FIP-Elo+TrueSkill LL={ll(p_base, yt):.5f} ===")

    variants = {
        "quality only":        [qz],
        "fatigue only (3d+1d)": [f3z, f1z],
        "quality + fatigue":   [qz, f3z, f1z],
    }
    for name, feats in variants.items():
        Xd = np.column_stack([lf[dmask], tsz[dmask]] + [f[dmask] for f in feats])
        Xt = np.column_stack([lf[tmask], tsz[tmask]] + [f[tmask] for f in feats])
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(Xd, y[dmask])
        p_plus = mdl.predict_proba(Xt)[:, 1]
        d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
        coefs = [round(c, 4) for c in mdl.coef_[0][2:]]
        sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
        print(f"  + {name:22s} LL={ll(p_plus, yt):.5f}  d={d:+.5f} "
              f"CI[{lo:+.5f},{hi:+.5f}] {sig:5s} coefs={coefs}")


if __name__ == "__main__":
    main()
