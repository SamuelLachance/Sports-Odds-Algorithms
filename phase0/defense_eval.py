"""Does team defensive efficiency (DER) add over the 0.67509 model? Locked test.

DER = share of balls in play the defense converts to outs — the standard
range/efficiency measure. Computed from the parsed appearances (per pitcher-game
outs/BF/H/HR/BB/HBP/SO aggregated to the fielding team):
    BIP      = BF - SO - BB - HBP - HR
    BIP_outs = BIP - (H - HR)
    DER      = BIP_outs / BIP
Rolling season-to-date, empirical-Bayes shrunk to the league DER, read as-of
before each game (leak-safe). Home-minus-away differential added to the DEV blend,
scored once on TEST. Paired bootstrap.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

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
LG_DER = 0.69
PRIOR = 300.0


def team_game_defense():
    """(game_id, team) -> (BIP, BIP_outs) for the fielding team."""
    agg = defaultdict(lambda: [0, 0])   # BIP, BIP_outs
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        bf, so, bb, hbp, hr, h = (int(r[k]) for k in ("bf", "SO", "BB", "HBP", "HR", "H"))
        bip = bf - so - bb - hbp - hr
        bip_outs = bip - (h - hr)
        if bip > 0:
            a = agg[(r["game_id"], r["team"])]
            a[0] += bip
            a[1] += max(bip_outs, 0)
    return agg


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
    defn = team_game_defense()
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    roll = defaultdict(lambda: [0.0, 0.0])   # (season,team) -> cum BIP, BIP_outs
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]

        def der(team):
            c = roll[(g["season"], team)]
            return (c[1] + PRIOR * LG_DER) / (c[0] + PRIOR)
        feat = der(g["home"]) - der(g["away"])
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]], feat))

        m.update(g)
        for team in (g["home"], g["away"]):
            bip, outs = defn.get((g["game_id"], team), (0, 0))
            c = roll[(g["season"], team)]
            c[0] += bip
            c[1] += outs

    arr = np.array(rows, dtype=float)
    seasons = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]; tse = arr[:, 3]; der_d = arr[:, 4]
    dmask = np.isin(seasons, list(DEV)); tmask = np.isin(seasons, TEST)

    def z(v):
        return (v - v[dmask].mean()) / (v[dmask].std() or 1.0)
    tsz, derz = z(tse), z(der_d)

    base = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask]]), y[dmask])
    plus = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask], derz[dmask]]), y[dmask])
    print(f"games: {len(y):,}  DER differential sd (raw): {der_d.std():.4f}")
    print(f"DEV coef(DER) = {plus.coef_[0][2]:.4f}")

    yt = y[tmask]
    p_base = base.predict_proba(np.column_stack([lf[tmask], tsz[tmask]]))[:, 1]
    p_plus = plus.predict_proba(np.column_stack([lf[tmask], tsz[tmask], derz[tmask]]))[:, 1]
    print(f"\n=== LOCKED TEST  n={len(yt):,} ===")
    print(f"  FIP-Elo + TrueSkill      LL={ll(p_base, yt):.5f}   <- current model")
    print(f"  + defensive efficiency   LL={ll(p_plus, yt):.5f}")
    d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
    print(f"\ncurrent  minus  +defense: {d:+.5f} nats (positive = defense helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT - defense helps' if lo > 0 else ('SIGNIFICANT - it HURTS' if hi < 0 else 'NOT significant - no measurable gain')}")


if __name__ == "__main__":
    main()
