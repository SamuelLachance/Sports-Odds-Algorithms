"""Does BABIP add over the 0.67509 model? Locked-split test.

Two rolling, leak-safe, season-to-date team features from the parsed appearances:
  off_babip  = the team's hitters' BABIP (hits on balls in play)  [by batting team]
  def_babip  = BABIP allowed by the team's pitchers/defense = 1 - DER  [by fielding team]
Home-minus-away differentials added to the DEV blend, scored once on TEST.
Paired bootstrap. (def_babip is the complement of the DER feature already tested;
off_babip is the only genuinely new part.)
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
LG_BABIP = 0.30
PRIOR = 300.0


def team_game_babip():
    """(game_id, team) -> {'off':(BIP,hits), 'def':(BIP,hits)}.
    off = team as the batting side (rows where opp==team);
    def = team as the fielding side (rows where team==team)."""
    off = defaultdict(lambda: [0, 0])
    dfn = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        bf, so, bb, hbp, hr, h = (int(r[k]) for k in ("bf", "SO", "BB", "HBP", "HR", "H"))
        bip = bf - so - bb - hbp - hr
        hits_bip = h - hr
        if bip <= 0:
            continue
        dfn[(r["game_id"], r["team"])][0] += bip
        dfn[(r["game_id"], r["team"])][1] += max(hits_bip, 0)
        off[(r["game_id"], r["opp"])][0] += bip
        off[(r["game_id"], r["opp"])][1] += max(hits_bip, 0)
    return off, dfn


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
    off, dfn = team_game_babip()
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    ro = defaultdict(lambda: [0.0, 0.0])   # rolling offensive (season,team)
    rd = defaultdict(lambda: [0.0, 0.0])   # rolling defensive
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]

        def babip(store, team):
            c = store[(g["season"], team)]
            return (c[1] + PRIOR * LG_BABIP) / (c[0] + PRIOR)
        off_d = babip(ro, g["home"]) - babip(ro, g["away"])         # higher = home hits more
        def_d = babip(rd, g["away"]) - babip(rd, g["home"])         # home allows fewer = advantage
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]], off_d, def_d))
        m.update(g)
        for team in (g["home"], g["away"]):
            ob, oh = off.get((g["game_id"], team), (0, 0))
            db, dh = dfn.get((g["game_id"], team), (0, 0))
            ro[(g["season"], team)][0] += ob; ro[(g["season"], team)][1] += oh
            rd[(g["season"], team)][0] += db; rd[(g["season"], team)][1] += dh

    arr = np.array(rows, dtype=float)
    seasons = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]; tse = arr[:, 3]
    off_d, def_d = arr[:, 4], arr[:, 5]
    dmask = np.isin(seasons, list(DEV)); tmask = np.isin(seasons, TEST)

    def z(v):
        return (v - v[dmask].mean()) / (v[dmask].std() or 1.0)
    tsz, offz, defz = z(tse), z(off_d), z(def_d)

    base = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask]]), y[dmask])
    plus = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask], offz[dmask], defz[dmask]]), y[dmask])
    print(f"games: {len(y):,}")
    print(f"DEV coeffs: off_babip={plus.coef_[0][2]:.4f}  def_babip={plus.coef_[0][3]:.4f}")

    yt = y[tmask]
    p_base = base.predict_proba(np.column_stack([lf[tmask], tsz[tmask]]))[:, 1]
    p_plus = plus.predict_proba(np.column_stack([lf[tmask], tsz[tmask], offz[tmask], defz[tmask]]))[:, 1]
    print(f"\n=== LOCKED TEST  n={len(yt):,} ===")
    print(f"  FIP-Elo + TrueSkill      LL={ll(p_base, yt):.5f}   <- current model")
    print(f"  + BABIP (off + def)      LL={ll(p_plus, yt):.5f}")
    d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
    print(f"\ncurrent  minus  +BABIP: {d:+.5f} nats (positive = BABIP helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT - BABIP helps' if lo > 0 else ('SIGNIFICANT - it HURTS' if hi < 0 else 'NOT significant - no measurable gain')}")


if __name__ == "__main__":
    main()
