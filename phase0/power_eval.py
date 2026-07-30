"""Does batter POWER/slugging add over the shipped 0.67492 model? Locked test.

The TrueSkill lineup feature rates ON-BASE ability (a walk and a homer are the same
"reached base"), so it is blind to power. This tests whether an explicit power term
helps. Per-batter career isolated-power rating, EB-shrunk (matches how the TrueSkill
lineup rating accumulates over all history), read AS-OF, averaged over the actual
starting lineup, home-minus-away. Tested three ways:
  lineup ISO   mean starter (TB-H)/PA differential   <- the real gap (power beyond OBP)
  lineup SLG   mean starter TB/PA differential        (includes singles, more OBP-like)
  team ISO     rolling season team (TB-H)/PA diff     (team level, expected ~Elo-redundant)
Increment over the CURRENT model [logit_fip, ts_edge_z, bp_z] (=0.67492). Bootstrap.
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
PRIOR_PA = 150.0     # empirical-Bayes shrink strength for a batter's power rate
TEAM_PRIOR_PA = 800.0


def load_power():
    """(game_id -> list[(batter, pa, h, tb)]) and league ISO/SLG per PA."""
    by_game = defaultdict(list)
    tot_pa = tot_h = tot_tb = 0
    with open("data/retro_events/power.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pa, h, tb = int(r["pa"]), int(r["h"]), int(r["tb"])
            by_game[r["game_id"]].append((r["batter"], pa, h, tb))
            tot_pa += pa; tot_h += h; tot_tb += tb
    lg_iso = (tot_tb - tot_h) / tot_pa
    lg_slg = tot_tb / tot_pa
    return by_game, lg_iso, lg_slg


def load_lineups():
    """(game_id, side) -> [batters]; Retrosheet side 0 = visitor/away, 1 = home."""
    lu = defaultdict(list)
    with open("data/retro_events/lineups.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lu[(r["game_id"], int(r["side"]))].append(r["batter"])
    return lu


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
    power, lg_iso, lg_slg = load_power()
    lu = load_lineups()
    per_game_bp, bp_lg = FZ.reliever_aggs()
    bpd = FZ.bullpen_diffs(games, per_game_bp, bp_lg)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    car = defaultdict(lambda: [0, 0, 0])            # batter -> career PA,H,TB (as-of)
    team = defaultdict(lambda: [0, 0, 0])           # (season,team) -> rolling PA,H,TB
    iso_p = PRIOR_PA * lg_iso
    slg_p = PRIOR_PA * lg_slg
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
            team.clear()
        prev = g["season"]

        def bat_iso(b):
            c = car[b]
            return (c[2] - c[1] + iso_p) / (c[0] + PRIOR_PA)     # (TB-H)/PA shrunk
        def bat_slg(b):
            c = car[b]
            return (c[2] + slg_p) / (c[0] + PRIOR_PA)
        def lineup(side, fn):
            bs = lu.get((g["game_id"], side), [])
            vals = [fn(b) for b in bs]
            return sum(vals) / len(vals) if vals else None
        h_iso, a_iso = lineup(1, bat_iso), lineup(0, bat_iso)
        h_slg, a_slg = lineup(1, bat_slg), lineup(0, bat_slg)

        def team_iso(t):
            c = team[(g["season"], t)]
            return (c[2] - c[1] + TEAM_PRIOR_PA * lg_iso) / (c[0] + TEAM_PRIOR_PA)
        tdiff = team_iso(g["home"]) - team_iso(g["away"])

        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts and h_iso is not None and a_iso is not None:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]], bpd[g["game_id"]],
                         h_iso - a_iso, h_slg - a_slg, tdiff))

        m.update(g)
        # advance career power (all batters) and team power (by batting side) with
        # THIS game's lines, AFTER reading the as-of rating above (leak-safe).
        home_set = set(lu.get((g["game_id"], 1), []))
        away_set = set(lu.get((g["game_id"], 0), []))
        for b, pa, h, tb in power.get(g["game_id"], []):
            c = car[b]; c[0] += pa; c[1] += h; c[2] += tb
            tcode = g["home"] if b in home_set else (g["away"] if b in away_set else None)
            if tcode is not None:
                t = team[(g["season"], tcode)]; t[0] += pa; t[1] += h; t[2] += tb

    arr = np.array(rows, dtype=float)
    seasons = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]
    tse, bpe = arr[:, 3], arr[:, 4]
    iso_d, slg_d, tiso_d = arr[:, 5], arr[:, 6], arr[:, 7]
    dm = np.isin(seasons, list(DEV)); tm = np.isin(seasons, TEST)

    def z(v):
        return (v - v[dm].mean()) / (v[dm].std() or 1.0)
    tsz, bpz = z(tse), z(bpe)
    isoz, slgz, tisoz = z(iso_d), z(slg_d), z(tiso_d)

    yt = y[tm]
    base_cols = [lf, tsz, bpz]
    base = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(base_cols)[dm], y[dm])
    p_base = base.predict_proba(np.column_stack(base_cols)[tm])[:, 1]
    print(f"games: {len(y):,}  lg ISO/PA={lg_iso:.3f} SLG/PA={lg_slg:.3f}")
    print(f"\n=== LOCKED TEST  n={len(yt):,}   base [fip+ts+bullpen] LL={ll(p_base, yt):.5f} (=shipped 0.67492) ===")

    for name, feat in [("lineup ISO (power beyond OBP)", isoz),
                       ("lineup SLG (incl. singles)", slgz),
                       ("team-level ISO", tisoz)]:
        cols = base_cols + [feat]
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        p_plus = mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]
        d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
        sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
        print(f"  + {name:32s} LL={ll(p_plus, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig:5s} coef={mdl.coef_[0][-1]:+.4f}")


if __name__ == "__main__":
    main()
