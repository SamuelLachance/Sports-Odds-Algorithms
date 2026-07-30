"""Does a wOBA lineup rating add to, or replace, on-base + power? (over 0.67368)

wOBA weights every offensive event by run value (BB .69, HBP .72, 1B .89, 2B 1.27,
3B 1.62, HR 2.10). We build a per-batter career wOBA-per-PA rating (EB-shrunk),
average it over the actual lineup, home-minus-away, and:
  A) add it over the full model [logit_xfip + ts_edge_z + bp_z + power_z] (=0.67368)
  B) replace ts + power with wOBA alone: [logit_xfip + bp_z + woba_z]
The baseline uses the xFIP core (load_games returns xFIP lines). Paired bootstrap.
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
PRIOR_PA = 150.0
W = dict(bb=0.69, hbp=0.72, s1=0.89, s2=1.27, s3=1.62, hr=2.10)


def load_woba():
    by_game = defaultdict(list)
    tot_num = tot_pa = 0.0
    with open("data/retro_events/woba.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pa = int(r["pa"])
            num = (W["bb"] * int(r["bb"]) + W["hbp"] * int(r["hbp"]) + W["s1"] * int(r["s1"])
                   + W["s2"] * int(r["s2"]) + W["s3"] * int(r["s3"]) + W["hr"] * int(r["hr"]))
            by_game[r["game_id"]].append((r["batter"], pa, num))
            tot_num += num; tot_pa += pa
    return by_game, tot_num / tot_pa


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
    games = load_games()          # xFIP lines
    woba, lg_woba = load_woba()
    lu = FZ.load_lineups()
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lg_iso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, lu, lg_iso)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    car = defaultdict(lambda: [0.0, 0.0])   # batter -> career [num, pa]
    prior = PRIOR_PA * lg_woba
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        def bw(b):
            c = car[b]
            return (c[0] + prior) / (c[1] + PRIOR_PA)

        def lineup(side):
            bs = lu.get((g["game_id"], side), [])
            v = [bw(b) for b in bs]
            return sum(v) / len(v) if v else None
        h, a = lineup(1), lineup(0)
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts and pdiff.get(g["game_id"]) is not None and h is not None and a is not None:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]], bpd[g["game_id"]],
                         pdiff[g["game_id"]], h - a))
        m.update(g)
        for b, pa, num in woba.get(g["game_id"], []):
            car[b][0] += num; car[b][1] += pa

    arr = np.array(rows, dtype=float)
    s = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]
    tse, bpe, pwe, wbe = arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
    dm = np.isin(s, list(DEV)); tm = np.isin(s, TEST)

    def z(v):
        return (v - v[dm].mean()) / (v[dm].std() or 1.0)
    tsz, bpz, pwz, wbz = z(tse), z(bpe), z(pwe), z(wbe)
    yt = y[tm]

    def fit_score(cols):
        mdl = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols)[dm], y[dm])
        return mdl.predict_proba(np.column_stack(cols)[tm])[:, 1]

    p_full = fit_score([lf, tsz, bpz, pwz])
    print(f"games: {len(y):,}  league wOBA={lg_woba:.3f}")
    print(f"\n=== LOCKED TEST  n={len(yt):,}  full [xfip+ts+bp+power] LL={ll(p_full, yt):.5f} (=0.67368) ===")

    # A) increment
    p_a = fit_score([lf, tsz, bpz, pwz, wbz])
    d, lo, hi = boot(ll_vec(p_full, yt), ll_vec(p_a, yt))
    sig = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "n.s.")
    print(f"  A) + wOBA (increment)          LL={ll(p_a, yt):.5f}  d={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig}")

    # B) replacement: wOBA instead of ts + power
    p_b = fit_score([lf, bpz, wbz])
    d2, lo2, hi2 = boot(ll_vec(p_full, yt), ll_vec(p_b, yt))
    sig2 = "wOBA-only BETTER" if lo2 > 0 else ("ts+power BETTER" if hi2 < 0 else "tie (n.s.)")
    print(f"  B) [xfip+bp+wOBA] (replaces ts+power) LL={ll(p_b, yt):.5f}  full-minus-this d={d2:+.5f} CI[{lo2:+.5f},{hi2:+.5f}] {sig2}")


if __name__ == "__main__":
    main()
