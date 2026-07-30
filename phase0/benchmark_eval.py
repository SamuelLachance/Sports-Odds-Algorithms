"""Score the SHIPPED full model on recent windows for a market benchmark.

Applies the FROZEN blend (mlbwp/artifacts/blend.json, full tier incl. baserunning)
to every game with its actual lineup -- the model's real pre-game number -- and
reports log loss on windows chosen to line up with the DRatings "Season Prediction
Results" sample (Sportsbooks consensus 0.67379, DRatings 0.67680 on 14,878 games).

This is APPLES-TO-APPLES on the games (a comparable recent out-of-sample window,
all seasons >= 2016 so none touch the DEV 2000-2015 fit), NOT the locked-test
number (2016-2024 ex-2020, n=19,435). Market-free: our model never sees a line.
"""

from __future__ import annotations

import csv
import json
import os
import sys

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.siera import SieraRater  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402
import freeze_trueskill as FZ  # noqa: E402

EPS = 1e-15


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def boot_ci(v, n=10000, seed=7):
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    games = load_games()
    lu = FZ.load_lineups()
    bpd = FZ.bullpen_diffs(games, *FZ.reliever_aggs())
    power, lgiso = FZ.load_power()
    pdiff, _ = FZ.power_diffs(games, power, lu, lgiso)
    brdiff, _ = FZ.baserun_diffs(games, FZ.load_baserun(), lu)
    tsf = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}
    ss, bbl = FZ.load_siera_inputs()
    slg = FZ.league_siera_rates(games, ss, bbl)
    B = json.load(open("mlbwp/artifacts/blend.json"))

    sr = SieraRater(PARAMS["decay"], 120.0, slg)
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    rows = []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            m.new_season()
        prev = g["season"]
        gid = g["game_id"]
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        sd = sr.edge(g["home_sp"], g["away_sp"])
        rows.append((g["season"], p, g["y"], tsf.get(gid), bpd.get(gid),
                     pdiff.get(gid), sd, brdiff.get(gid)))
        m.update(g)
        for pid in (g["home_sp"], g["away_sp"]):
            so, bb, bf = ss.get((gid, pid), (0, 0, 0))
            gb, fb, pu = bbl.get((gid, pid), (0, 0, 0))
            sr.update(pid, so, bb, gb, fb, pu, bf)

    fb, rb = B["full"], B["recbp"]

    def z(v, mu, sd):
        return (v - B[mu]) / B[sd]

    def prob(r):
        _, p, _, ts, bp, pw, si, br = r
        lf = logit(np.array([p]))[0]
        bpz = z(bp, "bp_mu", "bp_sd") if bp is not None else 0.0
        siz = z(si, "si_mu", "si_sd") if si is not None else 0.0
        if ts is None or pw is None or br is None:      # no lineup -> recbp tier
            return sig(rb["b0"] + rb["b1"] * lf + rb["b3"] * bpz + rb.get("b5", 0.0) * siz)
        return sig(fb["b0"] + fb["b1"] * lf + fb["b2"] * z(ts, "ts_mu", "ts_sd")
                   + fb["b3"] * bpz + fb["b4"] * z(pw, "pw_mu", "pw_sd")
                   + fb["b5"] * siz + fb["b6"] * z(br, "br_mu", "br_sd"))

    P = np.array([prob(r) for r in rows])
    Y = np.array([r[2] for r in rows], float)
    S = np.array([r[0] for r in rows])

    print("SHIPPED full model (xFIP-Elo + on-base + power + SIERA + bullpen + baserunning),")
    print("actual lineups, applied with the frozen DEV-fit blend. Market-free.\n")
    print(f"{'window':30s} {'n':>7s} {'log_loss':>9s}   vs Sportsbooks 0.67379 / DRatings 0.67680")
    windows = [
        ("2019-2025", list(range(2019, 2026))),
        ("2019-2024 (ex-2020)", [y for y in range(2019, 2025) if y != 2020]),
        ("2021-2025", list(range(2021, 2026))),
        ("locked test 2016-24 ex-20", [y for y in range(2016, 2025) if y != 2020]),
        ("2015-2025 (all modern)", list(range(2015, 2026))),
    ]
    for label, seas in windows:
        mask = np.isin(S, seas)
        v = ll_vec(P[mask], Y[mask])
        n = int(mask.sum())
        lo, hi = boot_ci(v)
        edge = 0.67379 - v.mean()
        tag = "EDGES market" if edge > 0 else "below market"
        print(f"{label:30s} {n:>7,d} {v.mean():>9.5f}   CI[{lo:.5f},{hi:.5f}]  {tag} by {edge:+.5f}")


if __name__ == "__main__":
    main()
