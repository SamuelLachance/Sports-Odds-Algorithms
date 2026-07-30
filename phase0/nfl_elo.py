"""NFL rebuild, step 1: plain tuned Elo — the baseline every later model must beat.

Protocol (LOCKED before any results, same discipline as the MLB rebuild):
  DEV  1999-2015, scored from 2001 (two warmup seasons)  -> all tuning happens here
  TEST 2016-2025                                          -> scored exactly once
  ties y=0.5; playoffs included; neutral sites hfa=0; franchise continuity
  (STL->LA, SD->LAC, OAK->LV); market columns (spread/moneyline) NEVER read.

Data: nflverse games.csv (data/nfl_games.csv, community-maintained, 1999+).
"""
from __future__ import annotations

import csv
import json
import time
import numpy as np

DATA = "data/nfl_games.csv"
FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
DEV_YEARS = range(1999, 2016)
DEV_SCORE_FROM = 2001
TEST_YEARS = range(2016, 2026)


def load_games():
    games = []
    for r in csv.DictReader(open(DATA)):
        if r["home_score"] == "" or r["away_score"] == "":
            continue                                   # future/unplayed
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        games.append({
            "season": int(r["season"]), "date": r["gameday"],
            "home": FRANCHISE.get(r["home_team"], r["home_team"]),
            "away": FRANCHISE.get(r["away_team"], r["away_team"]),
            "y": 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5),
            "neutral": r["location"] == "Neutral",
            "type": r["game_type"],
        })
    games.sort(key=lambda g: (g["date"],))
    return games


def run_elo(games, *, k, hfa, regress, score_from, score_to=None):
    """Chronological walk; returns (probs, ys) for scored seasons."""
    R = {}
    ps, ys = [], []
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - regress)
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        h = 0.0 if g["neutral"] else hfa
        p = 1.0 / (1.0 + 10 ** (-((rh + h) - ra) / 400.0))
        if g["season"] >= score_from and (score_to is None or g["season"] <= score_to):
            ps.append(p); ys.append(g["y"])
        d = k * (g["y"] - p)
        R[g["home"]] += d
        R[g["away"]] -= d
    return np.array(ps), np.array(ys)


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    games = load_games()
    dev = [g for g in games if g["season"] in DEV_YEARS]
    n_dev_scored = sum(1 for g in dev if g["season"] >= DEV_SCORE_FROM)
    n_test = sum(1 for g in games if g["season"] in TEST_YEARS)
    print(f"games {len(games)} | DEV {len(dev)} (scored {n_dev_scored}) | TEST {n_test}")

    # naive baselines on DEV, for scale
    ydev = np.array([g["y"] for g in dev if g["season"] >= DEV_SCORE_FROM])
    hp = ydev.mean()
    print(f"DEV home rate {hp:.4f} | constant-0.5 LL 0.69315 | "
          f"home-prior LL {ll(np.full_like(ydev, hp), ydev):.5f}\n")

    rng = np.random.default_rng(20260723)
    N = 1200
    t0 = time.time()
    best = None
    for i in range(N):
        cand = dict(
            k=float(np.exp(rng.uniform(np.log(2.0), np.log(80.0)))),
            hfa=float(rng.uniform(0.0, 120.0)),
            regress=float(rng.uniform(0.0, 0.8)),
        )
        p, y = run_elo(dev, score_from=DEV_SCORE_FROM, **cand)
        s = ll(p, y)
        if best is None or s < best[0]:
            best = (s, cand)
        if (i + 1) % 400 == 0:
            print(f"  ...{i+1}/{N}  best DEV LL {best[0]:.5f}  ({time.time()-t0:.0f}s)")
    print(f"\nbest DEV LL {best[0]:.5f}  params "
          f"{ {a: round(b, 4) for a, b in best[1].items()} }")

    # ---- TEST: scored exactly once, full-history walk with DEV-chosen params ----
    p, y = run_elo(games, score_from=TEST_YEARS.start, score_to=TEST_YEARS.stop - 1, **best[1])
    test_ll = ll(p, y)
    hp_test = ll(np.full_like(y, hp), y)     # DEV home-prior applied to TEST
    print(f"\nTEST 2016-2025 (n={len(y)}, scored ONCE):")
    print(f"  tuned Elo        LL {test_ll:.5f}")
    print(f"  home-prior (DEV) LL {hp_test:.5f}")
    print(f"  constant-0.5     LL 0.69315")
    json.dump({"protocol": {"dev": "1999-2015 scored from 2001", "test": "2016-2025",
                            "ties": 0.5, "neutral_hfa": 0, "playoffs": "included"},
               "dev_ll": round(best[0], 5), "params": best[1],
               "test_n": int(len(y)), "test_ll": round(test_ll, 5),
               "home_prior_test_ll": round(hp_test, 5)},
              open("data/nfl_elo_base.json", "w"), indent=1)
    print("\nwrote data/nfl_elo_base.json")


if __name__ == "__main__":
    main()
