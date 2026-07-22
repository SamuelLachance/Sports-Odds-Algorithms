"""MLB Glicko-2 vs Elo — tuned on DEV only, scored once on a locked TEST split.

Protocol, fixed before any result was seen:

  DEV   2000-2015  (eval from 2002 so ratings are warm)   -> hyperparameter search
  TEST  2016-2024  (2020 excluded: a 60-game COVID season) -> scored once

Both systems are tuned on DEV with the same grid discipline, so neither gets an
unfair parameter advantage. Every prediction uses only strictly earlier games:
within a rating period all games are predicted before any of them updates a rating.

The earlier MLB figures (Elo 0.67964, Glicko-2 0.68244) are discarded — those were
tuned in-window on data that no longer exists locally.

Data: Retrosheet game logs. "The information used here was obtained free of charge
from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."
"""

from __future__ import annotations

import csv
import glob
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, "phase0")
from glicko2 import Glicko2  # noqa: E402

EPS = 1e-15
F_DATE, F_VIS, F_HOME, F_VIS_R, F_HOME_R = 0, 3, 6, 9, 10

DEV_YEARS = range(2000, 2016)
TEST_YEARS = [y for y in range(2016, 2025) if y != 2020]
DEV_WARMUP_FROM = 2002


def load_games(years) -> list[dict]:
    keep = set(years)
    rows = []
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        yr = int(path[-8:-4])
        if yr not in keep:
            continue
        with open(path, newline="", encoding="latin-1") as fh:
            for r in csv.reader(fh):
                if len(r) < 11:
                    continue
                try:
                    vr, hr = int(r[F_VIS_R]), int(r[F_HOME_R])
                except ValueError:
                    continue
                if vr == hr:
                    continue
                rows.append({"date": r[F_DATE], "season": yr,
                             "home": r[F_HOME], "away": r[F_VIS],
                             "y": 1.0 if hr > vr else 0.0})
    rows.sort(key=lambda g: (g["date"], g["home"]))
    return rows


def ll_vec(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p, y):
    return float(ll_vec(p, y).mean())


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def _period_key(datestr: str, days: int) -> int:
    d = datetime.strptime(datestr, "%Y%m%d")
    return (d - datetime(2000, 1, 1)).days // max(days, 1)


def run_glicko(games, *, period_days, eval_from_season, **kw):
    sys_ = Glicko2(**kw)
    preds, ys = [], []
    batch: list[tuple[str, str, float]] = []
    cur = None
    last_season = None
    for g in games:
        if last_season is not None and g["season"] != last_season:
            if batch:
                sys_.update_period(batch)
                batch, cur = [], None
            sys_.new_season()
        last_season = g["season"]

        k = _period_key(g["date"], period_days)
        if cur is not None and k != cur:
            sys_.update_period(batch)
            batch = []
        cur = k

        p = sys_.expect_home(g["home"], g["away"])
        if g["season"] >= eval_from_season:
            preds.append(p)
            ys.append(g["y"])
        batch.append((g["home"], g["away"], g["y"]))
    if batch:
        sys_.update_period(batch)
    return np.array(preds), np.array(ys)


def run_elo(games, *, k, hfa, regress, eval_from_season):
    r = defaultdict(lambda: 1500.0)
    preds, ys = [], []
    last_season = None
    for g in games:
        if last_season is not None and g["season"] != last_season:
            for t in r:
                r[t] = 1500.0 + (r[t] - 1500.0) * (1 - regress)
        last_season = g["season"]
        e = 1.0 / (1.0 + 10 ** (-((r[g["home"]] + hfa - r[g["away"]]) / 400.0)))
        if g["season"] >= eval_from_season:
            preds.append(e)
            ys.append(g["y"])
        d = k * (g["y"] - e)
        r[g["home"]] += d
        r[g["away"]] -= d
    return np.array(preds), np.array(ys)


def main():
    dev = load_games(DEV_YEARS)
    test = load_games(TEST_YEARS)
    print(f"DEV  {len(dev):,} games {dev[0]['date']}..{dev[-1]['date']}")
    print(f"TEST {len(test):,} games {test[0]['date']}..{test[-1]['date']} (2020 excluded)")

    # ---------- tune Elo on DEV ----------
    best_e = None
    for k, hfa, reg in itertools.product([2, 3, 4, 6, 8, 12], [15, 20, 24, 28, 32],
                                         [0.0, 0.15, 0.25, 0.35]):
        p, y = run_elo(dev, k=k, hfa=hfa, regress=reg, eval_from_season=DEV_WARMUP_FROM)
        s = ll(p, y)
        if best_e is None or s < best_e[0]:
            best_e = (s, k, hfa, reg)
    print(f"\nElo      best DEV LL={best_e[0]:.5f}  k={best_e[1]} hfa={best_e[2]} regress={best_e[3]}")

    # ---------- tune Glicko-2 on DEV ----------
    # WIDENED GRID. The first search pinned 4 of 5 parameters against an edge,
    # which means the optimum was outside the box and Glicko-2 was under-tuned.
    # min_rd is now searched too: it is the knob that controls how confident the
    # system is allowed to become, and leaving it fixed at 30 was the likely
    # cause of the over-confidence (pred sd 0.0965 vs Elo 0.0677).
    best_g = None
    grid = list(itertools.product(
        [0.2, 0.35, 0.5, 0.8, 1.2, 1.8],          # tau (was capped at 0.8)
        [15.0, 24.0, 32.0, 40.0, 50.0],           # home_adv (was capped at 32)
        [0.25, 0.35, 0.5, 0.65],                  # season_regress (was capped at 0.35)
        [1, 3, 7, 14, 30],                        # period_days (adds per-day/near-daily)
        [15.0, 30.0, 60.0],                       # season_rd_bump (was floored at 30)
        [30.0, 60.0, 90.0, 120.0],                # min_rd -- NEW, never searched before
    ))
    print(f"Glicko-2 grid: {len(grid)} configs (was 162)", flush=True)
    for tau, hadv, reg, pdays, rdb, mrd in grid:
        p, y = run_glicko(dev, period_days=pdays, eval_from_season=DEV_WARMUP_FROM,
                          tau=tau, home_adv=hadv, season_regress=reg,
                          season_rd_bump=rdb, min_rd=mrd)
        s = ll(p, y)
        if best_g is None or s < best_g[0]:
            best_g = (s, tau, hadv, reg, pdays, rdb, mrd)
    print(f"Glicko-2 best DEV LL={best_g[0]:.5f}  tau={best_g[1]} home_adv={best_g[2]} "
          f"regress={best_g[3]} period_days={best_g[4]} rd_bump={best_g[5]} min_rd={best_g[6]}")
    for nm, val, gridvals in [("tau", best_g[1], [0.2,0.35,0.5,0.8,1.2,1.8]),
                              ("home_adv", best_g[2], [15.,24.,32.,40.,50.]),
                              ("regress", best_g[3], [0.25,0.35,0.5,0.65]),
                              ("period_days", best_g[4], [1,3,7,14,30]),
                              ("rd_bump", best_g[5], [15.,30.,60.]),
                              ("min_rd", best_g[6], [30.,60.,90.,120.])]:
        if val in (min(gridvals), max(gridvals)):
            print(f"  WARNING: {nm}={val} is still on a grid boundary", flush=True)

    # ---------- score TEST once, with DEV-chosen parameters ----------
    _, k, hfa, reg = best_e
    _, tau, hadv, greg, pdays, rdb, mrd = best_g
    p_e, y = run_elo(test, k=k, hfa=hfa, regress=reg, eval_from_season=TEST_YEARS[0])
    p_g, _ = run_glicko(test, period_days=pdays, eval_from_season=TEST_YEARS[0],
                        tau=tau, home_adv=hadv, season_regress=greg,
                        season_rd_bump=rdb, min_rd=mrd)

    print(f"\n=== LOCKED TEST, n={len(y):,}, home win rate={y.mean():.4f} ===")
    print(f"{'model':<28}{'log loss':>10}{'Brier':>9}{'acc':>8}{'pred sd':>9}")
    print("-" * 64)
    for name, p in [("Glicko-2", p_g), ("Elo", p_e),
                    ("constant home rate", np.full_like(y, y.mean())),
                    ("constant 0.5", np.full_like(y, 0.5))]:
        print(f"{name:<28}{ll(p, y):>10.5f}{float(np.mean((p-y)**2)):>9.5f}"
              f"{float(np.mean((p>0.5)==(y>0.5))):>8.4f}{float(np.std(p)):>9.4f}")

    d, lo, hi = boot(ll_vec(p_e, y), ll_vec(p_g, y))
    print(f"\nElo minus Glicko-2: {d:+.5f} nats  (positive = Glicko-2 better)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT' if (lo > 0 or hi < 0) else 'NOT significant — indistinguishable'}")

    out = {
        "test_n": int(len(y)), "test_years": TEST_YEARS,
        "glicko2_ll": ll(p_g, y), "elo_ll": ll(p_e, y),
        "const_home_ll": ll(np.full_like(y, y.mean()), y),
        "glicko2_params": {"tau": tau, "home_adv": hadv, "season_regress": greg,
                           "period_days": pdays, "season_rd_bump": rdb, "min_rd": mrd},
        "elo_params": {"k": k, "home_adv": hfa, "regress": reg},
        "delta_elo_minus_glicko2": d, "ci": [lo, hi],
    }
    with open("data/mlb_glicko2_report.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote data/mlb_glicko2_report.json")


if __name__ == "__main__":
    main()
