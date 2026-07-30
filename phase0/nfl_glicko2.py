"""NFL: fair Glicko-2 vs tuned-Elo comparison on the locked split.

Same discipline as the MLB/NHL comparisons: IDENTICAL trial budget to Elo's search
(1,200), continuous boundary-free ranges, DEV 1999-2015 (scored from 2001) for all
tuning, TEST 2016-2025 scored exactly once, paired bootstrap on the per-game LL delta
vs the frozen Elo baseline (data/nfl_elo_base.json).

Within a rating period every game is predicted before any of them updates a rating.
Neutral sites are predicted with home_adv=0 (same rule the Elo baseline uses).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date

import numpy as np

sys.path.insert(0, "phase0")
from glicko2 import Glicko2  # noqa: E402
from nfl_elo import DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, ll, load_games, run_elo  # noqa: E402


def _pkey(datestr: str, days: int) -> int:
    y, m, d = int(datestr[:4]), int(datestr[5:7]), int(datestr[8:10])
    return date(y, m, d).toordinal() // days


def expect(sys_: Glicko2, home: str, away: str, neutral: bool) -> float:
    if not neutral:
        return sys_.expect_home(home, away)
    saved = sys_.home_adv
    sys_.home_adv = 0.0
    try:
        return sys_.expect_home(home, away)
    finally:
        sys_.home_adv = saved


def run_glicko(games, *, period_days, score_from, score_to=None, **kw):
    sys_ = Glicko2(**kw)
    ps, ys = [], []
    batch, bkey, prev = [], None, None
    for g in games:
        if prev is not None and g["season"] != prev:
            if batch:
                sys_.update_period(batch)
                batch, bkey = [], None
            sys_.new_season()
        prev = g["season"]
        k = _pkey(g["date"], period_days)
        if bkey is not None and k != bkey and batch:
            sys_.update_period(batch)
            batch = []
        bkey = k
        if g["season"] >= score_from and (score_to is None or g["season"] <= score_to):
            ps.append(expect(sys_, g["home"], g["away"], g["neutral"]))
            ys.append(g["y"])
        batch.append((g["home"], g["away"], g["y"]))
    if batch:
        sys_.update_period(batch)
    return np.array(ps), np.array(ys)


def main():
    games = load_games()
    dev = [g for g in games if g["season"] in DEV_YEARS]
    base = json.load(open("data/nfl_elo_base.json"))
    print(f"Elo baseline: DEV {base['dev_ll']}  TEST {base['test_ll']}  (n={base['test_n']})")

    rng = np.random.default_rng(20260723)
    N = 1200
    PERIODS = [1, 3, 7, 14]
    t0 = time.time()
    best = None
    for i in range(N):
        cfg = dict(
            tau=float(np.exp(rng.uniform(np.log(0.05), np.log(2.5)))),
            home_adv=float(rng.uniform(0.0, 120.0)),
            season_regress=float(rng.uniform(0.0, 0.8)),
            season_rd_bump=float(rng.uniform(5.0, 120.0)),
            min_rd=float(rng.uniform(10.0, 150.0)),
        )
        pdays = int(rng.choice(PERIODS))
        p, y = run_glicko(dev, period_days=pdays, score_from=DEV_SCORE_FROM, **cfg)
        s = ll(p, y)
        if best is None or s < best[0]:
            best = (s, cfg, pdays)
        if (i + 1) % 300 == 0:
            print(f"  ...{i+1}/{N}  best DEV LL {best[0]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    cfg, pdays = best[1], best[2]
    print(f"\nGlicko-2 best DEV LL {best[0]:.5f}  (Elo DEV {base['dev_ll']})")
    print("  params:", {k: round(v, 4) for k, v in cfg.items()}, f"period_days={pdays}")

    # ---- TEST: both systems scored once on the same games, paired ----
    pg, y = run_glicko(games, period_days=pdays, score_from=TEST_YEARS.start,
                       score_to=TEST_YEARS.stop - 1, **cfg)
    pe, ye = run_elo(games, score_from=TEST_YEARS.start, score_to=TEST_YEARS.stop - 1,
                     **base["params"])
    assert len(pg) == len(pe) and np.array_equal(y, ye)
    llg, lle = ll(pg, y), ll(pe, y)

    eps = 1e-12
    def v(p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = v(pe) - v(pg)                      # >0 => Glicko-2 better
    rng2 = np.random.default_rng(7)
    n = len(d)
    bs = d[rng2.integers(0, n, size=(10000, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = ("Glicko-2 SIG better" if lo > 0 else
           ("Elo SIG better" if hi < 0 else "inside noise"))
    print(f"\nTEST 2016-2025 (n={n}, scored once):")
    print(f"  tuned Elo      LL {lle:.5f}")
    print(f"  tuned Glicko-2 LL {llg:.5f}")
    print(f"  paired delta {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
    json.dump({"glicko2": {"dev_ll": round(best[0], 5), "params": cfg, "period_days": pdays,
                           "test_ll": round(llg, 5)},
               "elo_test_ll": round(lle, 5),
               "delta": round(float(d.mean()), 5), "ci": [round(lo, 5), round(hi, 5)],
               "verdict": sig},
              open("data/nfl_glicko2_report.json", "w"), indent=1)
    print("wrote data/nfl_glicko2_report.json")


if __name__ == "__main__":
    main()
