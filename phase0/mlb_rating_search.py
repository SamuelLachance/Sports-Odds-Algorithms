"""Fair, boundary-free comparison of Glicko-2 vs Elo for MLB.

Why random search rather than a grid: the first attempt used a coarse grid and four
of five Glicko-2 parameters landed on an edge, which means the optimum lay outside
the search box — Glicko-2 was under-tuned and then reported as the loser. Sampling
continuous ranges removes boundaries as a concept, and both systems get an IDENTICAL
number of trials so neither wins on search budget.

Protocol (unchanged, fixed before results were seen):
  DEV  2000-2015, scored from 2002 -> all tuning
  TEST 2016-2024 excluding 2020    -> scored exactly once, with DEV-chosen params

Data: Retrosheet game logs. "The information used here was obtained free of charge
from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, "phase0")
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", "phase0/mlb_glicko2_eval.py")
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)

N_TRIALS = 1200
PERIOD_CHOICES = [1, 2, 3, 5, 7, 10, 14, 21, 30]


def main():
    rng = np.random.default_rng(20260721)
    dev = _ev.load_games(_ev.DEV_YEARS)
    test = _ev.load_games(_ev.TEST_YEARS)
    print(f"DEV  {len(dev):,} games   TEST {len(test):,} games")
    print(f"random search: {N_TRIALS} trials each, continuous ranges\n", flush=True)

    # ---------------- Elo ----------------
    t0 = time.time()
    best_e = None
    for i in range(N_TRIALS):
        k = float(np.exp(rng.uniform(np.log(0.5), np.log(20.0))))
        hfa = float(rng.uniform(5.0, 60.0))
        reg = float(rng.uniform(0.0, 0.8))
        p, y = _ev.run_elo(dev, k=k, hfa=hfa, regress=reg,
                           eval_from_season=_ev.DEV_WARMUP_FROM)
        s = _ev.ll(p, y)
        if best_e is None or s < best_e[0]:
            best_e = (s, dict(k=k, hfa=hfa, regress=reg))
    print(f"Elo      best DEV LL={best_e[0]:.5f}  "
          f"{ {a: round(b,3) for a,b in best_e[1].items()} }   ({time.time()-t0:.0f}s)",
          flush=True)

    # ---------------- Glicko-2 ----------------
    t0 = time.time()
    best_g = None
    for i in range(N_TRIALS):
        cfg = dict(
            tau=float(np.exp(rng.uniform(np.log(0.05), np.log(2.5)))),
            home_adv=float(rng.uniform(5.0, 60.0)),
            season_regress=float(rng.uniform(0.0, 0.8)),
            season_rd_bump=float(rng.uniform(5.0, 100.0)),
            min_rd=float(rng.uniform(10.0, 150.0)),
        )
        pdays = int(rng.choice(PERIOD_CHOICES))
        p, y = _ev.run_glicko(dev, period_days=pdays,
                              eval_from_season=_ev.DEV_WARMUP_FROM, **cfg)
        s = _ev.ll(p, y)
        if best_g is None or s < best_g[0]:
            best_g = (s, {**cfg, "period_days": pdays})
        if (i + 1) % 300 == 0:
            print(f"  ...{i+1}/{N_TRIALS} trials, best {best_g[0]:.5f}"
                  f" ({time.time()-t0:.0f}s)", flush=True)
    print(f"Glicko-2 best DEV LL={best_g[0]:.5f}  "
          f"{ {a: (round(b,3) if isinstance(b,float) else b) for a,b in best_g[1].items()} }"
          f"   ({time.time()-t0:.0f}s)", flush=True)

    dev_gap = best_e[0] - best_g[0]
    print(f"\nDEV: Elo - Glicko-2 = {dev_gap:+.5f} "
          f"({'Glicko-2 better' if dev_gap > 0 else 'Elo better'} on the tuning set)")

    # ---------------- score TEST once ----------------
    pe, y = _ev.run_elo(test, **best_e[1], eval_from_season=_ev.TEST_YEARS[0])
    gp = dict(best_g[1])
    pd_ = gp.pop("period_days")
    pg, _ = _ev.run_glicko(test, period_days=pd_,
                           eval_from_season=_ev.TEST_YEARS[0], **gp)

    print(f"\n=== LOCKED TEST  n={len(y):,}  home win rate={y.mean():.4f} ===")
    print(f"{'model':<24}{'log loss':>10}{'Brier':>9}{'acc':>8}{'pred sd':>9}")
    print("-" * 60)
    for name, p in [("Glicko-2", pg), ("Elo", pe),
                    ("constant home rate", np.full_like(y, y.mean())),
                    ("constant 0.5", np.full_like(y, 0.5))]:
        print(f"{name:<24}{_ev.ll(p, y):>10.5f}{float(np.mean((p-y)**2)):>9.5f}"
              f"{float(np.mean((p>0.5)==(y>0.5))):>8.4f}{float(np.std(p)):>9.4f}")

    d, lo, hi = _ev.boot(_ev.ll_vec(pe, y), _ev.ll_vec(pg, y))
    print(f"\nElo - Glicko-2 on TEST: {d:+.5f} nats (positive = Glicko-2 better)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT' if (lo > 0 or hi < 0) else 'NOT significant — indistinguishable'}")

    json.dump({
        "n_trials_each": N_TRIALS,
        "elo": {"dev_ll": best_e[0], "params": best_e[1], "test_ll": _ev.ll(pe, y)},
        "glicko2": {"dev_ll": best_g[0], "params": best_g[1], "test_ll": _ev.ll(pg, y)},
        "test_n": int(len(y)), "delta_elo_minus_glicko2": d, "ci": [lo, hi],
    }, open("data/mlb_rating_search.json", "w"), indent=2)
    print("\nwrote data/mlb_rating_search.json")


if __name__ == "__main__":
    main()
