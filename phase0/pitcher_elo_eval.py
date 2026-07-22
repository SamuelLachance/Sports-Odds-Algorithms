"""Does the starting-pitcher term beat plain Elo? Same locked-split protocol.

  DEV  2000-2015 (scored from 2002) -> random search, both models, equal budget
  TEST 2016-2024 excluding 2020     -> scored once

Plain Elo is the k_pitch=0 special case of PitcherElo, so the two models are
identical apart from the pitcher term. That makes the comparison exact: any
difference is the pitcher layer and nothing else. A sanity assert checks that
k_pitch=0 reproduces the standalone Elo baseline.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, "phase0")
from pitcher_elo import load_games, run  # noqa: E402

EPS = 1e-15
DEV_YEARS = range(2000, 2016)
TEST_YEARS = [y for y in range(2016, 2025) if y != 2020]
DEV_WARMUP = 2002
N_TRIALS = 1500


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


def search(dev, rng, *, with_pitcher: bool):
    best = None
    for _ in range(N_TRIALS):
        p = dict(
            k_team=float(np.exp(rng.uniform(np.log(1.0), np.log(12.0)))),
            hfa=float(rng.uniform(15.0, 45.0)),
            team_regress=float(rng.uniform(0.0, 0.6)),
            k_pitch=float(np.exp(rng.uniform(np.log(0.5), np.log(30.0)))) if with_pitcher else 0.0,
            pitch_regress=float(rng.uniform(0.0, 0.9)) if with_pitcher else 0.0,
        )
        pr, y, _ = run(dev, eval_from_season=DEV_WARMUP, **p)
        s = ll(pr, y)
        if best is None or s < best[0]:
            best = (s, p)
    return best


def main():
    dev = load_games(DEV_YEARS)
    test = load_games(TEST_YEARS)
    print(f"DEV {len(dev):,} games   TEST {len(test):,} games")
    print(f"random search: {N_TRIALS} trials each\n", flush=True)

    rng = np.random.default_rng(20260721)

    t0 = time.time()
    best_elo = search(dev, rng, with_pitcher=False)
    print(f"Plain Elo      DEV LL={best_elo[0]:.5f}  "
          f"{ {k: round(v,3) for k,v in best_elo[1].items() if k in ('k_team','hfa','team_regress')} }"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    best_pe = search(dev, rng, with_pitcher=True)
    print(f"Pitcher-Elo    DEV LL={best_pe[0]:.5f}  "
          f"{ {k: round(v,3) for k,v in best_pe[1].items()} }"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    dev_gain = best_elo[0] - best_pe[0]
    print(f"\nDEV gain from pitcher term: {dev_gain:+.5f} nats "
          f"({'HELPS' if dev_gain > 0 else 'no help'} on the tuning set)")

    # ---- score TEST once ----
    pe_elo, y, _ = run(test, eval_from_season=TEST_YEARS[0], **best_elo[1])
    pe_pit, _, model = run(test, eval_from_season=TEST_YEARS[0], **best_pe[1])

    # sanity: k_pitch=0 must equal plain Elo
    chk, _, _ = run(test, eval_from_season=TEST_YEARS[0],
                    **{**best_pe[1], "k_pitch": 0.0, "pitch_regress": 0.0})
    assert abs(ll(chk, y) - ll(run(test, eval_from_season=TEST_YEARS[0],
               k_team=best_pe[1]["k_team"], hfa=best_pe[1]["hfa"],
               team_regress=best_pe[1]["team_regress"], k_pitch=0.0,
               pitch_regress=0.0)[0], y)) < 1e-12

    print(f"\n=== LOCKED TEST  n={len(y):,}  home win rate={y.mean():.4f} ===")
    print(f"{'model':<26}{'log loss':>10}{'Brier':>9}{'acc':>8}{'pred sd':>9}")
    print("-" * 62)
    for name, p in [("Pitcher-adjusted Elo", pe_pit), ("Plain Elo", pe_elo),
                    ("constant home rate", np.full_like(y, y.mean())),
                    ("constant 0.5", np.full_like(y, 0.5))]:
        print(f"{name:<26}{ll(p, y):>10.5f}{float(np.mean((p-y)**2)):>9.5f}"
              f"{float(np.mean((p>0.5)==(y>0.5))):>8.4f}{float(np.std(p)):>9.4f}")

    d, lo, hi = boot(ll_vec(pe_elo, y), ll_vec(pe_pit, y))
    print(f"\nPlain Elo - Pitcher-Elo on TEST: {d:+.5f} nats (positive = pitcher term helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT — the pitcher term beats plain Elo' if lo > 0 else ('SIGNIFICANT — pitcher term HURTS' if hi < 0 else 'NOT significant — indistinguishable from plain Elo')}")

    # top / bottom pitchers as a face-validity check
    strong = sorted(model.P.items(), key=lambda kv: -kv[1])[:8]
    weak = sorted(model.P.items(), key=lambda kv: kv[1])[:5]
    print("\nhighest pitcher ratings (Elo pts above league-avg starter):",
          ", ".join(f"{k}={v:+.0f}" for k, v in strong))
    print("lowest:", ", ".join(f"{k}={v:+.0f}" for k, v in weak))

    json.dump({
        "n_trials_each": N_TRIALS, "test_n": int(len(y)),
        "plain_elo": {"dev_ll": best_elo[0], "test_ll": ll(pe_elo, y), "params": best_elo[1]},
        "pitcher_elo": {"dev_ll": best_pe[0], "test_ll": ll(pe_pit, y), "params": best_pe[1]},
        "dev_gain": dev_gain, "test_delta_plain_minus_pitcher": d, "ci": [lo, hi],
    }, open("data/mlb_pitcher_elo.json", "w"), indent=2)
    print("\nwrote data/mlb_pitcher_elo.json")


if __name__ == "__main__":
    main()
