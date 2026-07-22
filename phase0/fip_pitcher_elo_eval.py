"""FIP-based pitcher-adjusted Elo vs plain Elo — same locked-split protocol.

  DEV  2000-2015 (from 2002) -> tune both, equal budget
  TEST 2016-2024 ex-2020     -> score once, paired bootstrap

beta=0 collapses to plain Elo, so the two models differ only by the FIP pitcher
adjustment. The league FIP constant is computed on DEV only (never on test data).
"""

from __future__ import annotations

import json
import os
import sys
import time

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np  # noqa: E402

sys.path.insert(0, "phase0")
from fip_pitcher_elo import league_fip_core, load_games, run  # noqa: E402

EPS = 1e-15
DEV_YEARS = range(2000, 2016)
TEST_YEARS = [y for y in range(2016, 2025) if y != 2020]
DEV_WARMUP = 2002
N_TRIALS = 1200


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


def search(dev, lg_fip, rng, *, with_pitcher):
    best = None
    for _ in range(N_TRIALS):
        p = dict(
            k_team=float(np.exp(rng.uniform(np.log(1.0), np.log(12.0)))),
            hfa=float(rng.uniform(15.0, 45.0)),
            team_regress=float(rng.uniform(0.0, 0.6)),
            beta=float(rng.uniform(2.0, 40.0)) if with_pitcher else 0.0,
            decay=float(rng.uniform(0.80, 0.99)) if with_pitcher else 0.9,
            prior_ip=float(rng.uniform(20.0, 200.0)) if with_pitcher else 100.0,
            season_decay=float(rng.uniform(0.0, 0.7)) if with_pitcher else 0.0,
        )
        pr, y, _ = run(dev, eval_from_season=DEV_WARMUP, lg_fip=lg_fip, **p)
        s = ll(pr, y)
        if best is None or s < best[0]:
            best = (s, p)
    return best


def main():
    dev = load_games(DEV_YEARS)
    test = load_games(TEST_YEARS)
    lg_fip = league_fip_core(dev)
    cov = np.mean([bool(g["home_line"]) and bool(g["away_line"]) for g in test])
    print(f"DEV {len(dev):,}  TEST {len(test):,}  league FIP-core(dev)={lg_fip:.3f}")
    print(f"TEST games with both starter lines parsed: {cov*100:.1f}%")
    print(f"random search {N_TRIALS} trials each\n", flush=True)

    rng = np.random.default_rng(20260721)

    t0 = time.time()
    be = search(dev, lg_fip, rng, with_pitcher=False)
    print(f"Plain Elo   DEV LL={be[0]:.5f}  "
          f"{ {k: round(v,3) for k,v in be[1].items() if k in ('k_team','hfa','team_regress')} }"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    bp = search(dev, lg_fip, rng, with_pitcher=True)
    print(f"FIP-Elo     DEV LL={bp[0]:.5f}  "
          f"{ {k: round(v,3) for k,v in bp[1].items()} }  ({time.time()-t0:.0f}s)", flush=True)

    dev_gain = be[0] - bp[0]
    print(f"\nDEV gain from FIP pitcher term: {dev_gain:+.5f} nats", flush=True)

    pe, y, _ = run(test, eval_from_season=TEST_YEARS[0], lg_fip=lg_fip, **be[1])
    pp, _, model = run(test, eval_from_season=TEST_YEARS[0], lg_fip=lg_fip, **bp[1])

    print(f"\n=== LOCKED TEST  n={len(y):,}  home win rate={y.mean():.4f} ===")
    print(f"{'model':<28}{'log loss':>10}{'Brier':>9}{'acc':>8}{'pred sd':>9}")
    print("-" * 64)
    for name, p in [("FIP pitcher-adjusted Elo", pp), ("Plain Elo", pe),
                    ("constant home rate", np.full_like(y, y.mean())),
                    ("constant 0.5", np.full_like(y, 0.5))]:
        print(f"{name:<28}{ll(p, y):>10.5f}{float(np.mean((p-y)**2)):>9.5f}"
              f"{float(np.mean((p>0.5)==(y>0.5))):>8.4f}{float(np.std(p)):>9.4f}")

    d, lo, hi = boot(ll_vec(pe, y), ll_vec(pp, y))
    print(f"\nPlain Elo - FIP-Elo on TEST: {d:+.5f} nats (positive = FIP term helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT — FIP pitcher term beats plain Elo' if lo > 0 else ('SIGNIFICANT — it HURTS' if hi < 0 else 'NOT significant')}")

    # face validity: best/worst starters by FIP adjustment at end of test
    adj = {p: model._adj(p) for p in model.pf if model.pf[p]["outs"] > 200}
    top = sorted(adj.items(), key=lambda kv: -kv[1])[:8]
    print("\nbest starters by FIP adj (Elo pts):", ", ".join(f"{k}={v:+.0f}" for k, v in top))

    json.dump({
        "test_n": int(len(y)), "league_fip": lg_fip,
        "plain_elo": {"dev_ll": be[0], "test_ll": ll(pe, y), "params": be[1]},
        "fip_elo": {"dev_ll": bp[0], "test_ll": ll(pp, y), "params": bp[1]},
        "dev_gain": dev_gain, "test_delta": d, "ci": [lo, hi],
    }, open("data/mlb_fip_elo.json", "w"), indent=2)
    print("\nwrote data/mlb_fip_elo.json")


if __name__ == "__main__":
    main()
