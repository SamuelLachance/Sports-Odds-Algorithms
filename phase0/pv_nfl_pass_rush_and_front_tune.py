"""pv NFL pass_rush_and_front -- step 3: small DEV grid for the PLAYER-rating dynamics. DEV ONLY.

Objective: the rating's own likelihood -- Poisson deviance of each PRESENT
front defender's NEXT-GAME credited events (sacks, pressure = sack + non-sack
hit, run stops), predicted by his PRE-GAME theta x the game's opportunities x
the pre-game league rate x the pre-game opponent (and scorer) factor. Rows:
DEV seasons 2006-2015, buckets DE/DT/OLB/ILB. The dynamics of the unit factors
(K_u, d_u, s_u) are held at the defaults (unit-level validity is measured in
the validation step, not tuned here).

Grid: K_p (prior strength, expected-unit-event units) x d_p (per-game decay) x
s_p (season carry). 27 builds, ~10 s each.
Output: data/pv_nfl_pass_rush_and_front_tune.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_nfl_pass_rush_and_front as E  # noqa: E402

FRONT = {"DE", "DT", "OLB", "ILB"}


def pois_dev(y, mu):
    mu = np.clip(mu, 1e-9, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (t - (y - mu))


def score(PGo, TGo):
    t = TGo.set_index(["game_id", "defteam"])
    x = PGo[(PGo.presence != "") & PGo.bucket.isin(FRONT) & PGo.season.between(2006, 2015)]
    x = x.join(t[["g_n_db", "g_n_run", "L_pr", "L_sk", "L_st", "Oopp_pr", "Oopp_sk", "Oopp_st", "S_ht"]],
               on=["game_id", "team"])
    x = x[x.L_pr.notna() & x.L_sk.notna() & x.L_st.notna()]
    res = {}
    for k, yc, n, sc in (("pr", "g_pr", "g_n_db", True), ("sk", "g_sack", "g_n_db", False),
                         ("st", "g_stop", "g_n_run", False)):
        mu = x[f"th_{k}"] * x[n] * x[f"L_{k}"] * x[f"Oopp_{k}"] * (x.S_ht if sc else 1.0)
        res[k] = float(pois_dev(x[yc].to_numpy(float), mu.to_numpy(float)).mean())
    res["n"] = int(len(x))
    return res


def main():
    inputs = E.load_inputs(2015)
    # round 1 (K 6/12/25 x d .97/.985/.995 x s .7/.85/1) put every kind's optimum on the fast edge
    # (d .97, s .7); round 2 extends the grid past that edge.
    grid = list(itertools.product([8.0, 16.0, 32.0], [0.93, 0.95, 0.97], [0.4, 0.55, 0.7]))
    out = []
    for K, d, s in grid:
        PGo, TGo, _ = E.build({"K_p": K, "d_p": d, "s_p": s}, through=2015, inputs=inputs)
        r = score(PGo, TGo)
        r.update(K_p=K, d_p=d, s_p=s)
        out.append(r)
        print(f"K={K:5.1f} d={d:.3f} s={s:.2f}  dev pr {r['pr']:.5f} sk {r['sk']:.5f} st {r['st']:.5f}  n={r['n']}",
              flush=True)
    best = {k: min(out, key=lambda r: r[k]) for k in ("pr", "sk", "st")}
    json.dump({"objective": "mean Poisson deviance of next-game credits, present DE/DT/OLB/ILB, DEV 2006-2015",
               "round1_best": {"pr": [12, .97, .70], "sk": [12, .97, .85], "st": [25, .97, .70]}, "grid": out, "best": best}, open(os.path.join(E.DATA, "pv_nfl_pass_rush_and_front_tune.json"), "w"),
              indent=1)
    for k, r in best.items():
        print("best", k, {kk: r[kk] for kk in ("K_p", "d_p", "s_p", k)})


if __name__ == "__main__":
    main()
