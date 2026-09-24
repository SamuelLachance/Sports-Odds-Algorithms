"""NHL gap anatomy -- PROBE (not a screen): does the MLB construction transfer?
Starting goalie INSIDE the Elo, measured from his own process stat (GSAx =
xGA - GA per appearance), EWMA-decayed, EB-regressed, converted to Elo points
and added to his team's rating for that game; the team Elo then updates on the
goalie-adjusted expectation, so the team rating learns "team net of goalie".

DEV ONLY. Every goalie/team state is read before the game updates it. Tonight's
starter (first-shot goalie; 99.99% = sole dressed goalie) is day-of public info.
Probe grid is tiny and fixed a priori; it only sizes the channel so the anatomy
can say whether the MLB lesson has room in hockey. Output: data/bt_nhl_anatomy_probe.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import llv  # noqa: E402
import bt_nhl_anatomy_build as B  # noqa: E402
from bt_nhl_anatomy import boot_mean  # noqa: E402

OUT = "data/bt_nhl_anatomy_probe.json"
K, HA, REG = Hd.SHIPPED_ELO
DECAY, PRIOR_N = 0.98, 20.0


def goalie_elo(games, starters, gg, c_pts, role_prior):
    R = {}
    num, den, starts = defaultdict(float), defaultdict(float), defaultdict(int)
    out = np.empty(len(games))
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - REG)
        prev = g["season"]
        st = starters.get(g["game_id"], {})
        adj = {}
        for side, flag in (("home", 1), ("away", 0)):
            gk = st.get(flag)
            if gk is None:
                adj[side] = 0.0
                continue
            m0 = 0.0
            if role_prior:
                n = starts[gk]
                m0 = -0.10 if n < 30 else (-0.05 if n < 100 else 0.0)
            rt = (num[gk] + m0 * PRIOR_N) / (den[gk] + PRIOR_N)
            adj[side] = c_pts * rt
        rh = R.setdefault(g["home"], 1500.0) + adj["home"]
        ra = R.setdefault(g["away"], 1500.0) + adj["away"]
        p = 1.0 / (1.0 + 10 ** (-((rh + HA) - ra) / 400.0))
        out[i] = p
        R[g["home"]] += K * (g["y"] - p)
        R[g["away"]] -= K * (g["y"] - p)
        for side, flag in (("home", 1), ("away", 0)):
            gk = st.get(flag)
            if gk is not None:
                starts[gk] += 1
        for gk, _team, xga, ga in gg.get(g["game_id"], []):
            num[gk] = DECAY * num[gk] + (xga - ga)
            den[gk] = DECAY * den[gk] + 1.0
    return out


def main():
    ctx = Hd.Ctx()
    y = ctx.yv()
    folds = ctx.folds
    ids = set(g["game_id"] for g in ctx.games)
    starters = B.load_starters(ids)
    gg, _ = B.load_goalie_games(ids)
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    l0 = llv(y, p0)
    print(f"shipped DEV LOSO {l0.mean():.6f} n={len(y)}")
    # harness sanity: dropping xG must cost ~0.003 (recorded +0.00304)
    pnx = folds.loso_pred(X0[:, :5], y)
    print(f"harness check: drop xG costs {llv(y, pnx).mean() - l0.mean():+.5f} (recorded +0.00304)")
    res = {"shipped": round(float(l0.mean()), 6),
           "drop_xg_cost": round(float(llv(y, pnx).mean() - l0.mean()), 5), "arms": {}}
    for c_pts in (32.0, 64.0, 96.0):
        for rp in (False, True):
            pe = np.clip(goalie_elo(ctx.games, starters, gg, c_pts, rp), 1e-9, 1 - 1e-9)
            el = np.log(pe / (1 - pe))[ctx.mask]
            X = X0.copy()
            X[:, 1] = el
            p = folds.loso_pred(X, y)
            d = l0 - llv(y, p)
            ci = boot_mean(d)
            name = f"c={int(c_pts)} role_prior={rp}"
            res["arms"][name] = {"dev_ll": round(float(llv(y, p).mean()), 6),
                                 "gain": round(float(d.mean()), 5), "ci": ci}
            print(f"  {name:<26} LL {llv(y, p).mean():.6f} gain {d.mean():+.5f} "
                  f"[{ci[0]:+.5f},{ci[1]:+.5f}]")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
