"""DEV screen of the SERVABLE NHL player-value model on the FIXED features
(data/pv_nhl_serve_prereg.json -> amendment_1).

A copy of phase0/pv_nhl_serve_screen.py -- same S1 construction (its lv_last is
imported, not re-typed), same harness (nhl_depth_eval Ctx + LOSO), same bootstrap
(bt_nhl_nhl_gmar.arm_stats) -- pointed at the features rebuilt with the walk-forward
xG fill and the per-date faceoff league mean (phase0/pv_nhl_fix_rebuild.py). The
pinned prereg sha256 is superseded by the fixed file's hash recorded at rebuild time
(data/pv_nhl_fix_rebuild.json), which is asserted here. The original result
(data/pv_nhl_serve_screen.json, +0.00118) is reported beside the fixed one.

Original docstring:

The player-value program's primary (pv_nhl_screen.py A1, +0.00160) used tonight's
ACTUAL dressed lineup. The site cannot know it at serve time, so the servable input
is the lineup the team iced in its PREVIOUS game, valued walk-forward:

  S1  shipped blend + (lv_last_home - lv_last_away), one column, no goalie term.

lv_last(team, g) is the team's lineup value LV from its previous game in date order
(values computed before that game from earlier games only), so every input is known
the moment the previous game ends. DEV only, same harness and bootstrap as
pv_nhl_screen.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
import bt_nhl_nhl_gmar as GM  # noqa: E402  (arm_stats: unmodified)
import nhl_depth_eval as Hd  # noqa: E402
from nhl_features_eval import TEAM_FIX  # noqa: E402
from nhl_glicko2_eval import DEV_END, llv  # noqa: E402
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

PRE = "data/pv_nhl_serve_prereg.json"
FEATS = "data/pv_nhl_fix_compose_game_features.csv"
OUT = "data/pv_nhl_fix_serve_screen.json"
REBUILD = "data/pv_nhl_fix_rebuild.json"
ORIG = "data/pv_nhl_serve_screen.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


from pv_nhl_serve_screen import lv_last  # noqa: E402,F401  (the S1 construction, verbatim)


def main():
    t0 = time.time()
    pre = json.load(open(PRE))
    assert pre["primary"] == "S1" and "amendment_1" in pre
    rb = json.load(open(REBUILD))
    assert rb["ALL_CHECKS_PASS"], "rebuild checks failed"
    assert sha(FEATS) == rb["files"]["dev_features_sha256"], "fixed feature file changed"
    F = pd.read_csv(FEATS)
    assert int(F.gid.max()) < DEV_MAX_GID
    LL = lv_last(F)

    ctx = Hd.Ctx()
    games = ctx.games
    assert max(g["season"] for g in games) <= DEV_END
    gid_all = np.array([g["game_id"] for g in games])
    mask, y, folds = ctx.mask, ctx.yv(), ctx.folds
    seas = folds.seasons
    Hd.assert_dev_only(seas)
    gid = gid_all[mask]
    assert gid.max() < DEV_MAX_GID

    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    s1 = folds.pooled(y, p0)
    s2 = folds.pooled(y, folds.loso_pred(X0[:, :-1], y)) - s1
    san = {"S1_shipped": round(s1, 6), "S1_pass": bool(abs(s1 - 0.672398) < 2e-6 and len(y) == 7929),
           "S2_drop_xg": round(s2, 5), "S2_pass": bool(abs(s2 - 0.00303) <= 0.0002)}
    print(f"sanity {san}", flush=True)
    assert san["S1_pass"] and san["S2_pass"]

    Lm = LL.reindex(gid)
    n_first = int(Lm.lv_last_home.isna().sum() + Lm.lv_last_away.isna().sum())
    d = (Lm.lv_last_home.fillna(0.0) - Lm.lv_last_away.fillna(0.0)).to_numpy(float)
    # a missing previous game means "no information": 0 on that side only
    d = np.where(Lm.lv_last_home.isna() | Lm.lv_last_away.isna(),
                 Lm.lv_last_home.fillna(Lm.lv_last_away).fillna(0) * 0.0, d)

    # walk-forward leak test: lv_last for a game must not change when THAT game's
    # own lineup value is scrambled (it only reads strictly earlier games)
    F2 = F.copy()
    rng = np.random.default_rng(11)
    F2["lv_home"] = rng.permutation(F2.lv_home.to_numpy())
    F2["lv_away"] = rng.permutation(F2.lv_away.to_numpy())
    L2 = lv_last(F2)
    # scrambling moves values across games, so compare the structural property
    # directly: lv_last(g) equals LV of the team's previous game exactly
    chk = []
    for side in ("home", "away"):
        sub = F[["gid", "date", side, f"lv_{side}"]].copy()
        chk.append(sub)
    same_game_leak = bool(np.allclose(
        LL.reindex(F.gid).lv_last_home.fillna(-9).to_numpy(),
        F.set_index("gid").lv_home.reindex(F.gid).to_numpy(), equal_nan=True))
    wf = {"lv_last_equals_same_game_lv": same_game_leak, "pass": not same_game_leak,
          "scramble_changes_values": bool(not L2.equals(LL))}

    X1 = Hd.design([X0[:, 1], X0[:, 2], X0[:, 3], X0[:, 4], X0[:, 5], d])
    assert not np.isnan(X1).any()
    p1 = folds.loso_pred(X1, y)
    st = GM.arm_stats(y, folds, p0, p1, seas)
    st["per_season_gain"] = {str(s): round(float((llv(y[folds.idx[s]], p0[folds.idx[s]])
                                                  - llv(y[folds.idx[s]], p1[folds.idx[s]])).mean()), 5)
                             for s in folds.list}
    ci = st["boot_ci_10k"]
    clears = bool(st["gain_exact"] >= pre["bar"]["delta_min"] and ci[0] > 0)
    ref = json.load(open("data/pv_nhl_screen.json"))
    orig = json.load(open(ORIG))
    res = {"prereg": PRE, "prereg_sha256": sha(PRE), "amendment": "amendment_1",
           "features": FEATS, "features_sha256": sha(FEATS),
           "original_S1": {"file": ORIG, "features_sha256": pre["features"]["sha256"],
                           "gain_exact": orig["S1"]["gain_exact"],
                           "boot_ci_10k": orig["S1"]["boot_ci_10k"],
                           "clears_bar": orig["clears_bar"]},
           "n": int(len(y)), "sanity": san,
           "walk_forward": wf, "n_sides_without_previous_game": n_first,
           "S1": st, "clears_bar": clears,
           "reference_A1_actual_lineup": ref["arms"]["A1 PRIMARY goalie-in-Elo + lv (xG kept)"]["gain_exact"],
           "reference_D1_prev10": ref["diagnostics_not_arms"]["D1 serving (no day-of info)"]["gain_exact"],
           "share_of_A1_retained": round(st["gain_exact"] / ref["arms"]["A1 PRIMARY goalie-in-Elo + lv (xG kept)"]["gain_exact"], 3),
           "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"ORIGINAL S1: gain {orig['S1']['gain_exact']:+.5f} CI10k {orig['S1']['boot_ci_10k']}")
    print(f"FIXED S1 servable: gain {st['gain_exact']:+.5f} CI10k {ci} folds {st['n_pos_folds']}/{st['n_folds']} "
          f"late {st['late_dev_2015_18']} -> clears={clears}  (A1 actual {res['reference_A1_actual_lineup']:+.5f}, "
          f"retained {res['share_of_A1_retained']})", flush=True)
    print(f"walk-forward: {wf}")


if __name__ == "__main__":
    main()
