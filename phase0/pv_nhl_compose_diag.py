"""Player-value program, NHL -- COMPOSE post-hoc DIAGNOSTICS of the DEV screen
(DEV ONLY; run AFTER the pre-registered screen; nothing here is a candidate and
nothing here can change the screen's verdict).

  D1  no fitted weight at all: lineup / goalie value at the declared PRIOR weights
      (every goal-unit component at 1, assists at 0) beside the blend -- does the
      gain survive without the GD-fitted weights?
  D2  leave-one-component-out: C1 with one component's term removed from lv
      (weights unchanged) -- which components carry the gain?
  D3  one-component-at-a-time: blend + that component's weighted lineup diff.
  D4  starter-change nights: C1 gain on games where a team starts a goalie other
      than its most frequent starter of its previous 10 games (the anatomy's
      non-#1 nights) vs the rest.

Output data/pv_nhl_compose_diag.json.

    python phase0/pv_nhl_compose_diag.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE)
import nhl_depth_eval as Hd  # noqa: E402
import bt_nhl_nhl_gmar as GM  # noqa: E402
import pv_nhl_compose_core as C  # noqa: E402
from nhl_glicko2_eval import llv  # noqa: E402
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

FEATS = "data/pv_nhl_compose_game_features.csv"
OUT = "data/pv_nhl_compose_diag.json"


def main():
    t0 = time.time()
    F = pd.read_csv(FEATS)
    assert int(F.gid.max()) < DEV_MAX_GID
    W = json.load(open("data/pv_nhl_compose_weights.json"))["walk_forward"]
    F = F.set_index("gid")
    ctx = Hd.Ctx()
    y = ctx.yv()
    folds = ctx.folds
    seas = folds.seasons
    gid = np.array([g["game_id"] for g in ctx.games])[ctx.mask]
    Fg = F.reindex(gid)
    s_arr = Fg.season.to_numpy()

    def d(col):
        return (Fg[col + "_home"] - Fg[col + "_away"]).to_numpy(float)

    # raw (unweighted) lineup aggregate differences, rebuilt from the panel
    sk, gk, tg = C.load_panel()
    x, m, _ = C.components(sk)
    G = C.game_matrix(C.lineup(sk, x, m, C.goalie_component(gk, tg)), tg).set_index("gid")
    LU = {c: G.reindex(gid)["d_" + c].to_numpy(float) for c in C.COMPS}
    wl = {c: d("wLU_" + c) for c in C.COMPS}
    for c in C.COMPS:          # consistency: weighted term = season weight x raw diff
        w = np.array([W[str(int(s))]["w"][c] for s in s_arr])
        assert np.allclose(wl[c], w * LU[c], atol=2e-5), c
    lv = sum(wl[c] for c in C.SK_COMPS)
    assert np.allclose(lv, d("lv"), atol=1e-4), "lv must equal the sum of its weighted terms"
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    res = {"note": "post-hoc diagnostics AFTER the pre-registered screen; not candidates"}

    def arm(cols):
        X = np.column_stack([X0] + cols)
        p = folds.loso_pred(X, y)
        st = GM.arm_stats(y, folds, p0, p, seas)
        return {k: st[k] for k in ("gain", "boot_ci_10k", "sig", "n_pos_folds",
                                   "late_dev_2015_18")}, p

    lv0 = sum(C.PRIOR_W[c] * LU[c] for c in C.SK_COMPS)
    gv0 = C.PRIOR_W["goalie"] * LU["goalie"]
    res["D1_prior_weights_lv_gv"], _ = arm([lv0, gv0])
    res["D1_prior_weights_lv_plus_gv"], _ = arm([lv0 + gv0])
    res["C1_reference"], p1 = arm([lv, d("gv")])
    res["D2_leave_one_out"] = {}
    for c in C.SK_COMPS:
        res["D2_leave_one_out"][c], _ = arm([lv - wl[c], d("gv")])
    res["D3_single"] = {}
    for c in C.COMPS:
        res["D3_single"][c], _ = arm([wl[c]])
    # D4 starter-change nights
    gs = pd.read_parquet("data/pv_nhl_compose_goalie_games.parquet")
    gs = gs[gs.g_start == 1][["gid", "date", "team", "pid"]].sort_values(["date", "gid"])
    gs["usual"] = gs.groupby("team").pid.transform(
        lambda v: v.shift(1).rolling(10, min_periods=1).apply(
            lambda a: pd.Series(a).mode().iloc[0], raw=True))
    gs["non1"] = (gs.usual.notna() & (gs.pid != gs.usual)).astype(int)
    nn = gs.groupby("gid").non1.max()
    non1 = nn.reindex(gid).fillna(0).to_numpy().astype(bool)
    dl = llv(y, p0) - llv(y, p1)
    res["D4_starter_change_nights"] = {
        "n_non1": int(non1.sum()), "gain_non1": round(float(dl[non1].mean()), 5),
        "ci_non1": GM.boot10k(dl[non1]),
        "n_other": int((~non1).sum()), "gain_other": round(float(dl[~non1].mean()), 5),
        "ci_other": GM.boot10k(dl[~non1]),
        "definition": "a side starts a goalie other than the modal starter of its previous "
                      "10 games (strictly earlier)"}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=0)[:4000])


if __name__ == "__main__":
    main()
