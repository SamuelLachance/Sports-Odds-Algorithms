"""Player-value program, NHL -- COMPOSE step 3: DEV-only game-level screen of the
composite personnel value beside the shipped blend.

NOT a TEST look. Pre-registration: data/pv_nhl_compose_screen_prereg.json (written
before any game log-loss was computed on these features; it pins the SHA-256 of the
feature and weight files, asserted below).

Harness: phase0/nhl_depth_eval.py Ctx (DEV regular season, LOSO over 2011-12..
2017-18, n = 7,929), stats helpers of phase0/bt_nhl_nhl_gmar.py (arm_stats: 4k seed-7
and 10k bootstraps, late-DEV split; nested: pair-loss nested grid selection).

Features (data/pv_nhl_compose_game_features.csv, walk-forward):
  lv   tonight's dressed-lineup value, goals / game = sum over dressed skaters of the
       composite goals/60 x expected minutes / 60 (season weights fit on earlier
       seasons only)
  gv   tonight's starting goalie's value, goals / game
  *_dev  tonight minus the mean of the team's previous 10 games

Arms (pre-registered):
  C0  shipped blend (S1 must reproduce 0.672398; S2 drop-xG cost +0.00303)
  C1  PRIMARY  blend + lv diff + gv diff
  C2  blend + (lv + gv) diff
  C3  blend + lv diff          C4  blend + gv diff
  C5  blend + lv, gv, lv_dev, gv_dev diffs
  C6  control: blend + incumbent RAPM lineup power P diff
  C7  inside: W/L Elo learns net of (lv + gv), gamma nested in {0,60,120,180,240}
Bar: gain >= +0.00100 and 10k-bootstrap 95% CI lower bound > 0 (C1 only).

    python phase0/pv_nhl_compose_screen.py
"""
from __future__ import annotations

import hashlib
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
import bt_nhl_nhl_gmar as GM  # noqa: E402  (arm_stats, nested; unmodified)
from nhl_features_eval import TEAM_FIX  # noqa: E402
from nhl_glicko2_eval import llv  # noqa: E402
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

PRE = "data/pv_nhl_compose_screen_prereg.json"
FEATS = "data/pv_nhl_compose_game_features.csv"
WEIGHTS = "data/pv_nhl_compose_weights.json"
OUT = "data/pv_nhl_compose_screen.json"
GAMMAS = [0.0, 60.0, 120.0, 180.0, 240.0]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def elo_inside(games, adj_h, adj_a, gamma, k=8, ha=30, reg=0.30):
    """nhl_depth_eval.run_elo_arr with an additive pre-game Elo term gamma * adj."""
    R = {}
    out = np.empty(len(games))
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - reg)
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        e = (rh + gamma * adj_h[i] + ha) - (ra + gamma * adj_a[i])
        p = 1.0 / (1.0 + 10 ** (-e / 400.0))
        out[i] = p
        R[g["home"]] += k * (g["y"] - p)
        R[g["away"]] += k * ((1 - g["y"]) - (1 - p))
    return out


def main():
    t0 = time.time()
    pre = json.load(open(PRE))
    assert sha(FEATS) == pre["feature_file_sha256"], "feature file changed since prereg"
    assert sha(WEIGHTS) == pre["weights_file_sha256"], "weights changed since prereg"
    F = pd.read_csv(FEATS)
    assert int(F.gid.max()) < DEV_MAX_GID
    F = F.set_index("gid")
    ctx = Hd.Ctx()
    games_all = ctx.games
    gid_all = np.array([g["game_id"] for g in games_all])
    y = ctx.yv()
    folds = ctx.folds
    seas = folds.seasons
    gid = gid_all[ctx.mask]
    assert gid.max() < DEV_MAX_GID
    cov = float(np.isin(gid, F.index.to_numpy()).mean())
    fh = np.array([TEAM_FIX.get(t, t) for t in F.reindex(gid).home.astype(str)])
    code_ok = float(np.mean(fh == np.array([g["home"] for g, m in zip(games_all, ctx.mask)
                                            if m])))

    def diff(col):
        return (F.reindex(gid)[col + "_home"] - F.reindex(gid)[col + "_away"]).to_numpy(float)

    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    s1 = folds.pooled(y, p0)
    s2 = folds.pooled(y, folds.loso_pred(X0[:, :-1], y)) - s1
    res = {"protocol": {"dev_only": True, "n": int(len(y)),
                        "seasons": sorted(int(s) for s in set(seas)),
                        "test_seasons_touched": 0, "market_inputs": 0,
                        "prereg": PRE, "prereg_hashes_verified": True},
           "sanity": {"S1_shipped_dev_loso_ll": round(s1, 6), "S1_target": 0.672398,
                      "S1_pass": bool(abs(s1 - 0.672398) < 2e-6 and len(y) == 7929),
                      "S2_drop_xg_cost": round(s2, 5), "S2_target": 0.00303,
                      "S2_pass": bool(abs(s2 - 0.00303) < 0.0002),
                      "S3_feature_coverage": cov, "S3_home_code_match": code_ok,
                      "S3_pass": bool(cov == 1.0 and code_ok == 1.0)}}
    print(json.dumps(res["sanity"]), flush=True)

    d_lv, d_gv = diff("lv"), diff("gv")
    d_ld, d_gd = diff("lv_dev"), diff("gv_dev")
    rp = pd.read_csv("data/bt_nhl_rapmel_feats.csv").set_index("game_id")
    d_P = (rp.reindex(gid).P_home - rp.reindex(gid).P_away).to_numpy(float)
    miss = {k: int(np.isnan(v).sum()) for k, v in
            {"lv": d_lv, "gv": d_gv, "lv_dev": d_ld, "gv_dev": d_gd, "P": d_P}.items()}
    d_ld, d_gd = np.nan_to_num(d_ld), np.nan_to_num(d_gd)
    assert miss["lv"] == 0 and miss["gv"] == 0 and miss["P"] == 0
    res["feature_missing"] = miss
    res["feature_sd"] = {k: round(float(np.std(v)), 4) for k, v in
                         {"lv": d_lv, "gv": d_gv, "lv_gv": d_lv + d_gv, "lv_dev": d_ld,
                          "gv_dev": d_gd, "P": d_P}.items()}
    res["corr_lv_gv_with_blend_cols"] = {
        "lv_vs_elo_logit": round(float(np.corrcoef(d_lv, X0[:, 1])[0, 1]), 3),
        "lv_vs_xg_diff": round(float(np.corrcoef(d_lv, X0[:, -1])[0, 1]), 3),
        "gv_vs_elo_logit": round(float(np.corrcoef(d_gv, X0[:, 1])[0, 1]), 3),
        "P_vs_xg_diff": round(float(np.corrcoef(d_P, X0[:, -1])[0, 1]), 3)}

    arms = {"C1 PRIMARY lv+gv (2 cols)": [d_lv, d_gv],
            "C2 lv+gv (1 col)": [d_lv + d_gv],
            "C3 lv": [d_lv], "C4 gv": [d_gv],
            "C5 lv,gv,lv_dev,gv_dev": [d_lv, d_gv, d_ld, d_gd],
            "C6 control RAPM P": [d_P]}
    for name, cols in arms.items():
        X = np.column_stack([X0] + cols)
        p = folds.loso_pred(X, y)
        st = GM.arm_stats(y, folds, p0, p, seas)
        w_all = Hd.fit_logit(X, y, Hd.BLEND_C)
        st["coef_all_dev_descriptive"] = [round(float(c), 4) for c in w_all[-len(cols):]]
        st["per_season_gain"] = {str(s): round(float(
            (llv(y[folds.idx[s]], p0[folds.idx[s]]) - llv(y[folds.idx[s]], p[folds.idx[s]])
             ).mean()), 5) for s in folds.list}
        res[name] = st
        print(name, json.dumps({k: st[k] for k in ("gain", "boot_ci_10k", "sig",
                                                   "n_pos_folds", "dev_ll")}),
              json.dumps(st["late_dev_2015_18"]), flush=True)

    # C7: personnel value INSIDE the W/L Elo, gamma nested
    adj = (F.reindex(gid_all).lv_home + F.reindex(gid_all).gv_home).to_numpy(float)
    adj_a = (F.reindex(gid_all).lv_away + F.reindex(gid_all).gv_away).to_numpy(float)
    nmiss = int(np.isnan(adj).sum() + np.isnan(adj_a).sum())
    adj, adj_a = np.nan_to_num(adj), np.nan_to_num(adj_a)
    cells = {}
    base = ctx.base_cols()
    xgc = ctx.xg(*Hd.SHIPPED_XG)
    for gm in GAMMAS:
        pe = np.clip(elo_inside(games_all, adj, adj_a, gm), 1e-9, 1 - 1e-9)
        el = np.log(pe / (1 - pe))[ctx.mask]
        cells[gm] = Hd.design([el] + base + [xgc])
    assert np.allclose(cells[0.0], X0), "gamma=0 cell must equal the shipped blend"
    pn, picks, loso, cell_ll, best = GM.nested(cells, y, folds)
    st = GM.arm_stats(y, folds, p0, pn, seas)
    st["picks"] = {str(k): v for k, v in picks.items()}
    st["cell_loso_ll"] = {str(k): round(v, 6) for k, v in cell_ll.items()}
    st["best_single_cell_OPTIMISTIC"] = {"gamma": best,
                                         "gain": round(cell_ll[0.0] - cell_ll[best], 5)}
    st["n_missing_adj_all_games_incl_warmup"] = nmiss
    res["C7 inside Elo (gamma nested)"] = st
    print("C7", json.dumps({k: st[k] for k in ("gain", "boot_ci_10k", "sig", "picks")}),
          flush=True)

    c1 = res["C1 PRIMARY lv+gv (2 cols)"]
    sane = res["sanity"]["S1_pass"] and res["sanity"]["S2_pass"] and res["sanity"]["S3_pass"]
    clears = bool(sane and c1["gain_exact"] >= 0.00100 and c1["boot_ci_10k"][0] > 0)
    res["verdict"] = {"primary": "C1", "gain": c1["gain"], "boot_ci_10k": c1["boot_ci_10k"],
                      "sanity_pass": bool(sane), "bar": "+0.00100, CI lo > 0",
                      "clears_bar": clears,
                      "note": "walk-forward + leak audit: data/pv_nhl_compose_audit.json"}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
