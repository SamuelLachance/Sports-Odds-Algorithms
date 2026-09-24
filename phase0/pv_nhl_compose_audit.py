"""Player-value program, NHL -- COMPOSE walk-forward audit of the composition layer
(DEV ONLY). The four component files carry their own passed audits (creation
truncation, finishing permutation at 3 cutoffs, duels truncation at 2 cutoffs,
defense fit-cutoff asserts); this audits what COMPOSE adds on top: the walk-forward
position centring, the season weights fit on earlier seasons, the goalie value, the
lineup-deviation columns.

  A1 target permutation  every realised goal label of games dated >= cutoff is
                         permuted across those games (and 1 in 7 sign-flipped);
                         features of games dated < cutoff must be bitwise identical.
  A2 truncation          every panel row dated > cutoff is deleted; features of
                         games dated <= cutoff must be identical.
  A3 input perturbation  every component input (rates, logits, minutes) of player-
                         games dated >= cutoff is replaced by noise; features of
                         games dated < cutoff must be identical (same-day inputs
                         can only move their own game).
  A4 same-game facts     every realised same-game column (TOI, on-ice goals, shots,
                         goalie xGA / GA faced) is scrambled in EVERY game; every
                         feature must be identical (they are labels only).
Cutoffs 2013-02-15, 2015-01-15, 2017-02-01.  Output data/pv_nhl_compose_audit.json.

    python phase0/pv_nhl_compose_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pv_nhl_compose_core as C  # noqa: E402
import pv_nhl_compose_fit as FT  # noqa: E402

OUT = C.D("pv_nhl_compose_audit.json")
CUTS = ["2013-02-15", "2015-01-15", "2017-02-01"]
FEAT = ["lv_home", "lv_away", "gv_home", "gv_away", "lv_dev_home", "lv_dev_away",
        "gv_dev_home", "gv_dev_away"] + [f"wLU_{c}_{s}" for c in C.COMPS
                                         for s in ("home", "away")]
LABELS_TG = ["gf_s5", "gf_pp", "gf_sh", "gf_oth", "gf_en", "ga_s5", "ga_pp", "ga_sh",
             "ga_oth", "ga_en", "gf_noen", "ga_noen", "gd_noen", "gd_s5", "gd_st",
             "xgf_all", "xga_all"]
INPUTS_SK = ["ixg_ev_hat", "a1_ev_hat", "a2_ev_hat", "ixg_pp_hat", "a1_pp_hat", "a2_pp_hat",
             "fin_mu", "d_xga", "dd_g60", "exp_min_ev", "exp_min_pp", "exp_toi_s"]
SAME_GAME_SK = ["toi_s", "toi_ev", "toi_pp", "toi_sh", "on_gf_ev", "on_ga_ev", "on_xgf_ev",
                "on_xga_ev", "on_gf_pp", "on_ga_pp", "on_gf_sh", "on_ga_sh", "tm_gf_ev",
                "tm_ga_ev", "tm_xgf_ev", "tm_xga_ev", "tm_toi_ev", "tm_gf_pp", "tm_ga_pp",
                "tm_gf_sh", "tm_ga_sh", "g_ev", "a1_ev", "a2_ev", "ixg_ev", "g_pp", "a1_pp",
                "a2_pp"]
SAME_GAME_GK = ["toi_s", "fa_n", "fa_xg", "fa_xrc", "fa_g"]


def feats(sk, gk, tg):
    F = FT.build_features(sk, gk, tg)[0]
    return F.set_index("gid")[FEAT + ["date"]]


def compare(F0, F1, mask_ids):
    a = F0.loc[mask_ids, FEAT].to_numpy(float)
    b = F1.loc[mask_ids, FEAT].to_numpy(float)
    same = np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(
        np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0))
    return bool(same), float(np.nanmax(np.abs(a - b))) if a.size else 0.0


def main():
    t0 = time.time()
    rng = np.random.default_rng(11)
    sk, gk, tg = C.load_panel()
    F0 = feats(sk, gk, tg)
    res = {"n_games": int(len(F0)), "cutoffs": {}}
    for cut in CUTS:
        r = {}
        # A1 target permutation
        t1 = tg.copy()
        late = (t1.date >= cut).to_numpy()
        perm = rng.permutation(np.nonzero(late)[0])
        vals = t1.loc[late, LABELS_TG].to_numpy()
        t1.loc[late, LABELS_TG] = t1.loc[perm, LABELS_TG].to_numpy()
        flip = rng.random(late.sum()) < 1 / 7
        for c in ("gd_noen", "gd_s5", "gd_st"):
            v = t1.loc[late, c].to_numpy().copy()
            v[flip] = -v[flip]
            t1.loc[late, c] = v
        del vals
        F1 = feats(sk, gk, t1)
        pre = F0.index[F0.date < cut]
        post = F0.index[F0.date >= cut]
        same, mx = compare(F0, F1, pre)
        moved = int((np.abs(F0.loc[post, FEAT].to_numpy(float)
                            - F1.loc[post, FEAT].to_numpy(float)) > 1e-12).any(axis=1).sum())
        r["A1_target_permutation"] = {"n_pre": int(len(pre)), "identical_pre": same,
                                      "max_abs_diff_pre": mx, "n_post_games_moved": moved,
                                      "pass": same}
        # A2 truncation
        sk2 = sk[sk.date <= cut]
        gk2 = gk[gk.date <= cut]
        tg2 = tg[tg.date <= cut]
        F2 = feats(sk2.reset_index(drop=True), gk2.reset_index(drop=True),
                   tg2.reset_index(drop=True))
        keep = F0.index[F0.date <= cut]
        same2, mx2 = compare(F0, F2, keep)
        r["A2_truncation"] = {"n_kept": int(len(keep)), "identical": same2,
                              "max_abs_diff": mx2, "pass": same2}
        # A3 input perturbation (component inputs of player-games dated >= cut)
        sk3 = sk.copy()
        lt = (sk3.date >= cut).to_numpy()
        for c in INPUTS_SK:
            v = sk3[c].to_numpy(float).copy()
            v[lt] = rng.normal(np.nanmean(v), np.nanstd(v) + 1e-9, lt.sum())
            sk3[c] = v.astype(sk[c].dtype)   # keep dtype: float32 sums must round alike
        gk3 = gk.copy()
        lg = (gk3.date >= cut).to_numpy()
        gk3.loc[lg, "sav_mu"] = rng.normal(0, 0.1, lg.sum())
        F3 = feats(sk3, gk3, tg)
        same3, mx3 = compare(F0, F3, pre)
        moved3 = int((np.abs(F0.loc[post, FEAT].to_numpy(float)
                             - F3.loc[post, FEAT].to_numpy(float)) > 1e-12).any(axis=1).sum())
        r["A3_input_perturbation"] = {"n_pre": int(len(pre)), "identical_pre": same3,
                                      "max_abs_diff_pre": mx3, "n_post_games_moved": moved3,
                                      "pass": same3}
        res["cutoffs"][cut] = r
        print(cut, json.dumps(r), f"{time.time()-t0:.0f}s", flush=True)
    # A4 same-game facts scrambled everywhere
    sk4 = sk.copy()
    for c in SAME_GAME_SK:
        sk4[c] = rng.permutation(sk4[c].to_numpy())
    gk4 = gk.copy()
    for c in SAME_GAME_GK:
        gk4[c] = rng.permutation(gk4[c].to_numpy())
    F4 = feats(sk4, gk4, tg)
    same4, mx4 = compare(F0, F4, F0.index)
    res["A4_same_game_facts"] = {"n": int(len(F0)), "identical": same4, "max_abs_diff": mx4,
                                 "pass": same4}
    res["pass"] = bool(same4 and all(v[k]["pass"] for v in res["cutoffs"].values()
                                     for k in v))
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("A4", res["A4_same_game_facts"], "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()
