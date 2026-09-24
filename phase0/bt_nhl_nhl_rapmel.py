"""NHL breakthrough DEV screen nhl_rapmel -- walk-forward stint-RAPM lineup power
plus goalie level, INSIDE both team ratings (the full MLB transplant).

Every dressed skater is valued from his own process stat (5v5 on-ice net xG/60 from
stint RAPM refit WALK-FORWARD: phase0/bt_nhl_rapmel_stints.py -> _fit.py), times
his own trailing 5v5 minutes; tonight's starter is valued by GSAx level
(phase0/bt_nhl_gmar_gl.py, the shared nhl_gmar module). Both terms are read INSIDE
the ratings (W/L Elo: gamma*(G+P); xG-Elo: beta*P) so the team ratings learn net of
personnel (phase0/bt_nhl_rapmel.py).

Binding protocol: documents/breakthrough_program_prereg_2026_09_24.md
  * market-blind: odds enter ONLY the evaluation diagnostic (model minus de-vigged
    DEV close), never a feature or a fit;
  * DEV only: ratings run on nhl_depth_eval.dev_games() (season <= 2017-18); every
    file is filtered to game_id < 2018000000 on read; every scored mask passes
    nhl_depth_eval.assert_dev_only; nothing from 2018-19+ is loaded or scored;
  * nhl_rapm2.main()/build_stints() are never called; all-season rating files
    (nhl_rapm2_ratings.json etc.), win_goalie, max-shots starters and odds are never
    read by a feature path.

Harness: nhl_depth_eval Ctx / Folds / fit_logit; LOSO over 2011-12..2017-18
(n = 7,929, ctx.mask kept exactly). Nested grids via bt_nhl_nhl_gmar.nested (the
axis_a pair_loss pattern); bootstrap via bt_nhl_nhl_gmar.arm_stats (10,000
resamples, seed 7, plus the harness 4,000/seed-7 CI).

Arms
  B0 shipped blend (S1 reproduction)
  B1 GL only (gamma nested)                               == nhl_gmar A1 (S4a)
  B2 P only inside both ratings, G = 0 ((gamma, beta) nested)
  B3 GL + P inside both ratings ((gamma, beta) nested)    <- PRIMARY, only bar arm
  B4 GL inside (gamma nested) + dP beside (BESIDE control)
  B5 B3 without the Jan-1 refits (refit-cadence diagnostic)

Outputs: data/bt_nhl_nhl_rapmel.json (mirrored to data/bt_nhl_rapmel.json) and
data/bt_nhl_rapmel_feats.csv (per-game DEV features).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import DEV_END, DEV_WARM_BEFORE, TEST_START, llv  # noqa: E402
import bt_nhl_gmar_gl as GL  # noqa: E402
import bt_nhl_nhl_gmar as GM  # noqa: E402
import bt_nhl_rapmel as R  # noqa: E402

OUT = "data/bt_nhl_nhl_rapmel.json"
OUT_MIRROR = "data/bt_nhl_rapmel.json"
OUT_FEATS = "data/bt_nhl_rapmel_feats.csv"
MAX_GID = 2018000000
assert DEV_END == 20172018 and TEST_START == 20182019 and DEV_WARM_BEFORE == 20112012

GAMMAS = [0.0, 60.0, 120.0, 180.0]      # Elo per goal/game (pre-declared)
BETAS = [0.0, 0.5, 1.0]                  # xG-Elo units per goal/game of P (pre-declared)
W2_CUTS = ("2012-02-01", "2015-01-15", "2017-12-01")
GMAR_PRIMARY_GAIN = None                 # read from data/bt_nhl_nhl_gmar.json


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ------------------------------------------------------------ feature set ---
def build_features(games, starters, gg, dressed, toi5, seed, values, values_pre):
    """Every personnel feature and rating column used by any arm (full length)."""
    F = {}
    Pf = R.build_P(games, dressed, toi5, values, seed)
    Pp = R.build_P(games, dressed, toi5, values_pre, seed)
    F["P_h"], F["P_a"], F["dP_h"], F["dP_a"] = Pf["P_h"], Pf["P_a"], Pf["dP_h"], Pf["dP_a"]
    F["Ppre_h"], F["Ppre_a"] = Pp["P_h"], Pp["P_a"]
    F["pfb_h"], F["pfb_a"] = Pf["fb_h"], Pf["fb_a"]
    F["n_unvalued"], F["n_prior_minutes"] = Pf["n_unvalued"], Pf["n_prior_minutes"]
    F["_Pmeta"] = {k: Pf[k] for k in ("first_use", "deb_prior_final", "deb_games_final")}
    F["_Pmeta_pre"] = {k: Pp[k] for k in ("first_use",)}
    zero = np.zeros(len(games))
    # W/L Elo cells
    for gm in GAMMAS:
        r = GL.gl_elo(games, starters, gg, gm, mode="new")              # B1/B4 (G only)
        F[f"elo_G_{gm:g}"] = r["p"]
        if gm == 0.0:
            F["G_h"], F["G_a"] = r["G_h"], r["G_a"]
            F["gk_h"], F["gk_a"] = r["gk_h"], r["gk_a"]
            F["gfb_h"], F["gfb_a"] = r["fb_h"], r["fb_a"]
        F[f"elo_P_{gm:g}"] = R.elo_cell(GL, games, starters, gg, 0.0, gm,
                                        F["P_h"], F["P_a"])["p"]          # B2
        F[f"elo_GP_{gm:g}"] = R.elo_cell(GL, games, starters, gg, gm, gm,
                                         F["P_h"], F["P_a"])["p"]         # B3
        F[f"elo_GPpre_{gm:g}"] = R.elo_cell(GL, games, starters, gg, gm, gm,
                                            F["Ppre_h"], F["Ppre_a"])["p"]  # B5
    # xG-Elo cells
    for b in BETAS:
        F[f"xg_P_{b:g}"] = R.xg_beta(games, F["P_h"], F["P_a"], b)
        F[f"xg_Ppre_{b:g}"] = R.xg_beta(games, F["Ppre_h"], F["Ppre_a"], b)
    del zero
    return F


# ------------------------------------------------------------------ W2 ------
def perturb(games, starters, gg, dressed, toi5, cutoff, seed):
    """bt_nhl_nhl_gmar.perturb (same permutation stream) + the 5v5 TOI rows."""
    g2, st2, gg2, dr2, n_after, changed = GM.perturb(games, starters, gg, dressed,
                                                     cutoff, seed)
    rng = np.random.default_rng(seed)
    after = [i for i, g in enumerate(games) if g["date"] >= cutoff]
    perm = rng.permutation(after)                  # identical draw to GM.perturb
    t2 = dict(toi5)
    txg = Hd.load_team_xg_dev(games)
    tx2 = dict(txg)
    for i, j in zip(after, perm):
        a, b = games[i]["game_id"], games[j]["game_id"]
        t2[a] = toi5.get(b, {})
        # team xG rows (the xG-Elo target) moved home->home, away->away
        src = txg.get(b, {})
        gi, gj = games[i], games[j]
        new = {}
        if gj["home"] in src:
            new[gi["home"]] = src[gj["home"]]
        if gj["away"] in src:
            new[gi["away"]] = src[gj["away"]]
        tx2[a] = new
    # consistency: GM.perturb used the same permutation
    for i, j in zip(after[:50], perm[:50]):
        assert g2[i]["y"] == games[j]["y"]
    return g2, st2, gg2, dr2, t2, tx2, n_after, changed


# --------------------------------------------------------------- diagnostics --
def top10_by_cutoff(values):
    names = json.load(open("data/nhl_player_names.json", encoding="utf-8"))
    allv = pd.read_csv(f"{R.PFX}values.csv")
    out = {}
    for cut, df in allv.groupby("cutoff_date"):
        t = df.sort_values("v", ascending=False).head(10)
        out[cut] = [{"pid": int(r.pid), "name": names.get(str(int(r.pid)), {}).get("name"),
                     "pos": r.pos, "v": round(float(r.v), 4),
                     "net_raw": round(float(r.net_raw), 4),
                     "toi5_min_window": round(float(r.T) / 60.0, 0)} for r in t.itertuples()]
    return out


def main():
    t0 = time.time()
    ctx = Hd.Ctx()
    games = ctx.games
    assert max(g["season"] for g in games) <= DEV_END
    y = ctx.yv()
    folds = ctx.folds
    mask = ctx.mask
    seas = ctx.seasons[mask]
    Hd.assert_dev_only(seas)
    res = {"candidate": "nhl_rapmel",
           "title": "Walk-forward stint-RAPM lineup power plus goalie level, INSIDE both "
                    "team ratings (the full MLB transplant)",
           "protocol": {"dev_seasons_scored": sorted(set(seas.tolist())),
                        "n_dev": int(mask.sum()),
                        "eval": "LOSO over DEV seasons, nested (gamma,beta) grids",
                        "market_blind": True, "test_seasons_touched": 0,
                        "pre_declared_variants": ["B3 primary", "B4 beside control",
                                                  "B5 no Jan-1 refits"],
                        "bar": "nested B3 gain >= +0.00100 AND 95% CI lo > 0 AND all "
                               "sanity + walk-forward checks pass"}}

    starters, gg, dressed = GM.load_inputs(games)      # all filtered gid < MAX_GID
    ids = set(g["game_id"] for g in games)
    assert max(ids) < MAX_GID
    toi5 = R.load_toi5(ids)
    assert max(toi5) < MAX_GID
    seed = R.seed_positions()
    values = R.load_values()
    values_pre = R.load_values(pre_only=True)
    print(f"DEV games {len(games)} masked {int(mask.sum())}; {len(values)} value fits "
          f"({len(values_pre)} pre-season); loaded {time.time()-t0:.1f}s", flush=True)

    # ============================ HARNESS SANITY ============================
    san = {}
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    l0 = llv(y, p0)
    s1 = float(l0.mean())
    san["S1"] = {"shipped_dev_loso_ll": round(s1, 6), "n": int(len(y)), "target": 0.672398,
                 "pass": bool(abs(s1 - 0.672398) < 2e-6 and len(y) == 7929)}
    pnx = folds.loso_pred(X0[:, :5], y)
    s2 = float(llv(y, pnx).mean() - s1)
    san["S2"] = {"drop_xg_cost": round(s2, 5), "target": 0.00303, "tol": 0.0002,
                 "recorded_0811_sweep": 0.00304, "ledger_row5_TEST_recorded": 0.00316,
                 "pass": bool(abs(s2 - 0.00303) <= 0.0002)}
    print(f"S1 shipped DEV LOSO {s1:.6f} n={len(y)}   S2 drop-xG cost {s2:+.5f}", flush=True)

    F = build_features(games, starters, gg, dressed, toi5, seed, values, values_pre)
    print(f"features built {time.time()-t0:.1f}s", flush=True)

    # S3: the (0,0) cell reproduces the shipped ratings and blend
    ref_e = Hd.run_elo_arr(games, *Hd.SHIPPED_ELO)
    ref_x = Hd.run_xg_arr(games, *Hd.SHIPPED_XG)
    d_e = float(np.abs(F["elo_GP_0"] - ref_e).max())
    xa, xb = F["xg_P_0"], ref_x
    same_nan = bool(np.array_equal(np.isnan(xa), np.isnan(xb)))
    d_x = float(np.nanmax(np.abs(xa - xb)))
    rest = [X0[:, 2], X0[:, 3], X0[:, 4]]
    X00 = Hd.design([logit(F["elo_GP_0"])[mask]] + rest + [np.nan_to_num(F["xg_P_0"])[mask]])
    p00 = folds.loso_pred(X00, y)
    d_b = float(np.abs(p00 - p0).max())
    san["S3"] = {"elo_cell00_vs_run_elo_arr_maxabs": d_e, "xg_cell00_vs_run_xg_arr_maxabs": d_x,
                 "xg_nan_pattern_identical": same_nan, "blend_pred_maxabs": d_b,
                 "pass": bool(d_e < 1e-12 and d_x < 1e-12 and same_nan and d_b < 1e-12)}

    # S4a: B1 == nhl_gmar A1 (computed below after the arm runs), G arrays vs gmar cache
    gfe = pd.read_csv("data/bt_nhl_gmar_feats.csv")
    gfe = gfe[gfe.game_id < MAX_GID].set_index("game_id").loc[[g["game_id"] for g in games]]
    dG = float(max(np.abs(gfe.G_home.to_numpy() - F["G_h"]).max(),
                   np.abs(gfe.G_away.to_numpy() - F["G_a"]).max()))
    gmar = json.load(open("data/bt_nhl_nhl_gmar.json", encoding="utf-8"))

    # S4b/S4c/S4d: value build
    fits = json.load(open(f"{R.PFX}fits.json"))
    s4 = fits["s4"]
    stsum = json.load(open(f"{R.PFX}stints_summary.json"))
    posd = json.load(open(f"{R.PFX}posdiag.json"))
    san["S4b"] = {"heldout_stint_R2_2014_17_odd_even": round(s4["S4b_heldout_R2"], 5),
                  "halves": [round(v, 5) for v in s4["S4b_halves"]],
                  "expected_order": [0.010, 0.014], "stop_if_leq": 0.0,
                  "decomposition_same_split": {"context_only": 0.00342, "players_only": 0.00095,
                                               "team_season_indicators_only": 0.00053,
                                               "players_plus_context": 0.00434,
                                               "note": "odd-gid fit, even-gid scored half; "
                                                       "exact normal-equation solve agrees with "
                                                       "sparse_cg to 6e-5 in coefficients"},
                  "stop_rule_triggered": bool(s4["S4b_heldout_R2"] <= 0),
                  "within_expected_order": bool(0.008 <= s4["S4b_heldout_R2"] <= 0.02),
                  "pass_rule": "conservative, fixed before any arm was scored: positive AND "
                               "within the expected order (0.008..0.02)",
                  "pass": bool(s4["S4b_heldout_R2"] > 0
                               and 0.008 <= s4["S4b_heldout_R2"] <= 0.02)}
    san["S4c"] = {"split_half_r_net_ge_1000min": round(s4["S4c_split_half_r_net_1000min"], 4),
                  "n_players": s4["S4c_n_players"], "r_F": round(s4["S4c_r_F"], 4),
                  "r_D": round(s4["S4c_r_D"], 4), "expected": [0.3, 0.5],
                  "pass": bool(0.3 <= s4["S4c_split_half_r_net_1000min"] <= 0.5)}
    san["S4d"] = {"kept_5v5_stint_xg_over_all_strict5v5_xg": round(stsum["S4d_kept_over_all_strict5v5"], 4),
                  "strength_leak_dropped_share": round(stsum["strength_leak_dropped_share"], 4),
                  "pass": bool(stsum["S4d_kept_over_all_strict5v5"] >= 0.9)}
    san["positions"] = {"why_not_infer_positions": posd,
                        "per_fit": {k: v["positions"] for k, v in fits["fits"].items()},
                        "pass": all(v["positions"]["share_sides_3F2D"] >= 0.97
                                    for v in fits["fits"].values())}

    # ============================ ARMS ============================
    arms = {}
    arm_hash = {}
    mask_hash = hashlib.sha1(mask.tobytes()).hexdigest()
    y_hash = hashlib.sha1(y.tobytes()).hexdigest()
    base_ship = [X0[:, 2], X0[:, 3], X0[:, 4], X0[:, 5]]
    dPd = (F["dP_h"] - F["dP_a"])[mask]

    def rec(name, p, extra=None):
        st = GM.arm_stats(y, folds, p0, p, seas)
        if extra:
            st.update(extra)
        arms[name] = st
        arm_hash[name] = mask_hash
        print(f"  {name:<46} LL {st['dev_ll']:.6f} gain {st['gain_exact']:+.5f} "
              f"CI10k [{st['boot_ci_10k'][0]:+.5f},{st['boot_ci_10k'][1]:+.5f}] "
              f"{st['n_pos_folds']}/{st['n_folds']} folds  late "
              f"{st['late_dev_2015_18']['gain']:+.5f}", flush=True)
        return st

    def pj(picks):
        return {str(s): (list(c) if isinstance(c, tuple) else c) for s, c in picks.items()}

    def cells_json(ll):
        return {(f"{k[0]:g},{k[1]:g}" if isinstance(k, tuple) else f"{k:g}"): round(v, 6)
                for k, v in ll.items()}

    def nested_arm(name, cells, extra=None):
        p, pk, _loso, ll, best = GM.nested(cells, y, folds)
        ex = {"picks_c_star_per_season": pj(pk), "cell_loso_ll": cells_json(ll),
              "best_single_cell_OPTIMISTIC": {"cell": (list(best) if isinstance(best, tuple)
                                                        else best),
                                              "gain": round(s1 - ll[best], 5)}}
        if extra:
            ex.update(extra)
        return rec(name, p, ex), p

    rec("B0 shipped blend (reproduction)", p0)
    cells1 = {gm: Hd.design([logit(F[f"elo_G_{gm:g}"])[mask]] + base_ship) for gm in GAMMAS}
    stB1, pB1 = nested_arm("B1 GL only (gamma nested)", cells1)

    def gp_cells(elo_key, xg_key):
        return {(gm, b): Hd.design([logit(F[f"{elo_key}_{gm:g}"])[mask]] + rest
                                   + [np.nan_to_num(F[f"{xg_key}_{b:g}"])[mask]])
                for gm in GAMMAS for b in BETAS}

    stB2, pB2 = nested_arm("B2 P only inside both ratings", gp_cells("elo_P", "xg_P"))
    stB3, pB3 = nested_arm("B3 GL + P inside both ratings [PRIMARY]",
                           gp_cells("elo_GP", "xg_P"))
    cells4 = {gm: np.column_stack([Hd.design([logit(F[f"elo_G_{gm:g}"])[mask]] + base_ship),
                                   dPd]) for gm in GAMMAS}
    wB4 = Hd.fit_logit(cells4[0.0], y)
    stB4, pB4 = nested_arm("B4 GL inside + dP beside (BESIDE control)", cells4,
                           {"coef_dP_diff_all_dev_descriptive_gamma0": round(float(wB4[-1]), 5)})
    stB5, pB5 = nested_arm("B5 B3 without Jan-1 refits", gp_cells("elo_GPpre", "xg_Ppre"))

    a1 = gmar["arms"]["A1 GL new col1 (gamma nested)"]
    s4a_gain = abs(stB1["gain_exact"] - a1["gain_exact"])
    san["S4a"] = {"B1_gain_exact": stB1["gain_exact"], "gmar_A1_gain_exact": a1["gain_exact"],
                  "abs_diff_gain": s4a_gain, "B1_dev_ll": stB1["dev_ll"],
                  "gmar_A1_dev_ll": a1["dev_ll"],
                  "picks_equal": pj({int(k): v for k, v in
                                     stB1["picks_c_star_per_season"].items()})
                  == a1["picks"],
                  "G_arrays_vs_gmar_feats_maxabs": dG,
                  "note": "gmar stores no per-game A1 predictions; B1 is built by the same "
                          "gl_elo + GM.nested code on the same inputs, so equality of the "
                          "exact pooled gain (a mean over 7,929 per-game losses), picks and "
                          "the per-game G arrays is the per-game check available",
                  "pass": bool(s4a_gain < 1e-12 and dG < 1e-12
                               and stB1["dev_ll"] == a1["dev_ll"])}
    for k, v in san.items():
        vv = {kk: vv_ for kk, vv_ in v.items() if kk not in ("why_not_infer_positions", "per_fit")}
        print(f"  {k}: pass={v['pass']}  {vv}", flush=True)
    res["harness_sanity"] = san

    # ============================ WALK-FORWARD ============================
    wf = {"W1": {"statement": "one ordered pass per run; per-player m5 / appearance / "
                              "debutant-pool state, per-goalie N/X (gl_elo), per-team R, X, "
                              "10-lineup window are read for game i before any of game i's "
                              "result, TOI, GA/xGA is applied; runtime write-index guards "
                              "(assert last write < i) in bt_nhl_rapmel.build_P and gl_elo",
                 "runtime_guards_raised": 0, "pass": True}}
    keys_chk = (["P_h", "P_a", "dP_h", "dP_a", "Ppre_h", "Ppre_a", "G_h", "G_a"]
                + [f"elo_G_{g:g}" for g in GAMMAS] + [f"elo_P_{g:g}" for g in GAMMAS]
                + [f"elo_GP_{g:g}" for g in GAMMAS] + [f"elo_GPpre_{g:g}" for g in GAMMAS]
                + [f"xg_P_{b:g}" for b in BETAS] + [f"xg_Ppre_{b:g}" for b in BETAS])
    w2 = {}
    for ci_, cut in enumerate(W2_CUTS):
        g2, st2, gg2, dr2, t2, tx2, n_after, n_changed = perturb(
            games, starters, gg, dressed, toi5, cut, seed=100 + ci_)
        v2 = R.load_values(drop=cut)
        v2p = R.load_values(pre_only=True, drop=cut)
        n_refit = sum(1 for vs in v2 if vs.cutoff >= cut)
        txg_keep = Hd._TXG
        Hd._TXG = tx2                                   # perturbed team-xG rows
        F2 = build_features(g2, st2, gg2, dr2, t2, seed, v2, v2p)
        Hd._TXG = txg_keep
        Fa = {k: F[k] for k in keys_chk}
        Fb = {k: F2[k] for k in keys_chk}
        r = GM.feats_equal_before(Fa, Fb, games, cut)
        r.update({"n_games_permuted_after": n_after, "n_moved": n_changed,
                   "value_files_refit_on_dropped_stints": n_refit})
        w2[cut] = r
        print(f"  W2 {cut}: pass={r['pass']} pre={r['n_pre_games']} checked="
              f"{r['n_features_checked']} changed_after={r['n_features_changed_after_cutoff']}"
              f" refits={n_refit} mismatched={r['mismatched_features']}", flush=True)
    wf["W2"] = {"cutoffs": w2, "pass": all(v["pass"] for v in w2.values()),
                "note": "y, goalie GA/xGA rows, dressed lists, starters and per-game 5v5 TOI "
                        "permuted among games on/after each cutoff; every stint on/after the "
                        "cutoff dropped and every value file with cutoff >= the date refit "
                        "(bt_nhl_rapmel_fit.py fit <name> --drop); P, dP, G and every Elo / "
                        "xG-Elo cell of every arm rebuilt and compared bitwise before it"}
    fitm = fits["fits"]
    fu = F["_Pmeta"]["first_use"]
    w3_rows = {k: {"cutoff": v["cutoff"], "max_train_date": v["max_train_date"],
                   "first_use": fu.get(k)} for k, v in fitm.items()}
    w3_ok = all(v["max_train_date"] <= int(v["cutoff"].replace("-", ""))
                and (w3_rows[k]["first_use"] is None or w3_rows[k]["first_use"] > v["cutoff"])
                for k, v in fitm.items())
    w3_ok &= all(json.load(open(f"{R.PFX}fitm_{k}.json"))["max_train_date"]
                 <= int(v["cutoff"].replace("-", "")) for k, v in fitm.items())
    wf["W3"] = {"ridge_fits": w3_rows,
                "v_repl": "computed inside each fit from its own window (dates <= cutoff)",
                "debutant_m5_prior": "committed only at date change; asserted "
                                     "last_committed_date < today",
                "gl_m_E_xbar": "bt_nhl_gmar_gl (completed seasons / strictly earlier dates)",
                "pass": bool(w3_ok)}
    pers = {k: F[k][mask] for k in ("P_h", "P_a", "dP_h", "dP_a", "Ppre_h", "Ppre_a",
                                    "G_h", "G_a")}
    nan_any = {k: int(np.isnan(v).sum()) for k, v in pers.items() if np.isnan(v).any()}
    wf["W4"] = {"eval_mask_sha1": mask_hash, "y_sha1": y_hash,
                "gmar_mask_sha1": gmar["walk_forward"]["W4"]["eval_mask_sha1"],
                "identical_across_arms": len(set(arm_hash.values())) == 1,
                "identical_to_gmar_mask": mask_hash == gmar["walk_forward"]["W4"]["eval_mask_sha1"],
                "n_masked": int(mask.sum()),
                "nan_personnel_terms_on_masked_games": nan_any,
                "dressed_fallback_sides_masked": int((F["pfb_h"] | F["pfb_a"])[mask].sum()),
                "pass": bool(len(set(arm_hash.values())) == 1 and not nan_any
                             and mask_hash == gmar["walk_forward"]["W4"]["eval_mask_sha1"])}
    wf["W5"] = {"statement": "same-game inputs are only tonight's starting goalie identity "
                             "(first-shot goalie) and tonight's dressed-skater membership "
                             "(announced lineup; when a side's dressed list is missing the "
                             "team's previous lineup is used). Tonight's 5v5 TOI, stints, xG, "
                             "GA and result enter only post-prediction updates or later ridge "
                             "fits whose cutoff precedes use. win_goalie, max-shots starters, "
                             "nhl_rapm*/shift rating files and odds never enter a feature.",
                "pass": True}
    res["walk_forward"] = wf
    res["arms"] = arms

    # ============================ value diagnostics ============================
    Ph, Pa = F["P_h"][mask], F["P_a"][mask]
    xg_ship = np.nan_to_num(ref_x)[mask]
    af = pd.read_csv("data/bt_nhl_anatomy_feats.csv")
    af = af[af.game_id < MAX_GID].set_index("game_id")
    afm = af.loc[[g["game_id"] for g, k in zip(games, mask) if k]]
    assert afm.season.max() <= DEV_END
    absd = (afm.abs_toi_home - afm.abs_toi_away).to_numpy()
    okd = ~np.isnan(absd)
    top10 = top10_by_cutoff(values)
    gm_primary = gmar["verdict"]["nested_gain"]
    res["value_diagnostics"] = {
        "S4b": san["S4b"], "S4c": san["S4c"],
        "corr_Pdiff_vs_shipped_xgelo_diff": round(float(np.corrcoef(Ph - Pa, xg_ship)[0, 1]), 4),
        "corr_expected": [0.3, 0.6],
        "corr_dPdiff_vs_anatomy_abs_toi_diff": round(float(np.corrcoef(dPd[okd], absd[okd])[0, 1]), 4),
        "n_for_abs_toi_corr": int(okd.sum()),
        "sd_P_side_goals": round(float(np.concatenate([Ph, Pa]).std()), 4),
        "sd_P_diff_goals": round(float((Ph - Pa).std()), 4),
        "sd_dP_diff_goals": round(float(dPd.std()), 4),
        "sd_G_diff_goals": round(float((F["G_h"] - F["G_a"])[mask].std()), 4),
        "corr_Pdiff_Gdiff": round(float(np.corrcoef(Ph - Pa, (F["G_h"] - F["G_a"])[mask])[0, 1]), 4),
        "mean_unvalued_skaters_per_game": round(float(F["n_unvalued"][mask].mean()), 3),
        "mean_prior_minutes_skaters_per_game": round(float(F["n_prior_minutes"][mask].mean()), 3),
        "debutant_m5_prior_final": F["_Pmeta"]["deb_prior_final"],
        "debutant_games_final": F["_Pmeta"]["deb_games_final"],
        "v_repl_by_fit": {k: v["v_repl"] for k, v in fitm.items()},
        "first_use_by_fit": fu,
        "P_in_2010_11": "0 for every game (no fit precedes it; never scored)",
        "top10_v_by_cutoff": top10,
    }
    print(json.dumps({k: v for k, v in res["value_diagnostics"].items()
                      if k not in ("top10_v_by_cutoff", "S4b", "S4c", "v_repl_by_fit",
                                   "first_use_by_fit")}), flush=True)

    # ======================= evaluation-only market diagnostic =======================
    from bt_nhl_anatomy import join_odds   # DEV-only join (date < 2018-07-01)
    Gm = [g for g, k in zip(games, mask) if k]
    pc, _po, _ = join_odds(Gm, y)
    ok = ~np.isnan(pc)
    lc = llv(y[ok], pc[ok])
    nm = ((afm.g_notmodal_home == 1) | (afm.g_notmodal_away == 1)).to_numpy()[ok]
    lk = (afm.abs_toi_home.notna() & afm.abs_toi_away.notna()).to_numpy()
    t4 = (np.nan_to_num(afm.abs_top4_home.to_numpy()) + np.nan_to_num(afm.abs_top4_away.to_numpy()))
    top4 = (lk & (t4 >= 1))[ok]
    early = ((afm.gp_home <= 5) | (afm.gp_away <= 5)).to_numpy()[ok]
    mkt = {"n_joined": int(ok.sum()), "close_ll": round(float(lc.mean()), 6),
           "note": "EVALUATION ONLY: multiplicative de-vig of the SBR close on DEV games; "
                   "odds never enter a feature or a fit"}
    for nm_, p_ in (("B0", p0), ("B1", pB1), ("B2", pB2), ("B3", pB3), ("B4", pB4),
                    ("B5", pB5)):
        dd = llv(y[ok], p_[ok]) - lc
        mkt[nm_] = {"model_minus_close_all": round(float(dd.mean()), 5),
                    "non1_goalie": {"n": int(nm.sum()), "model_minus_close": round(float(dd[nm].mean()), 5)},
                    "top4_absent": {"n": int(top4.sum()), "model_minus_close": round(float(dd[top4].mean()), 5)},
                    "team_games_0_5": {"n": int(early.sum()), "model_minus_close": round(float(dd[early].mean()), 5)}}
    g0 = mkt["B0"]["model_minus_close_all"]
    mkt["B3_gap_closed_share"] = round((g0 - mkt["B3"]["model_minus_close_all"]) / g0, 4)
    res["market_eval_diagnostic"] = mkt
    print(json.dumps(mkt), flush=True)

    # ============================ VERDICT ============================
    sanity_pass = all(v["pass"] for v in san.values())
    wf_pass = all(v["pass"] for v in wf.values())
    gain = stB3["gain_exact"]
    lo = stB3["boot_ci_10k"][0]
    clears = bool(gain >= 0.00100 and lo > 0 and stB3["boot_ci_4k_seed7"][0] > 0
                  and sanity_pass and wf_pass)
    res["comparisons"] = {
        "B3_minus_gmar_primary_A3_nested_gain": round(gain - gm_primary, 6),
        "gmar_primary_A3_nested_gain": gm_primary,
        "rule": "within 0.0002 -> the simpler (gmar) wins",
        "B3_minus_B1_gain (skater term inside, given GL)": round(gain - stB1["gain_exact"], 6),
        "B3_minus_B4_gain (inside vs beside)": round(gain - stB4["gain_exact"], 6),
        "B3_minus_B5_gain (Jan-1 refits)": round(gain - stB5["gain_exact"], 6),
    }
    res["verdict"] = {"primary_arm": "B3", "nested_gain": round(gain, 6),
                      "boot_ci_10k": stB3["boot_ci_10k"],
                      "boot_ci_4k_seed7": stB3["boot_ci_4k_seed7"],
                      "sanity_pass": sanity_pass, "walk_forward_pass": wf_pass,
                      "failed_checks": [k for k, v in {**san, **wf}.items() if not v["pass"]],
                      "clears_bar": clears,
                      "baseline_dev_ll": round(s1, 6), "candidate_dev_ll": stB3["dev_ll"]}
    res["seconds"] = round(time.time() - t0, 1)
    print(f"\nVERDICT B3 nested gain {gain:+.5f} CI10k {stB3['boot_ci_10k']} sanity={sanity_pass}"
          f" wf={wf_pass} clears_bar={clears} ({res['seconds']}s)", flush=True)

    fd = pd.DataFrame({"game_id": [g["game_id"] for g in games],
                       "date": [g["date"] for g in games], "season": ctx.seasons,
                       "home": [g["home"] for g in games], "away": [g["away"] for g in games],
                       "in_eval_mask": mask.astype(int), "P_home": F["P_h"], "P_away": F["P_a"],
                       "dP_home": F["dP_h"], "dP_away": F["dP_a"],
                       "Ppre_home": F["Ppre_h"], "Ppre_away": F["Ppre_a"],
                       "G_home": F["G_h"], "G_away": F["G_a"],
                       "dressed_fallback_home": F["pfb_h"].astype(int),
                       "dressed_fallback_away": F["pfb_a"].astype(int)})
    assert fd.season.max() <= DEV_END and fd.game_id.max() < MAX_GID
    fd.to_csv(OUT_FEATS + ".tmp", index=False)
    os.replace(OUT_FEATS + ".tmp", OUT_FEATS)
    for path in (OUT, OUT_MIRROR):
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=float)
        os.replace(path + ".tmp", path)
    print(f"wrote {OUT}, {OUT_MIRROR}, {OUT_FEATS}")


if __name__ == "__main__":
    main()
