"""Player-value program, NHL -- game-level DEV screen: goalie INSIDE the Elo
(MLB starter-adjustment style) + dressed-lineup value, beside or replacing xG.

NOT a TEST look. Pre-registration: data/pv_nhl_screen_prereg.json (written before
this script and before any A1-A3 log-loss; its feature-file hashes are asserted).

Harness: phase0/nhl_depth_eval.py Ctx (DEV regular season, LOSO over 2011-12..
2017-18, n = 7,929); stats and nested-grid helpers of phase0/bt_nhl_nhl_gmar.py
(arm_stats: 10k seed-7 paired game bootstrap, 4k harness bootstrap, late-DEV
2015-18 readout; nested: inner pair-loss grid selection).

Features (data/pv_nhl_compose_game_features.csv, walk-forward, audited):
  lv  tonight's dressed-skater lineup value, goals/game
  gv  tonight's confirmed starting goalie's value, goals saved/game vs average

Arms (pre-registered; bar judged on A1 only):
  A1 PRIMARY  W/L Elo learns net of the starter: E = (R_h + g gv_h + 30) - (R_a + g gv_a);
              blend col 1 = logit(p), rest/b2b/xg kept, + lv diff; g nested {0..240}
  A2          shipped blend with xg_diff REPLACED by lv diff
  A3          A1 without xg_diff (goalie inside the Elo + lv diff replaces xG)
Diagnostics (not arms): D1 serving = A1 with day-of lv/gv replaced by the team's
previous-10-game means; D2 per-cell LOSO; D3 paired contrasts.

Market-blind: no odds are read. DEV only: the harness truncates to <= 2017-18,
the feature file is asserted < 2018000000, every scored mask is assert_dev_only.

    python phase0/pv_nhl_screen.py
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
import bt_nhl_nhl_gmar as GM  # noqa: E402  (arm_stats, nested, boot10k; unmodified)
from nhl_features_eval import TEAM_FIX  # noqa: E402
from nhl_glicko2_eval import DEV_END, TEST_START, llv  # noqa: E402
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

PRE = "data/pv_nhl_screen_prereg.json"
FEATS = "data/pv_nhl_compose_game_features.csv"
WEIGHTS = "data/pv_nhl_compose_weights.json"
COMPOSE_SCREEN = "data/pv_nhl_compose_screen.json"
OUT = "data/pv_nhl_screen.json"
GAMMAS = [0.0, 60.0, 120.0, 180.0, 240.0]      # Elo per goal/game (pre-declared)
K_ELO, HA_ELO, REG_ELO = Hd.SHIPPED_ELO
CUTOFFS = ("2013-02-15", "2015-01-15", "2017-02-01")
assert DEV_END == 20172018 and TEST_START == 20182019


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def elo_inside(games, y, adj_h, adj_a, gamma):
    """Shipped W/L Elo with the starter's value added to the side's pre-game rating.

    The prediction uses R + gamma * adj; the update uses the same p, so R learns the
    team's strength NET of tonight's starter (MLB FipPitcherElo pattern).  Every
    read of R at game i happens before game i's result is applied.
    """
    R = {}
    out = np.empty(len(games))
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - REG_ELO)
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        e = (rh + gamma * adj_h[i] + HA_ELO) - (ra + gamma * adj_a[i])
        p = 1.0 / (1.0 + 10 ** (-e / 400.0))
        out[i] = p
        R[g["home"]] += K_ELO * (y[i] - p)
        R[g["away"]] += K_ELO * ((1 - y[i]) - (1 - p))
    return out


def main():
    t0 = time.time()
    pre = json.load(open(PRE))
    assert pre["primary"] == "A1"
    w1_feat = sha(FEATS) == pre["features"]["sha256"]
    w1_w = sha(WEIGHTS) == pre["features"]["weights_sha256"]
    assert w1_feat and w1_w, "feature/weight file changed since the pre-registration"
    F = pd.read_csv(FEATS)
    assert int(F.gid.max()) < DEV_MAX_GID
    F = F.set_index("gid")

    ctx = Hd.Ctx()
    games = ctx.games
    assert max(g["season"] for g in games) <= DEV_END
    gid_all = np.array([g["game_id"] for g in games])
    y_all = np.array([g["y"] for g in games], float)
    mask = ctx.mask
    y = ctx.yv()
    folds = ctx.folds
    seas = folds.seasons
    Hd.assert_dev_only(seas)
    gid = gid_all[mask]
    assert gid.max() < DEV_MAX_GID
    mask_sha = hashlib.sha1(mask.tobytes()).hexdigest()
    y_sha = hashlib.sha1(y.tobytes()).hexdigest()

    # ------------------------------------------------------------ sanity ----
    san = {}
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    s1 = folds.pooled(y, p0)
    san["S1"] = {"shipped_dev_loso_ll": round(s1, 6), "n": int(len(y)), "target": 0.672398,
                 "pass": bool(abs(s1 - 0.672398) < 2e-6 and len(y) == 7929)}
    s2 = folds.pooled(y, folds.loso_pred(X0[:, :-1], y)) - s1
    san["S2"] = {"drop_xg_cost": round(s2, 5), "target": 0.00303, "tol": 0.0002,
                 "pass": bool(abs(s2 - 0.00303) <= 0.0002)}
    print(f"S1 {s1:.6f} n={len(y)}  S2 drop-xG {s2:+.5f}", flush=True)

    Fm = F.reindex(gid)
    cov = float(Fm.lv_home.notna().mean() * Fm.gv_home.notna().mean()
                * Fm.lv_away.notna().mean() * Fm.gv_away.notna().mean())
    fh = np.array([TEAM_FIX.get(t, t) for t in Fm.home.astype(str)])
    fa = np.array([TEAM_FIX.get(t, t) for t in Fm.away.astype(str)])
    hh = np.array([g["home"] for g, m in zip(games, mask) if m])
    ha = np.array([g["away"] for g, m in zip(games, mask) if m])
    code_ok = float(np.mean((fh == hh) & (fa == ha)))
    san["S4"] = {"feature_coverage": cov, "home_away_code_match": code_ok,
                 "pass": bool(cov == 1.0 and code_ok == 1.0)}

    d_lv = (Fm.lv_home - Fm.lv_away).to_numpy(float)
    Fa = F.reindex(gid_all)
    gv_h = Fa.gv_home.to_numpy(float)
    gv_a = Fa.gv_away.to_numpy(float)
    n_gv_missing_all = int(np.isnan(gv_h).sum() + np.isnan(gv_a).sum())
    gv_h, gv_a = np.nan_to_num(gv_h), np.nan_to_num(gv_a)

    # inside-Elo columns for every gamma (full DEV game list, then masked)
    pin = {gm: elo_inside(games, y_all, gv_h, gv_a, gm) for gm in GAMMAS}
    el = {gm: logit(pin[gm])[mask] for gm in GAMMAS}
    ref = Hd.run_elo_arr(games, *Hd.SHIPPED_ELO)
    base = [X0[:, 2], X0[:, 3], X0[:, 4]]          # rest_diff, b2b_home, b2b_away
    xgc = X0[:, 5]
    cellsA1 = {gm: Hd.design([el[gm]] + base + [xgc, d_lv]) for gm in GAMMAS}
    cellsA3 = {gm: Hd.design([el[gm]] + base + [d_lv]) for gm in GAMMAS}
    XA2 = Hd.design([X0[:, 1]] + base + [d_lv])
    s3a = bool(np.array_equal(pin[0.0], ref))
    s3b = bool(np.array_equal(cellsA1[0.0], np.column_stack([X0, d_lv])))
    san["S3"] = {"gamma0_elo_bitwise_equals_run_elo_arr": s3a,
                 "A1_gamma0_design_equals_blend_plus_lv": s3b, "pass": bool(s3a and s3b)}

    comp = json.load(open(COMPOSE_SCREEN))["C3 lv"]["gain_exact"]
    pC3 = folds.loso_pred(cellsA1[0.0], y)
    gC3 = float((llv(y, p0) - llv(y, pC3)).mean())
    san["S5"] = {"blend_plus_lv_gain": gC3, "compose_C3_gain_exact": comp,
                 "pass": bool(abs(gC3 - comp) < 1e-9)}
    for k, v in san.items():
        print(f"  {k}: {v}", flush=True)

    # ------------------------------------------------------ walk-forward ----
    wf = {"W1": {"feature_sha256_matches_prereg_and_audited_file": w1_feat,
                 "weights_sha256_matches": w1_w,
                 "compose_audit_pass": bool(json.load(open(
                     "data/pv_nhl_compose_audit.json"))["pass"]),
                 }}
    wf["W1"]["pass"] = bool(w1_feat and w1_w and wf["W1"]["compose_audit_pass"])
    w2 = {}
    dates = np.array([g["date"] for g in games])
    for ci, cut in enumerate(CUTOFFS):
        rng = np.random.default_rng(500 + ci)
        after = np.where(dates >= cut)[0]
        y2 = y_all.copy()
        y2[after] = y_all[rng.permutation(after)]
        h2, a2 = gv_h.copy(), gv_a.copy()
        h2[after] = rng.normal(0, 0.3, len(after))
        a2[after] = rng.normal(0, 0.3, len(after))
        pre_m = dates < cut
        same, changed = True, False
        for gm in GAMMAS:
            q = elo_inside(games, y2, h2, a2, gm)
            same &= bool(np.array_equal(q[pre_m], pin[gm][pre_m]))
            changed |= bool(not np.array_equal(q[~pre_m], pin[gm][~pre_m]))
        w2[cut] = {"n_pre": int(pre_m.sum()), "n_after": int(len(after)),
                   "pre_identical_all_gammas": same, "post_changed": changed,
                   "pass": bool(same and changed)}
        print(f"  W2 {cut}: {w2[cut]}", flush=True)
    wf["W2"] = {"cutoffs": w2, "pass": all(v["pass"] for v in w2.values())}

    # --------------------------------------------------------------- arms ----
    res = {"protocol": {"dev_only": True, "n": int(len(y)),
                        "seasons": sorted(int(s) for s in set(seas)),
                        "test_seasons_touched": 0, "market_inputs": 0,
                        "prereg": PRE, "prereg_sha256": sha(PRE),
                        "eval_mask_sha1": mask_sha, "y_sha1": y_sha,
                        "gamma_grid_elo_per_goal": GAMMAS,
                        "n_gv_missing_all_dev_games_set_to_0": n_gv_missing_all},
           "sanity": san}
    arms, arm_mask = {}, {}

    def per_season(p):
        return {str(s): round(float((llv(y[folds.idx[s]], p0[folds.idx[s]])
                                     - llv(y[folds.idx[s]], p[folds.idx[s]])).mean()), 5)
                for s in folds.list}

    def rec(name, p, extra=None):
        st = GM.arm_stats(y, folds, p0, p, seas)
        st["per_season_gain"] = per_season(p)
        if extra:
            st.update(extra)
        arms[name] = st
        arm_mask[name] = mask_sha
        print(f"  {name:<40} LL {st['dev_ll']:.6f} gain {st['gain_exact']:+.5f} "
              f"CI10k {st['boot_ci_10k']} {st['n_pos_folds']}/{st['n_folds']} "
              f"late {st['late_dev_2015_18']}", flush=True)
        return st

    def nested_arm(name, cells):
        for c, X in cells.items():
            assert not np.isnan(X).any(), f"NaN in {name} cell {c}"
        pn, picks, loso, cell_ll, best = GM.nested(cells, y, folds)
        wmod = Hd.fit_logit(cells[max(set(picks.values()), key=list(picks.values()).count)],
                            y, Hd.BLEND_C)
        st = rec(name, pn, {
            "picks": {str(k): v for k, v in picks.items()},
            "cell_loso_ll_OPTIMISTIC": {f"{k:g}": round(v, 6) for k, v in cell_ll.items()},
            "best_single_cell_OPTIMISTIC": {"gamma": best,
                                            "gain": round(s1 - cell_ll[best], 5)},
            "coef_modal_cell_all_dev_descriptive": [round(float(c), 4) for c in wmod]})
        return st, pn

    stA1, pA1 = nested_arm("A1 PRIMARY goalie-in-Elo + lv (xG kept)", cellsA1)
    assert not np.isnan(XA2).any()
    pA2 = folds.loso_pred(XA2, y)
    stA2 = rec("A2 lv replaces xG", pA2, {"coef_all_dev_descriptive": [
        round(float(c), 4) for c in Hd.fit_logit(XA2, y, Hd.BLEND_C)]})
    stA3, pA3 = nested_arm("A3 goalie-in-Elo + lv replaces xG", cellsA3)
    res["arms"] = arms

    # --------------------------------------------------------- diagnostics ----
    diag = {}
    lvp_h = (Fa.lv_home - Fa.lv_dev_home).to_numpy(float)
    lvp_a = (Fa.lv_away - Fa.lv_dev_away).to_numpy(float)
    gvp_h = (Fa.gv_home - Fa.gv_dev_home).to_numpy(float)
    gvp_a = (Fa.gv_away - Fa.gv_dev_away).to_numpy(float)
    nan_proj_masked = int(np.isnan(lvp_h[mask]).sum() + np.isnan(lvp_a[mask]).sum())
    lvp_h, lvp_a, gvp_h, gvp_a = (np.nan_to_num(v) for v in (lvp_h, lvp_a, gvp_h, gvp_a))
    d_lvp = (lvp_h - lvp_a)[mask]
    cellsD1 = {gm: Hd.design([logit(elo_inside(games, y_all, gvp_h, gvp_a, gm))[mask]]
                             + base + [xgc, d_lvp]) for gm in GAMMAS}
    pnD1, pkD1, _, llD1, _ = GM.nested(cellsD1, y, folds)
    stD1 = GM.arm_stats(y, folds, p0, pnD1, seas)
    stD1["picks"] = {str(k): v for k, v in pkD1.items()}
    stD1["nan_projected_side_values_masked_set_to_0"] = nan_proj_masked
    stD1["note"] = ("NOT an arm. Tonight's lineup and starter replaced by the team's "
                    "previous-10-game mean lv / gv: no day-of information.")
    stD1["share_of_A1_gain_retained"] = round(stD1["gain_exact"] / stA1["gain_exact"], 3)
    diag["D1 serving (no day-of info)"] = stD1
    print(f"  D1 serving gain {stD1['gain_exact']:+.5f} CI {stD1['boot_ci_10k']} "
          f"retained {stD1['share_of_A1_gain_retained']}", flush=True)

    def contrast(pa, pb, lab):
        d = llv(y, pa) - llv(y, pb)            # > 0: pb better
        late = seas >= GM.LATE_FROM
        return {"contrast": lab, "gain": round(float(d.mean()), 5),
                "boot_ci_10k": GM.boot10k(d),
                "late_dev_2015_18": {"gain": round(float(d[late].mean()), 5),
                                     "boot_ci_10k": GM.boot10k(d[late])}}
    diag["D3"] = [contrast(pC3, pA1, "A1 vs blend+lv (value of goalie inside the Elo)"),
                  contrast(pA1, pA3, "A3 vs A1 (dropping xG once lv is in)"),
                  contrast(pC3, pA2, "A2 vs blend+lv (dropping xG, goalie not in)")]
    for c in diag["D3"]:
        print("  D3", c, flush=True)
    # D4 (post-hoc, added after A1-A3 were scored; cannot change the verdict):
    # placebo -- lv diff and both gv series permuted within season (seed 11). A
    # sound harness must show no gain from personnel values that carry no signal.
    rng = np.random.default_rng(11)
    seas_all = ctx.seasons
    ph, pa_ = gv_h.copy(), gv_a.copy()
    for s in np.unique(seas_all):
        ix = np.where(seas_all == s)[0]
        ph[ix] = gv_h[rng.permutation(ix)]
        pa_[ix] = gv_a[rng.permutation(ix)]
    d_lv_pl = d_lv.copy()
    for s in folds.list:
        ix = np.where(seas == s)[0]
        d_lv_pl[ix] = d_lv[rng.permutation(ix)]
    cellsPl = {gm: Hd.design([logit(elo_inside(games, y_all, ph, pa_, gm))[mask]]
                             + base + [xgc, d_lv_pl]) for gm in GAMMAS}
    pnPl, pkPl, _, _, _ = GM.nested(cellsPl, y, folds)
    stPl = GM.arm_stats(y, folds, p0, pnPl, seas)
    diag["D4 placebo (post-hoc)"] = {k: stPl[k] for k in ("gain_exact", "boot_ci_10k",
                                                          "n_pos_folds", "dev_ll")}
    diag["D4 placebo (post-hoc)"]["picks"] = {str(k): v for k, v in pkPl.items()}
    print("  D4 placebo", diag["D4 placebo (post-hoc)"], flush=True)
    el_ship = X0[:, 1]
    diag["corr"] = {"lv_diff_vs_xg_diff": round(float(np.corrcoef(d_lv, xgc)[0, 1]), 3),
                    "lv_diff_vs_elo_logit": round(float(np.corrcoef(d_lv, el_ship)[0, 1]), 3),
                    "gv_diff_sd_masked": round(float((gv_h - gv_a)[mask].std()), 4),
                    "elo180_vs_shipped_elo_logit": round(float(
                        np.corrcoef(el[180.0], el_ship)[0, 1]), 4)}
    res["diagnostics_not_arms"] = diag

    wf["W3"] = {"identical_mask_all_arms": len(set(arm_mask.values())) == 1,
                "eval_mask_sha1": mask_sha, "y_sha1": y_sha,
                "nan_in_any_design": False, "pass": len(set(arm_mask.values())) == 1}
    res["walk_forward"] = wf

    sane = all(v["pass"] for v in san.values())
    wf_ok = all(v["pass"] for v in wf.values())
    clears = bool(sane and wf_ok and stA1["gain_exact"] >= 0.00100
                  and stA1["boot_ci_10k"][0] > 0)
    res["verdict"] = {"primary": "A1", "baseline_dev_ll": round(s1, 6),
                      "candidate_dev_ll": stA1["dev_ll"], "gain": stA1["gain_exact"],
                      "boot_ci_10k": stA1["boot_ci_10k"],
                      "late_dev_2015_18": stA1["late_dev_2015_18"],
                      "sanity_pass": bool(sane), "walk_forward_pass": bool(wf_ok),
                      "bar": "+0.00100, 10k CI lo > 0", "clears_bar": clears,
                      "day_of_information": pre["day_of_information"]}
    res["seconds"] = round(time.time() - t0, 1)
    with open(OUT + ".tmp", "w", encoding="utf-8") as fh_:
        json.dump(res, fh_, indent=1, default=float)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps(res["verdict"]), flush=True)


if __name__ == "__main__":
    main()
