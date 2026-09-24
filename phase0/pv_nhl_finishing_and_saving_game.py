"""Game-level DEV screen of the NHL finishing_and_saving component (DEV ONLY).

Harness: phase0/nhl_depth_eval.py (Ctx: DEV regular season, scored 2011-12..2017-18,
LOSO over seasons, n=7,929), with the round-1 stats/nesting helpers of
phase0/bt_nhl_nhl_gmar.py (imported, not modified).

Inputs: data/pv_nhl_finishing_and_saving_team_games.csv, written by
pv_nhl_finishing_and_saving_validate.py from the walk-forward player values:
  fin_pre   dressed skaters' expected goals above xG tonight, sum over the lineup of
            (walk-forward ixG/GP EWMA) x (exp(mu_finishing) - 1)
  sav_pre   tonight's STARTING goalie's expected goals saved: league xGA/GP (strictly
            earlier dates) x (1 - exp(-mu_saving))
  fs_pre    fin_pre + sav_pre                 (goals / game, per side)
  fs_env    fs_pre + team-context terms (team finishing / save environment)

Arms (everything else held fixed = the shipped 4-feature blend):
  A0  shipped blend (must reproduce DEV LOSO 0.672398; S2 drop-xG cost 0.00303)
  A1  + fs_diff                          (home - away, goals)
  A2  + fin_diff + sav_diff
  A3  + fs_env_diff
  A4  + fs_delta_diff  (tonight's lineup value minus the team's own EWMA of its
                        recent lineup values: the lineup CHANGE the Elo cannot know)
  A5  FS INSIDE the W/L Elo (MLB starting-pitcher style): E = (R_h + g fs_h + 30) -
      (R_a + g fs_a); the team Elo learns net of its lineup; g nested over
      {0, 60, 120, 180, 240} Elo/goal (LOSO-nested selection, round-1 protocol)
  A6  starting-goalie saving only, inside the Elo (the direct analogue of the MLB
      FIP starter adjustment and of round-1 GL), g nested
  A7  shipped xG team rating rebuilt on RECALIBRATED xG (shipped K/HA/regress)
  A8  shipped xG team rating rebuilt on TALENT-ADJUSTED expected goals: the margin
      is sum over the game's shots of P(goal | shot, shooter, goalie, team, rink)
      from PRE-game ratings (the team rating learns finishing & saving, not luck)
  A9  A8 + fs_diff
Bar (pre-registered): gain >= +0.00100 with bootstrap 95% CI lower bound > 0.

Market-blind: no odds.  Only DEV games are ever loaded (Hd.dev_games asserts).
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
sys.path.insert(0, HERE)
os.chdir(ROOT)
import nhl_depth_eval as Hd  # noqa: E402
import bt_nhl_nhl_gmar as GM  # noqa: E402  (arm_stats, nested; unmodified)
from nhl_glicko2_eval import DEV_END, llv  # noqa: E402

IN_TG = "data/pv_nhl_finishing_and_saving_team_games.csv"
OUT = "data/pv_nhl_finishing_and_saving_game.json"
GAMMAS = [0.0, 60.0, 120.0, 180.0, 240.0]
DELTA_HL = 10.0          # team-baseline EWMA half-life (team games) for fs_delta


def elo_inside(games, adj_h, adj_a, gamma, k=8, ha=30, reg=0.30):
    """run_elo_arr with a per-game additive Elo adjustment gamma * adj (goals)."""
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


def lgt(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


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
    res = {"component": "nhl finishing_and_saving -- game-level DEV screen",
           "protocol": {"n_dev": int(mask.sum()), "dev_seasons_scored": sorted(set(seas.tolist())),
                        "eval": "LOSO over DEV seasons (nhl_depth_eval.Ctx), nested gamma",
                        "market_blind": True, "test_seasons_touched": 0,
                        "bar": "gain >= +0.00100 AND bootstrap 95% CI lo > 0"}}

    # ---------------- harness sanity (S1, S2, S3)
    X0 = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    s1 = float(llv(y, p0).mean())
    pnx = folds.loso_pred(X0[:, :5], y)
    s2 = float(llv(y, pnx).mean() - s1)
    san = {"S1": {"shipped_dev_loso_ll": round(s1, 6), "target": 0.672398, "n": int(len(y)),
                  "pass": bool(abs(s1 - 0.672398) < 2e-6 and len(y) == 7929)},
           "S2": {"drop_xg_cost": round(s2, 5), "target": 0.00303,
                  "pass": bool(abs(s2 - 0.00303) <= 0.0002)}}
    print(f"S1 {s1:.6f}  S2 {s2:+.5f}", flush=True)

    # ---------------- per-game lineup values (walk-forward, from the player file)
    tg = pd.read_csv(IN_TG)
    assert int(tg.gid.max()) < 2018000000
    tg = tg.sort_values(["gi", "side_h"]).reset_index(drop=True)
    # fs_delta: tonight's lineup value minus the team's EWMA of its own earlier values
    lam = 0.5 ** (1 / DELTA_HL)
    base = np.full(len(tg), np.nan)
    st = {}
    for i, (team, v) in enumerate(zip(tg.team.values, tg.fs_pre.values)):
        s = st.get(team)
        base[i] = s[0] / s[1] if s else v
        if s is None:
            s = st[team] = [0.0, 0.0]
        s[0] = lam * s[0] + v
        s[1] = lam * s[1] + 1
    tg["fs_delta"] = tg.fs_pre - base
    cols = ["fs_pre", "fin_pre", "sav_pre", "fs_env_pre", "fs_delta"]
    H = tg[tg.side_h == 1].set_index("gid")
    A = tg[tg.side_h == 0].set_index("gid")
    gid = np.array([g["game_id"] for g in games])
    have = np.isin(gid, H.index) & np.isin(gid, A.index)
    res["coverage"] = {"all_dev_games": int(len(gid)), "with_values": int(have.sum()),
                       "masked_with_values": int(have[mask].sum()), "masked": int(mask.sum())}
    assert have[mask].all(), "a scored game lacks lineup values"
    V = {}
    for c in cols:
        h = H[c].reindex(gid).fillna(0).values
        a = A[c].reindex(gid).fillna(0).values
        V[c] = (h, a)
    for c in cols:
        d = V[c][0][mask] - V[c][1][mask]
        res.setdefault("feature_sd_diff_goals", {})[c] = round(float(d.std()), 4)

    # ---------------- blend arms
    def with_cols(extra):
        return np.column_stack([X0] + [V[c][0][mask] - V[c][1][mask] for c in extra])

    arms = {}
    for name, extra in (("A1 +fs_diff", ["fs_pre"]),
                        ("A2 +fin_diff+sav_diff", ["fin_pre", "sav_pre"]),
                        ("A3 +fs_env_diff", ["fs_env_pre"]),
                        ("A4 +fs_delta_diff", ["fs_delta"])):
        X = with_cols(extra)
        p = folds.loso_pred(X, y)
        st_ = GM.arm_stats(y, folds, p0, p, seas)
        w = Hd.fit_logit(X, y)
        st_["coef_all_dev_descriptive"] = [round(float(x), 4) for x in w[6:]]
        arms[name] = st_
        print(f"{name:28s} gain {st_['gain']:+.5f} CI10k {st_['boot_ci_10k']} {st_['sig']}", flush=True)

    # ---------------- inside-Elo arms (nested gamma)
    ref = Hd.run_elo_arr(games, *Hd.SHIPPED_ELO)
    chk = elo_inside(games, np.zeros(len(games)), np.zeros(len(games)), 0.0)
    san["S3"] = {"gamma0_vs_run_elo_arr_maxabs": float(np.max(np.abs(chk - ref))),
                 "pass": bool(np.max(np.abs(chk - ref)) == 0.0)}
    base_cols = X0[:, 2:]            # rest, b2b_h, b2b_a, xg (col 1 = elo logit)
    for name, key in (("A5 FS inside Elo (nested gamma)", "fs_pre"),
                      ("A6 starting-goalie saving inside Elo (nested gamma)", "sav_pre")):
        cells = {}
        for gm in GAMMAS:
            pe = elo_inside(games, V[key][0], V[key][1], gm)
            cells[gm] = np.column_stack([np.ones(mask.sum()), lgt(pe)[mask], base_cols])
        pred, picks, loso, cell_ll, best = GM.nested(cells, y, folds)
        st_ = GM.arm_stats(y, folds, p0, pred, seas)
        st_["picks"] = {str(k): v for k, v in picks.items()}
        st_["cell_loso_ll"] = {str(k): round(v, 6) for k, v in cell_ll.items()}
        st_["best_single_cell_OPTIMISTIC"] = {"gamma": best,
                                              "gain": round(cell_ll[0.0] - cell_ll[best], 5)}
        arms[name] = st_
        print(f"{name:28s} gain {st_['gain']:+.5f} CI10k {st_['boot_ci_10k']} {st_['sig']} "
              f"picks {picks}", flush=True)
    # ---------------- xG team rating rebuilt on talent-adjusted expected goals
    # (the shipped xG-Elo updates on raw MoneyPuck xG margin; here the margin is the
    #  sum over the game's shots of the goal probability given the shot AND who took /
    #  faced it, with PRE-game ratings -- used only to update after the game)
    shots = pd.read_parquet("data/pv_nhl_finishing_and_saving_shots.parquet",
                            columns=["gid", "team", "p_rc", "p_fs", "xg"])
    assert int(shots.gid.max()) < 2018000000
    txg = {k: {} for k in ("xg", "p_rc", "p_fs")}
    agg = shots.groupby(["gid", "team"])[["xg", "p_rc", "p_fs"]].sum()
    for (g_, tm), r in agg.iterrows():
        for k in txg:
            txg[k].setdefault(g_, {})[tm] = r[k]

    def run_xg_custom(src, k=Hd.SHIPPED_XG[0], ha=Hd.SHIPPED_XG[1], regress=Hd.SHIPPED_XG[2]):
        xr = {}
        out = np.full(len(games), np.nan)
        prev = None
        for i, g in enumerate(games):
            if prev is not None and g["season"] != prev:
                for tt in xr:
                    xr[tt] *= (1 - regress)
            prev = g["season"]
            rh, ra = xr.get(g["home"], 0.0), xr.get(g["away"], 0.0)
            out[i] = (rh + ha) - ra
            tx = src.get(g["game_id"], {})
            hx, ax = tx.get(g["home"], 0.0), tx.get(g["away"], 0.0)
            err = (hx - ax) - (rh - ra + ha)
            xr[g["home"]] = rh + k / 100.0 * err
            xr[g["away"]] = ra - k / 100.0 * err
        return out

    shipped_xg = ctx.xg(*Hd.SHIPPED_XG)
    own_raw = run_xg_custom(txg["xg"])[mask]
    san["S4_own_raw_xg_rating_vs_shipped_corr"] = {
        "corr": round(float(np.corrcoef(own_raw, shipped_xg)[0, 1]), 5),
        "note": "same update on this pipeline's own MoneyPuck-joined shots; ~1 expected"}
    for name, src, extra in (("A7 xG rating on RECALIBRATED xG", "p_rc", []),
                             ("A8 xG rating on TALENT-ADJUSTED xG (p_fs)", "p_fs", []),
                             ("A9 A8 + fs_diff", "p_fs", ["fs_pre"])):
        xcol = run_xg_custom(txg[src])[mask]
        X = np.column_stack([X0[:, :5], xcol] + [V[c][0][mask] - V[c][1][mask] for c in extra])
        p = folds.loso_pred(X, y)
        st_ = GM.arm_stats(y, folds, p0, p, seas)
        arms[name] = st_
        print(f"{name:28s} gain {st_['gain']:+.5f} CI10k {st_['boot_ci_10k']} {st_['sig']}", flush=True)
    res["harness_sanity"] = san
    res["arms"] = arms
    best_name = max(arms, key=lambda k: arms[k]["gain_exact"])
    b = arms[best_name]
    res["verdict"] = {"best_arm": best_name, "gain": b["gain"], "boot_ci_10k": b["boot_ci_10k"],
                      "clears_bar": bool(b["gain_exact"] >= 0.00100 and b["boot_ci_10k"][0] > 0),
                      "sanity_pass": bool(all(v["pass"] for v in san.values() if "pass" in v)),
                      "baseline_dev_ll": round(s1, 6), "best_dev_ll": b["dev_ll"]}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res["verdict"], indent=1))
    print(f"wrote {OUT} ({res['seconds']}s)")


if __name__ == "__main__":
    main()
