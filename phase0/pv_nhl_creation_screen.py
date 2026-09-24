"""Player-value program, NHL -- component "creation", STEP 4: lineup aggregate +
DEV-only game-level screen over the shipped blend.

NOT a TEST look. Seasons scored: DEV 2011-12..2017-18 (LOSO, the nhl_depth_eval
harness, n = 7,929 regular-season games). TEST seasons are never loaded: the game
list is nhl_depth_eval.dev_games() (asserts season <= 2017-18) and the lineup
values come from data/pv_nhl_creation_wf.parquet (DEV rows only, asserted).

Lineup aggregate (per team-game, walk-forward): tonight's dressed skaters (pbp
rosterSpots = the confirmed lineup, the ledger row-10 argument), each weighted
by his pre-game expected 5v5 minutes (decayed average of his earlier games;
players with < 60 prior EV minutes get the newcomer median), scaled to 5 skaters:
    LU_x = 5 * sum_i em_i x_i / sum_i em_i
for x in {cre_evx, cre_ev, rapm_off_s (the incumbent xG RAPM, 0 when unrated)}.
Written to data/pv_nhl_creation_lineup.csv (DEV team-games).

Arms (declared before the first scored run; the team-game GF check in
pv_nhl_creation_validate had already shown cre_evx as the best lineup index,
so the choice of primary is informed by DEV, not blind):
  A0  shipped blend [elo_logit, rest_diff, b2b_home, b2b_away, xg_diff]   (S1)
  S2  A0 minus xg_diff (harness sanity: must cost ~ +0.00303)
  A1  A0 + (LU_cre_evx home - away)            PRIMARY
  A2  A0 + (LU_cre_ev  home - away)
  A3  A0 + (LU_rapm_off_s home - away)         incumbent control (ledger row 8: null)
  A4  A0 + LU_cre_evx split F / D (two diffs)
Added after A1-A4 were scored (MLB-style "day-of deviation" arms; the team Elo
already carries the team's usual strength, so the day-of information is tonight's
lineup relative to the team's own recent lineups; half-life 20 team games,
declared, not searched):
  A5  A0 + LU_cre_evx diff + DELTA diff, DELTA = LU - team EWMA of its earlier LUs
  A6  A0 + DELTA(cre_evx) diff only
  A7  A0 + LU_cre_ev diff + DELTA(cre_ev) diff
Pre-registered bar (documents/breakthrough_program_prereg_2026_09_24.md): NHL
gain >= +0.00100 with bootstrap 95% CI lower bound > 0, sanity reproduced,
adversarial audit sound. Whether anything goes to TEST is the lead's call.

    python phase0/pv_nhl_creation_screen.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhl_depth_eval import (SHIPPED_ELO, SHIPPED_XG, Ctx, design,  # noqa: E402
                            fold_stats)
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, "data/pv_nhl_creation_wf.parquet")
LU = os.path.join(ROOT, "data/pv_nhl_creation_lineup.csv")
OUT = os.path.join(ROOT, "data/pv_nhl_creation_screen.json")
VALS = ["cre_evx", "cre_ev", "rapm_off_s"]


def lineup():
    w = pd.read_parquet(WF)
    assert int(w.gid.max()) < DEV_MAX_GID
    w["rapm_off_s"] = w.rapm_off_s.fillna(0.0)
    newc = w.prior_min_ev < 300
    fill = w[newc & w.exp_min_ev.notna()].groupby("grp").exp_min_ev.median()
    ok = w.exp_min_ev.notna() & (w.prior_min_ev >= 60)
    w["em"] = np.where(ok, w.exp_min_ev, w.grp.map(fill))
    rows = []
    for tag, sub in (("", w), ("_F", w[w.grp == "F"]), ("_D", w[w.grp == "D"])):
        s = sub.assign(**{v + "_w": sub[v] * sub.em for v in VALS})
        g = s.groupby(["gid", "side", "team"]).agg(
            date=("date", "first"), em=("em", "sum"), n=("pid", "size"),
            n_new=("prior_min_ev", lambda x: int((x < 300).sum())),
            **{v: (v + "_w", "sum") for v in VALS})
        scale = 5.0 if tag == "" else (3.0 if tag == "_F" else 2.0)
        for v in VALS:
            g[v] = g[v] / g.em * scale
        g = g.rename(columns={c: f"lu{tag}_{c}" for c in g.columns})
        rows.append(g)
    L = pd.concat(rows, axis=1).reset_index()
    L["date"] = L["lu_date"]
    L = L.drop(columns=[c for c in L.columns if c.endswith("_date")])
    # team EWMA of its own earlier lineups (strictly earlier games), half-life 20
    L = L.sort_values(["team", "date", "gid"]).reset_index(drop=True)
    lam = 0.5 ** (1 / 20.0)
    k = L.groupby("team").cumcount().to_numpy().astype(float)
    wn = lam ** (-k)
    for v in ("lu_cre_evx", "lu_cre_ev"):
        x = L[v].to_numpy(float)
        cs = pd.Series(x * wn).groupby(L.team.to_numpy()).cumsum().to_numpy() - x * wn
        cw = pd.Series(wn).groupby(L.team.to_numpy()).cumsum().to_numpy() - wn
        with np.errstate(invalid="ignore", divide="ignore"):
            prior = np.where(k > 0, cs / cw, np.nan)
        L[v + "_delta"] = x - prior
    L.to_csv(LU, index=False)
    return L


def main():
    L = lineup()
    ctx = Ctx()
    games = [g for g, m in zip(ctx.games, ctx.mask) if m]
    y = ctx.yv()
    folds = ctx.folds
    gid = np.array([g["game_id"] for g in games])
    assert gid.max() < DEV_MAX_GID
    H = L[L.side == "H"].set_index("gid")
    A = L[L.side == "A"].set_index("gid")
    # team-code check (pbp codes vs harness codes)
    hc = H.team.reindex(gid).to_numpy()
    code_ok = float(np.mean(hc == np.array([g["home"] for g in games])))

    def diff(col):
        d = (H[col].reindex(gid) - A[col].reindex(gid)).to_numpy(float)
        return d

    X0 = ctx.X_for(SHIPPED_ELO, SHIPPED_XG)
    p0 = folds.loso_pred(X0, y)
    s1 = folds.pooled(y, p0)
    Xd = X0[:, :-1]
    p_drop = folds.loso_pred(Xd, y)
    s2 = folds.pooled(y, p_drop) - s1
    res = {"protocol": {"dev_only": True, "n": int(len(y)),
                        "seasons": sorted(set(int(s) for s in folds.seasons)),
                        "test_seasons_touched": 0, "market_inputs": 0},
           "sanity": {"S1_shipped_dev_loso_ll": round(s1, 6), "S1_target": 0.672398,
                      "S1_pass": bool(abs(s1 - 0.672398) < 2e-6 and len(y) == 7929),
                      "S2_drop_xg_cost": round(s2, 5), "S2_target": 0.00303,
                      "S2_pass": bool(abs(s2 - 0.00303) < 0.0002),
                      "home_code_match": code_ok}}
    arms = {"A1_cre_evx": ["lu_cre_evx"], "A2_cre_ev": ["lu_cre_ev"],
            "A3_rapm_off_s": ["lu_rapm_off_s"], "A4_cre_evx_F_D": ["lu_F_cre_evx", "lu_D_cre_evx"],
            "A5_cre_evx_level_delta": ["lu_cre_evx", "lu_cre_evx_delta"],
            "A6_cre_evx_delta": ["lu_cre_evx_delta"],
            "A7_cre_ev_level_delta": ["lu_cre_ev", "lu_cre_ev_delta"]}
    for name, cols in arms.items():
        ds = [diff(c) for c in cols]
        miss = int(sum(np.isnan(d).sum() for d in ds))
        ds = [np.nan_to_num(d) for d in ds]
        X = np.column_stack([X0] + ds)
        p = folds.loso_pred(X, y)
        st = fold_stats(y, folds, p0, p)
        full = np.linalg.lstsq(X, y, rcond=None)[0]  # sign check only (linear prob.)
        st["missing_diffs"] = miss
        st["feature_sd"] = [round(float(np.std(d)), 4) for d in ds]
        st["lpm_coef_sign"] = [float(np.sign(c)) for c in full[-len(ds):]]
        st["loso_ll"] = round(folds.pooled(y, p), 6)
        st["per_season_gain"] = {str(s): round(float(
            (np.mean(-(y[folds.idx[s]] * np.log(p0[folds.idx[s]]) + (1 - y[folds.idx[s]])
                       * np.log(1 - p0[folds.idx[s]])))
             - np.mean(-(y[folds.idx[s]] * np.log(p[folds.idx[s]]) + (1 - y[folds.idx[s]])
                         * np.log(1 - p[folds.idx[s]]))))), 5) for s in folds.list}
        res[name] = st
        print(name, json.dumps({k: st[k] for k in ("gain", "boot_ci", "sig", "n_pos_folds",
                                                   "loso_ll")}), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["sanity"]))


if __name__ == "__main__":
    main()
