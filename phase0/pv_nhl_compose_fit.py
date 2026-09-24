"""Player-value program, NHL -- COMPOSE step 2: walk-forward weights, validation
against the incumbent xG RAPM, and the per-game DEV features (DEV ONLY).

Specification: phase0/pv_nhl_compose_core.py (declared components, prior weights,
walk-forward weight fitting). This script:

  S0  goalie design rule (declared in the core): the experience-prior variant G2
      is adopted only if it improves walk-forward per-start goals-saved squared
      error with a bootstrap CI above zero. Recorded either way.
  W   walk-forward weights w_s (seasons 2011-12..s-1 -> season s), the prior
      weights, and a DESCRIPTIVE all-DEV fit (reported only, never used for a
      value or a feature).
  E1  GAME level, out of sample: predict each game's goal differential (actual
      goals, empty-net and shootout excluded) from the lineup aggregates with the
      season's walk-forward weights; compare with the incumbent lineup power
      P = sum v_RAPM * m5 / 60 (bt_nhl_rapmel, walk-forward) whose slope and
      intercept are also fit walk-forward. Scored 2012-13..2017-18 (every season
      has at least one earlier fitting season); 2011-12 reported separately at the
      prior weights. Paired game-bootstrap CIs on the MSE difference.
  E2  NEXT-PERIOD (team-month) level: each player's value frozen at his first game
      of the calendar month (= built only from games before the month), weighted
      by the minutes he actually played in the month; target = the team's actual
      goal differential over the month. Same comparison, team-season cluster
      bootstrap.
  E3  PLAYER level: value at the player's first game of season s vs his season-s
      on-ice outcomes (5v5 GD/60, relative 5v5 GD/60, all-strength on-ice GD/60,
      5v5 GF/60 and GA/60), TOI-weighted, within position, player-cluster
      bootstrap, team changers separately; incumbent = rapm_net_v.
  F   face validity lists (end-of-DEV snapshot) and the per-game feature files.

Outputs (DEV, gid < 2018000000 asserted):
  data/pv_nhl_compose_weights.json         weights per season + prior + descriptive
  data/pv_nhl_compose_validation.json      S0, E1, E2, E3, face validity
  data/pv_nhl_compose_player_values.parquet  per skater-game values (goals / 60)
  data/pv_nhl_compose_goalie_values.csv    per starting-goalie-game value
  data/pv_nhl_compose_game_features.csv    per game: lv_home/away, gv_home/away,
                                           lineup deviation, components (walk-forward)

    python phase0/pv_nhl_compose_fit.py
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
import pv_nhl_compose_core as C  # noqa: E402
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

D = C.D
OUT_W = D("pv_nhl_compose_weights.json")
OUT_V = D("pv_nhl_compose_validation.json")
OUT_PV = D("pv_nhl_compose_player_values.parquet")
OUT_GV = D("pv_nhl_compose_goalie_values.csv")
OUT_GF = D("pv_nhl_compose_game_features.csv")
SCORED = [20122013, 20132014, 20142015, 20152016, 20162017, 20172018]
NB = 4000
RNG = np.random.default_rng(20260924)


def r6(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 6)


def boot_mean(d, clusters=None, nb=NB):
    d = np.asarray(d, float)
    if clusters is None:
        bs = d[RNG.integers(0, len(d), size=(nb, len(d)))].mean(axis=1)
    else:
        cl = pd.factorize(np.asarray(clusters))[0]
        k = cl.max() + 1
        s = np.bincount(cl, weights=d, minlength=k)
        n = np.bincount(cl, minlength=k).astype(float)
        idx = RNG.integers(0, k, size=(nb, k))
        bs = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [r6(lo), r6(hi)]


# ------------------------------------------------------------------ S0 goalie --
def goalie_rule(gk, tg):
    g = C.goalie_experience(gk)
    rho = C.goalie_offsets(g)
    st = g[(g.g_start == 1) & (g.gtype == 2) & g.season.isin(SCORED)].copy()
    off = np.array([rho[s][b] for s, b in zip(st.season, st.bucket)])
    real = (st.fa_xrc - st.fa_g).to_numpy()
    p0 = (st.fa_xrc * (1 - np.exp(-st.sav_mu))).to_numpy()
    p2 = (st.fa_xrc * (1 - np.exp(-(st.sav_mu + off)))).to_numpy()
    d = (real - p0) ** 2 - (real - p2) ** 2
    ci = boot_mean(d, clusters=st.gid.to_numpy())
    adopt = bool(ci[0] > 0)
    return {"variant": "G2 experience-bucket prior offset (declared in core)",
            "n_starts": int(len(d)), "mse_gain_per_start": r6(d.mean()), "ci": ci,
            "per_season": {str(s): r6(d[st.season.to_numpy() == s].mean()) for s in SCORED},
            "rho_by_season": {str(s): {str(b): r6(v) for b, v in r.items()}
                              for s, r in rho.items()},
            "adopted": adopt,
            "decision": "G2 adopted" if adopt else "G2 REJECTED (worse or n.s.); the goalie "
            "value is the 1v1 saving rating"}, rho, adopt


# ------------------------------------------------------------------- RAPM P ---
def rapm_lineup():
    f = pd.read_csv(D("bt_nhl_rapmel_feats.csv"))
    f = f[f.game_id < DEV_MAX_GID]
    return f.rename(columns={"game_id": "gid"})[["gid", "P_home", "P_away"]]


def ols_wf(G, cols, target="gd_noen", face=None):
    """Walk-forward OLS (intercept + cols) for each scored season from 2011-12..s-1.
    Season 2011-12: `face` coefficients (dict) with intercept 0.25."""
    pred = np.full(len(G), np.nan)
    coefs = {}
    for s in [20112012] + SCORED:
        m = (G.season == s).to_numpy()
        tr = ((G.season >= C.FIRST_FIT_SEASON) & (G.season < s)).to_numpy()
        if tr.sum() == 0:
            b = np.r_[0.25, [face.get(c, 1.0) if face else 1.0 for c in cols]]
        else:
            Z = np.column_stack([np.ones(tr.sum())] + [G.loc[tr, c].to_numpy(float)
                                                        for c in cols])
            b = np.linalg.lstsq(Z, G.loc[tr, target].to_numpy(float), rcond=None)[0]
        Zs = np.column_stack([np.ones(m.sum())] + [G.loc[m, c].to_numpy(float) for c in cols])
        pred[m] = Zs @ b
        coefs[str(s)] = [r6(v) for v in b]
    return pred, coefs


def compare(y, preds, base_key, keys, clusters=None):
    out = {}
    e_base = (y - preds[base_key]) ** 2
    var = float(np.var(y))
    for k in keys:
        e = (y - preds[k]) ** 2
        d = e_base - e
        out[k] = {"mse": r6(e.mean()), "r2_vs_var": r6(1 - e.mean() / var),
                  f"mse_gain_vs_{base_key}": r6(d.mean()),
                  "ci": boot_mean(d, clusters)}
    return out


# ------------------------------------------------------------------------ E1 ---
def e1_game(G, W, Wp):
    P = rapm_lineup()
    G = G.merge(P, on="gid", how="left")
    cov = float(G.P_home.notna().mean())
    G["d_P"] = (G.P_home - G.P_away).fillna(0.0)
    y = G.gd_noen.to_numpy(float)
    preds = {}
    # HFA only
    preds["hfa"], _ = ols_wf(G, [], face={})
    preds["composite"] = C.apply_weights(G, W)
    preds["composite_prior_w"] = C.apply_weights(G, Wp)
    preds["composite_skaters"] = C.apply_weights(G, W, comps=C.SK_COMPS)
    preds["goalie_only"] = C.apply_weights(G, W, comps=["goalie"])
    preds["rapm_P"], cP = ols_wf(G, ["d_P"], face={"d_P": 1.0})
    G["d_LV"] = C.apply_weights(G, W) - np.array([W[s]["b"] for s in G.season])
    preds["composite_plus_rapm"], cPC = ols_wf(G, ["d_LV", "d_P"],
                                                face={"d_LV": 1.0, "d_P": 0.0})
    preds["composite_refit"], _ = ols_wf(G, ["d_LV"], face={"d_LV": 1.0})
    # face-value RAPM: slope 1, intercept walk-forward HFA
    preds["rapm_P_face"] = preds["hfa"] + G.d_P.to_numpy()
    # single components (walk-forward OLS slope), diagnostics
    for c in C.COMPS:
        preds["only_" + c], _ = ols_wf(G, ["d_" + c], face={"d_" + c: C.PRIOR_W[c]})
    sc = G.season.isin(SCORED).to_numpy()
    keys = [k for k in preds if k != "hfa"]
    res = {"n_games": int(sc.sum()), "rapm_P_coverage": r6(cov),
           "gd_var": r6(np.var(y[sc])),
           "vs_hfa": compare(y[sc], {k: v[sc] for k, v in preds.items()}, "hfa", keys),
           "composite_vs_rapm": compare(y[sc], {k: v[sc] for k, v in preds.items()},
                                        "rapm_P", ["composite", "composite_prior_w",
                                                   "composite_skaters", "composite_plus_rapm"]),
           "rapm_beyond_composite": compare(y[sc], {k: v[sc] for k, v in preds.items()},
                                            "composite_refit", ["composite_plus_rapm"]),
           "rapm_P_wf_coefs": cP, "composite_plus_rapm_coefs": cPC}
    per = {}
    for s in SCORED:
        m = (G.season == s).to_numpy()
        per[str(s)] = {k: r6(np.mean((y[m] - preds[k][m]) ** 2)) for k in
                       ("hfa", "composite", "rapm_P", "composite_prior_w")}
    res["per_season_mse"] = per
    res["seasons_composite_beats_rapm"] = int(sum(per[str(s)]["composite"] <
                                                  per[str(s)]["rapm_P"] for s in SCORED))
    # 2011-12 (prior weights, no fitted parameter anywhere)
    m = (G.season == 20112012).to_numpy()
    res["season_2011_12_prior_weights"] = {
        "n": int(m.sum()), "mse_hfa": r6(np.mean((y[m] - preds["hfa"][m]) ** 2)),
        "mse_composite_prior": r6(np.mean((y[m] - preds["composite"][m]) ** 2)),
        "mse_rapm_face": r6(np.mean((y[m] - preds["rapm_P"][m]) ** 2))}
    # GF / GA split: offence part vs goals for, defence + goalie vs goals against
    off = ["ev_shot", "ev_fin", "ev_a1", "ev_a2", "pp_xg", "pp_ast"]
    dfn = ["ev_def", "goalie"]
    oh = np.zeros(len(G))
    oa = np.zeros(len(G))
    dh = np.zeros(len(G))
    da = np.zeros(len(G))
    for s, ws in W.items():
        mm = (G.season == s).to_numpy()
        for c in off:
            oh[mm] += ws["w"][c] * G.loc[mm, "h_" + c].to_numpy()
            oa[mm] += ws["w"][c] * G.loc[mm, "a_" + c].to_numpy()
        for c in dfn:
            dh[mm] += ws["w"][c] * G.loc[mm, "h_" + c].to_numpy()
            da[mm] += ws["w"][c] * G.loc[mm, "a_" + c].to_numpy()
    # team-game rows: GF of side = f(own offence - opp defence)
    gf = np.r_[G.gf_noen.to_numpy(), G.ga_noen.to_numpy()]
    xo = np.r_[oh, oa]
    xd = np.r_[da, dh]           # opponent's defence+goalie value (goals prevented)
    ph = np.r_[G.P_home.fillna(0).to_numpy(), G.P_away.fillna(0).to_numpy()]
    pa = np.r_[G.P_away.fillna(0).to_numpy(), G.P_home.fillna(0).to_numpy()]
    ss = np.r_[G.season.to_numpy(), G.season.to_numpy()]
    msk = np.isin(ss, SCORED)
    cc = lambda a, b: r6(np.corrcoef(a[msk], b[msk])[0, 1])  # noqa: E731
    res["gf_split_corr"] = {
        "team_goals_for_vs_own_offence_value": cc(gf, xo),
        "team_goals_for_vs_minus_opp_defence_goalie_value": cc(gf, -xd),
        "team_goals_for_vs_own_P_minus_opp_P": cc(gf, ph - pa),
        "note": "descriptive correlations on team-game rows, scored seasons"}
    return res, G


# ------------------------------------------------------------------------ E2 ---
def e2_month(sk, x, m, pv, G, gv_st, W):
    """Team-month blocks, values frozen at each player's first game of the month."""
    s = sk[sk.gtype == 2][["gid", "date", "season", "team", "side", "pid",
                           "toi_ev", "toi_pp", "toi_s", "rapm_net_v"]].copy()
    s["month"] = s.date.str[:7]
    s["v_ev60"] = pv.loc[s.index, "v_ev60"].to_numpy()
    s["v_pp60"] = pv.loc[s.index, "v_pp60"].to_numpy()
    s["v_dd60"] = pv.loc[s.index, "v_dd60"].to_numpy()
    s = s.sort_values(["date", "gid"])
    # first ROW of each (team, month, player) -- drop_duplicates never skips NaNs
    # (groupby.first would borrow a later game's value when the first is NaN)
    keys = ["season", "team", "month", "pid"]
    fr = s.drop_duplicates(keys, keep="first")[keys + ["v_ev60", "v_pp60", "v_dd60",
                                                       "rapm_net_v"]]
    fr.columns = keys + ["f_ev", "f_pp", "f_dd", "f_rapm"]
    s = s.merge(fr, on=keys, how="left", validate="m:1")
    s["f_rapm"] = s.f_rapm.fillna(0.0)
    s["lu_comp"] = (s.f_ev * s.toi_ev + s.f_pp * s.toi_pp + s.f_dd * s.toi_s) / 3600.0
    s["lu_rapm"] = s.f_rapm * s.toi_ev / 3600.0
    lu = s.groupby(["gid", "side"])[["lu_comp", "lu_rapm"]].sum().reset_index()
    lu = lu.merge(gv_st[["gid", "side", "g_pid", "gv"]], on=["gid", "side"], how="left")
    # goalie frozen at his first start of the month for that team
    tm = sk[["gid", "side", "team", "date", "season"]].drop_duplicates(["gid", "side"])
    lu = lu.merge(tm, on=["gid", "side"], how="left")
    lu["month"] = lu.date.str[:7]
    lu = lu.sort_values(["date", "gid"])
    gk_ = ["season", "team", "month", "g_pid"]
    fg = lu.drop_duplicates(gk_, keep="first")[gk_ + ["gv"]].rename(columns={"gv": "gv_f"})
    lu = lu.merge(fg, on=gk_, how="left", validate="m:1")
    lu["lu_comp_g"] = lu.lu_comp + lu.gv_f.fillna(0.0)
    H = lu[lu.side == "H"].set_index("gid")
    A = lu[lu.side == "A"].set_index("gid")
    gg = G[["gid", "season", "gd_noen"]].set_index("gid")
    ids = gg.index.intersection(H.index).intersection(A.index)
    rows = []
    for side, own, opp, sign in (("H", H, A, 1), ("A", A, H, -1)):
        r = pd.DataFrame({"gid": ids, "season": gg.loc[ids, "season"].to_numpy(),
                          "team": own.loc[ids, "team"].to_numpy(),
                          "month": own.loc[ids, "month"].to_numpy(),
                          "home": 1 if side == "H" else 0,
                          "gd": sign * gg.loc[ids, "gd_noen"].to_numpy(),
                          "comp": own.loc[ids, "lu_comp_g"].to_numpy()
                          - opp.loc[ids, "lu_comp_g"].to_numpy(),
                          "comp_sk": own.loc[ids, "lu_comp"].to_numpy()
                          - opp.loc[ids, "lu_comp"].to_numpy(),
                          "rapm": own.loc[ids, "lu_rapm"].to_numpy()
                          - opp.loc[ids, "lu_rapm"].to_numpy()})
        rows.append(r)
    T = pd.concat(rows, ignore_index=True)
    B = T.groupby(["season", "team", "month"]).agg(
        n=("gid", "size"), nh=("home", "sum"), gd=("gd", "sum"), comp=("comp", "sum"),
        comp_sk=("comp_sk", "sum"), rapm=("rapm", "sum")).reset_index()
    B = B[B.n >= 5].reset_index(drop=True)
    B["hfa_n"] = 2 * B.nh - B.n
    y = B.gd.to_numpy(float)
    preds = {}
    for name, cols in (("hfa", ["hfa_n"]), ("composite", ["hfa_n", "comp"]),
                       ("composite_skaters", ["hfa_n", "comp_sk"]),
                       ("rapm", ["hfa_n", "rapm"]), ("both", ["hfa_n", "comp", "rapm"])):
        p = np.full(len(B), np.nan)
        for s in SCORED:
            mm = (B.season == s).to_numpy()
            tr = ((B.season >= C.FIRST_FIT_SEASON) & (B.season < s)).to_numpy()
            Z = np.column_stack([B.loc[tr, c].to_numpy(float) for c in cols])
            b = np.linalg.lstsq(Z, y[tr], rcond=None)[0]
            p[mm] = np.column_stack([B.loc[mm, c].to_numpy(float) for c in cols]) @ b
        preds[name] = p
    # face value: composite is already in goals (weights fit at game level); RAPM
    # P in xG, slope 1; HFA walk-forward from the hfa fit
    hfa_b = {}
    for s in SCORED:
        tr = ((B.season >= C.FIRST_FIT_SEASON) & (B.season < s)).to_numpy()
        hfa_b[s] = float(np.sum(B.hfa_n[tr] * y[tr]) / np.sum(B.hfa_n[tr] ** 2))
    hb = B.season.map(hfa_b).to_numpy(float) * B.hfa_n.to_numpy(float)
    preds["composite_face"] = hb + B.comp.to_numpy()
    preds["rapm_face"] = hb + B.rapm.to_numpy()
    sc = B.season.isin(SCORED).to_numpy()
    cl = (B.season.astype(str) + B.team).to_numpy()[sc]
    keys = [k for k in preds if k != "hfa"]
    res = {"n_blocks": int(sc.sum()), "mean_games_per_block": r6(B.n[sc].mean()),
           "block_gd_var": r6(np.var(y[sc])),
           "vs_hfa": compare(y[sc], {k: v[sc] for k, v in preds.items()}, "hfa", keys, cl),
           "composite_vs_rapm": compare(y[sc], {k: v[sc] for k, v in preds.items()}, "rapm",
                                        ["composite", "composite_skaters", "both"], cl),
           "face_composite_vs_face_rapm": compare(
               y[sc], {k: v[sc] for k, v in preds.items()}, "rapm_face",
               ["composite_face"], cl),
           "corr": {"composite": r6(np.corrcoef(y[sc], B.comp[sc])[0, 1]),
                    "rapm": r6(np.corrcoef(y[sc], B.rapm[sc])[0, 1])},
           "note": "values frozen at each player's first game of the month (built only from "
                   "games before the month); actual minutes in the month as weights; target "
                   "= actual month GD (no EN / SO goals); blocks with >= 5 games"}
    return res


# ------------------------------------------------------------------------ E3 ---
def wcorr(a, b, w):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.asarray(w, float)
    ma = np.sum(w * a) / w.sum()
    mb = np.sum(w * b) / w.sum()
    return np.sum(w * (a - ma) * (b - mb)) / np.sqrt(np.sum(w * (a - ma) ** 2)
                                                      * np.sum(w * (b - mb) ** 2))


def boot_corr_diff(P, xa, xb, y, w="w", cl="pid", nb=2000):
    """TOI-weighted corr(xa, y), corr(xb, y) and their difference, player-cluster boot."""
    ra = wcorr(P[xa], P[y], P[w])
    rb = wcorr(P[xb], P[y], P[w]) if xb else np.nan
    codes = pd.factorize(P[cl])[0]
    k = codes.max() + 1
    groups = np.split(np.argsort(codes, kind="stable"),
                      np.cumsum(np.bincount(codes, minlength=k))[:-1])
    A = P[xa].to_numpy(float)
    Bv = P[xb].to_numpy(float) if xb else None
    Y = P[y].to_numpy(float)
    Wt = P[w].to_numpy(float)
    ra_b, d_b = [], []
    for _ in range(nb):
        pick = RNG.integers(0, k, k)
        idx = np.concatenate([groups[i] for i in pick])
        x1 = wcorr(A[idx], Y[idx], Wt[idx])
        ra_b.append(x1)
        if xb:
            d_b.append(x1 - wcorr(Bv[idx], Y[idx], Wt[idx]))
    out = {"r": r6(ra), "ci": [r6(v) for v in np.percentile(ra_b, [2.5, 97.5])]}
    if xb:
        out.update({"r_incumbent": r6(rb), "diff": r6(ra - rb),
                    "diff_ci": [r6(v) for v in np.percentile(d_b, [2.5, 97.5])]})
    return out


def e3_player(sk, pv):
    s = sk[sk.gtype == 2].copy()
    s = s.join(pv[["v_ev60", "v_all60", "v_pp60", "v_dd60"]])
    s = s.sort_values(["date", "gid"])
    first = s.drop_duplicates(["season", "pid"], keep="first").set_index(["season", "pid"])
    agg = s.groupby(["season", "pid"]).agg(
        toi_ev=("toi_ev", "sum"), toi_pp=("toi_pp", "sum"), toi_sh=("toi_sh", "sum"),
        on_gf_ev=("on_gf_ev", "sum"), on_ga_ev=("on_ga_ev", "sum"),
        on_gf_pp=("on_gf_pp", "sum"), on_ga_pp=("on_ga_pp", "sum"),
        on_gf_sh=("on_gf_sh", "sum"), on_ga_sh=("on_ga_sh", "sum"),
        tm_gf_ev=("tm_gf_ev", "sum"), tm_ga_ev=("tm_ga_ev", "sum"),
        tm_toi_ev=("tm_toi_ev", "sum"), n_teams=("team", "nunique"),
        grp=("grp", "first"), team=("team", "first"))
    P = first[["v_ev60", "v_all60", "rapm_net_v", "prior_min_ev", "cre_ev"]].join(agg)
    P = P.reset_index()
    P = P[P.season.isin([20112012] + SCORED)]
    P = P[(P.toi_ev >= 400 * 60) & (P.prior_min_ev >= 200) & P.rapm_net_v.notna()].copy()
    te = P.toi_ev / 3600.0
    P["gd5"] = (P.on_gf_ev - P.on_ga_ev) / te
    P["gf5"] = P.on_gf_ev / te
    P["ga5_neg"] = -P.on_ga_ev / te
    # off-ice team 5v5 time: team 5v5 seconds (tm_toi_ev) minus his own
    off_t = (P.tm_toi_ev - P.toi_ev) / 3600.0
    P["rel_gd5"] = P.gd5 - ((P.tm_gf_ev - P.on_gf_ev) - (P.tm_ga_ev - P.on_ga_ev)) / off_t
    ta = (P.toi_ev + P.toi_pp + P.toi_sh) / 3600.0
    P["gd_all"] = (P.on_gf_ev + P.on_gf_pp + P.on_gf_sh - P.on_ga_ev - P.on_ga_pp
                   - P.on_ga_sh) / ta
    P["w"] = te
    # team changers: first team of season s differs from last team of season s-1
    last_team = sk[sk.gtype == 2].sort_values(["date", "gid"]).groupby(
        ["season", "pid"]).team.last().reset_index()
    idx = {s: i for i, s in enumerate(C.SEASONS)}
    last_team["season_next"] = last_team.season.map(
        lambda s: C.SEASONS[idx[s] + 1] if idx[s] + 1 < len(C.SEASONS) else -1)
    P = P.merge(last_team[["season_next", "pid", "team"]].rename(
        columns={"season_next": "season", "team": "prev_team"}), on=["season", "pid"],
        how="left")
    P["mover"] = P.prev_team.notna() & (P.prev_team != P.team)
    res = {"n": int(len(P)), "n_F": int((P.grp == "F").sum()), "n_D": int((P.grp == "D").sum()),
           "sample": "regular-season player-seasons 2011-12..2017-18 with >= 400 5v5 min in "
                     "season s, >= 200 prior 5v5 min and an incumbent RAPM value; value at "
                     "his first game of s; TOI-weighted; player-cluster bootstrap"}
    targets = {"gd5": "on-ice 5v5 GD/60", "rel_gd5": "relative 5v5 GD/60 (on - off)",
               "gd_all": "on-ice GD/60 all strengths (5v5+PP+SH)",
               "gf5": "on-ice 5v5 GF/60", "ga5_neg": "minus on-ice 5v5 GA/60"}
    for grp in ("F", "D"):
        Pg = P[P.grp == grp]
        for t, lab in targets.items():
            xa = "v_all60" if t == "gd_all" else "v_ev60"
            res[f"{grp}_{t}"] = dict(label=lab, value_col=xa,
                                     **boot_corr_diff(Pg, xa, "rapm_net_v", t))
        Pm = Pg[Pg.mover]
        res[f"{grp}_movers_n"] = int(len(Pm))
        for t in ("gd5", "rel_gd5"):
            res[f"{grp}_movers_{t}"] = boot_corr_diff(Pm, "v_ev60", "rapm_net_v", t)
    return res


# ------------------------------------------------------------------ outputs ---
def lineup_values(L, W, rho=None):
    """Per team-game lineup value LV and goalie value GV with the season's weights."""
    lv = np.zeros(len(L))
    gv = np.zeros(len(L))
    for s, ws in W.items():
        mm = (L.season == s).to_numpy()
        for c in C.SK_COMPS:
            lv[mm] += ws["w"][c] * L.loc[mm, "LU_" + c].to_numpy()
        gv[mm] += ws["w"]["goalie"] * L.loc[mm, "LU_goalie"].fillna(0.0).to_numpy()
    return lv, gv


def face_lists(sk, pv, gk, st, W):
    s = sk.join(pv[["v_ev60", "v_all60", "v_game"] + ["p_" + c for c in C.SK_COMPS]])
    # regular season only: the defense component has no playoff values (it reads 0)
    s = s[(s.season == 20172018) & (s.gtype == 2)].sort_values(["date", "gid"])
    last = s.groupby("pid").tail(1).set_index("pid")
    last = last[last.prior_min_all >= 1500]
    names = pd.read_csv(D("pv_nhl_players.csv"))
    nm = dict(zip(names.pid, names["first"].astype(str) + " " + names["last"].astype(str)))
    out = {}
    cols = ["v_all60", "v_ev60"] + ["p_" + c for c in C.SK_COMPS]
    for grp in ("F", "D"):
        q = last[last.grp == grp].sort_values("v_all60", ascending=False)
        fmt = lambda d: [{"name": nm.get(p, str(p)), **{c: round(float(r[c]), 3)  # noqa: E731
                                                          for c in cols}}
                         for p, r in d.iterrows()]
        out[f"top_{grp}"] = fmt(q.head(15))
        out[f"bottom_{grp}"] = fmt(q.tail(10))
        out[f"n_{grp}"] = int(len(q))
    g = st[(st.season == 20172018) & (st.gid // 10000 % 10 == 2)].sort_values(["gid"])
    gl = g.groupby("g_pid").agg(gv=("gv", "last"), starts=("gid", "size"))
    gl = gl[gl.starts >= 20].sort_values("gv", ascending=False)
    out["goalies_top"] = [{"name": nm.get(p, str(p)), "gv_goals_per_60": round(float(r.gv), 3),
                           "starts_1718": int(r.starts)} for p, r in gl.head(10).iterrows()]
    out["goalies_bottom"] = [{"name": nm.get(p, str(p)),
                              "gv_goals_per_60": round(float(r.gv), 3),
                              "starts_1718": int(r.starts)} for p, r in gl.tail(8).iterrows()]
    out["snapshot"] = ("each skater's last pre-game value of the 2017-18 REGULAR season, "
                       "players with >= 1,500 prior minutes; goalies: value at their last "
                       "2017-18 regular-season start, >= 20 starts")
    return out


def build_features(sk, gk, tg, rho=None, adopt=False):
    """Everything a game feature depends on, from the panel (used by main and by the
    walk-forward audit). Returns F (per-game features), W, L, G, st, pv, x, m."""
    x, m, cm = C.components(sk)
    st = C.goalie_component_g2(gk, tg, rho) if adopt else C.goalie_component(gk, tg)
    L = C.lineup(sk, x, m, st)
    G = C.game_matrix(L, tg)
    W = C.walk_forward_weights(G)
    pv = C.player_values(sk, x, m, W)
    L["season"] = L.gid.map(dict(zip(sk.gid, sk.season)))
    L["LV"], L["GV"] = lineup_values(L, W)
    st = st.merge(L[["gid", "side", "GV"]], on=["gid", "side"], how="left").rename(
        columns={"GV": "gv"})
    st["season"] = st.gid.map(dict(zip(sk.gid, sk.season)))
    tm = sk[["gid", "side", "team", "date", "season", "gtype"]].drop_duplicates(["gid", "side"])
    L2 = L.merge(tm, on=["gid", "side", "season"], how="left").sort_values(["team", "date",
                                                                            "gid"])
    # lineup deviation: tonight's LV minus the mean LV of the team's previous 10
    # games (strictly earlier games; crosses seasons)
    L2["lv_prev10"] = L2.groupby("team").LV.transform(
        lambda v: v.shift(1).rolling(10, min_periods=1).mean())
    L2["lv_dev"] = L2.LV - L2.lv_prev10
    L2["gv_prev10"] = L2.groupby("team").GV.transform(
        lambda v: v.shift(1).rolling(10, min_periods=1).mean())
    L2["gv_dev"] = L2.GV - L2.gv_prev10
    wcols = []
    for c in C.SK_COMPS + ["goalie"]:
        col = "wLU_" + c
        L2[col] = 0.0
        for s, ws in W.items():
            mm = (L2.season == s).to_numpy()
            L2.loc[mm, col] = ws["w"][c] * L2.loc[mm, "LU_" + c].fillna(0.0).to_numpy()
        wcols.append(col)
    keep = ["LV", "GV", "lv_dev", "gv_dev", "g_pid", "n_sk", "team"] + wcols
    H = L2[L2.side == "H"].set_index("gid")[keep].add_suffix("_home")
    A = L2[L2.side == "A"].set_index("gid")[keep].add_suffix("_away")
    games = tg[tg.side == "H"][["gid", "date", "season", "type", "home", "away"]].set_index(
        "gid")
    F = games.join(H, how="inner").join(A, how="inner").reset_index()
    F = F.rename(columns={"LV_home": "lv_home", "LV_away": "lv_away", "GV_home": "gv_home",
                          "GV_away": "gv_away", "g_pid_home": "g_start_home",
                          "g_pid_away": "g_start_away", "type": "gtype"})
    F["def_available"] = ((F.season >= 20112012) & (F.gtype == 2)).astype(int)
    # (row order = nhl_games.csv order, unchanged from the pre-registered file)
    assert int(F.gid.max()) < DEV_MAX_GID
    return F, W, L, G, st, pv, x, m


def main():
    t0 = time.time()
    sk, gk, tg = C.load_panel()
    s0, rho, adopt = goalie_rule(gk, tg)
    print("S0", s0["decision"], s0["mse_gain_per_start"], s0["ci"], flush=True)
    F, W, L, G, st, pv, x, m = build_features(sk, gk, tg, rho, adopt)
    Wp = {s: {"b": W[s]["b"], "w": dict(C.PRIOR_W), "n": 0} for s in W}
    # descriptive all-DEV fit (NOT used for any value or feature)
    Gd = G[G.season >= C.FIRST_FIT_SEASON]
    bd, sed = C.ridge_fit(Gd[["d_" + c for c in C.COMPS]].to_numpy(float),
                          Gd.gd_noen.to_numpy(float),
                          np.array([C.PRIOR_W[c] for c in C.COMPS]),
                          np.array([C.PRIOR_TAU[c] for c in C.COMPS]))
    bo = np.linalg.lstsq(np.column_stack([np.ones(len(Gd)), Gd[["d_" + c for c in C.COMPS]]]),
                         Gd.gd_noen.to_numpy(float), rcond=None)[0]
    wjson = {"spec": "phase0/pv_nhl_compose_core.py docstring (declared before any fit)",
             "prior": {"w": C.PRIOR_W, "tau": C.PRIOR_TAU},
             "walk_forward": {str(s): {"n_train_games": v["n"], "intercept": r6(v["b"]),
                                       "w": {c: r6(val) for c, val in v["w"].items()},
                                       "se": {c: r6(val) for c, val in v.get("se", {}).items()}}
                              for s, v in W.items()},
             "descriptive_all_dev_ridge": {c: [r6(b), r6(e)] for c, b, e in
                                           zip(C.COMPS, bd[1:], sed[1:])},
             "descriptive_all_dev_ols": {c: r6(b) for c, b in zip(C.COMPS, bo[1:])},
             "component_sd_of_home_minus_away": {c: r6(G["d_" + c].std()) for c in C.COMPS},
             "goalie_rule": s0["decision"]}
    json.dump(wjson, open(OUT_W, "w"), indent=1)
    print("weights", json.dumps(wjson["walk_forward"]["20172018"]["w"]), flush=True)

    e1, G1 = e1_game(G, W, Wp)
    print("E1", json.dumps(e1["composite_vs_rapm"]), flush=True)
    e2 = e2_month(sk, x, m, pv, G, st, W)
    print("E2", json.dumps(e2["composite_vs_rapm"]), json.dumps(e2["corr"]), flush=True)
    e3 = e3_player(sk, pv)
    print("E3 F gd5", e3["F_gd5"], "D gd5", e3["D_gd5"], flush=True)
    fl = face_lists(sk, pv, gk, st, W)

    F.to_csv(OUT_GF, index=False, float_format="%.6f")
    pvo = pd.concat([sk[["gid", "date", "season", "gtype", "team", "side", "pid", "grp"]],
                     m.rename(columns={"ev": "m_ev", "pp": "m_pp", "all": "m_all"}),
                     x.add_prefix("x_"), pv], axis=1)
    pvo.to_parquet(OUT_PV, index=False)
    st[["gid", "side", "g_pid", "season", "sav_mu", "sav_sd", "lg_xg_pre", "goalie", "gv"]
       ].to_csv(OUT_GV, index=False, float_format="%.6f")

    val = {"protocol": {"dev_only": True, "test_seasons_touched": 0, "market_inputs": 0,
                        "scored_seasons": SCORED},
           "S0_goalie_rule": s0, "E1_game": e1, "E2_team_month": e2, "E3_player": e3,
           "face_validity": fl, "seconds": round(time.time() - t0, 1)}
    json.dump(val, open(OUT_V, "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
