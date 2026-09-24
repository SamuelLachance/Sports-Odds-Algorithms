"""Player-value program, NHL -- COMPOSE core: components -> lineup aggregates ->
walk-forward weights -> one value per player in goals per 60 (DEV ONLY).

DECLARED BEFORE ANY FIT (this block is the specification; nothing in it was
searched):

Skater components, each a pre-game walk-forward rate per 60 minutes of the
strength it lives in, centred on the walk-forward minutes-weighted mean of his
position group (F / D) over strictly earlier dates (decayed, half-life 30,000
player-games):
  ev_shot  ixg_ev_hat                          individual 5v5 xG / 60 (creation)
  ev_fin   ixg_ev_hat * (exp(fin_mu) - 1)      finishing, goals above xG / 60
                                               (1v1 shooter-vs-goalie logit)
  ev_a1    a1_ev_hat                           primary assists / 60 at 5v5
  ev_a2    a2_ev_hat                           secondary assists / 60 at 5v5
  ev_def   d_xga                               on-ice xG-against RAPM, xG / 60
                                               prevented (defense_onice; already
                                               relative to his position; 0 where the
                                               component has no value: 2010-11, playoffs)
  pp_xg    ixg_pp_hat * exp(fin_mu)            own expected goals / 60 on the PP
  pp_ast   a1_pp_hat + a2_pp_hat               PP assists / 60
  duels    dd_g60                              faceoffs + penalties, goals / 60 of all
                                               situations (discrete_duels headline;
                                               TK/GV excluded by its inclusion rule)
Minutes (pre-game, the player's own walk-forward usage):
  ev_*  exp_min_ev (creation; first career game: F 10.0 / D 14.0, the declared
        rapmel / defense debutant constants)
  pp_*  exp_min_pp (first game: 0)
  duels exp_toi_s / 60 (discrete_duels)
Goalie component (tonight's STARTING goalie, g_start = in net for the first
opposing shot, the confirmed starter):
  goalie   lg_xg_pre * (1 - exp(-sav_mu))      goals saved above the model's average
                                               goalie per game (1v1 saving logit;
                                               lg_xg_pre = league recalibrated xGA per
                                               team-game, strictly earlier dates)

Lineup aggregate per team-game (goals per game): LU_k = sum over tonight's
dressed skaters of x_ik * m_ik / 60; LU_goalie = the starter's value.

Weights: ridge regression toward a STRUCTURAL prior (all goal-unit components at
weight 1, assist rates at 0; PP expected goals at 1), prior SD tau = 0.5 on
goal-unit weights and 1.0 on assist weights, noise variance = the training
variance of the target:
  target  y = home goal differential, empty-net and shootout goals excluded
  design  [1, LU_k(home) - LU_k(away) for the 9 components]
  sample  DEV regular-season games of seasons 2011-12 .. s-1 (2010-11 is the
          components' warm-up and has no ev_def) -> weights w_s used for every
          game of season s. Season 2011-12 uses the prior weights (no data yet).
Player value (goals per 60 of his expected all-situations TOI):
  v_all60 = [sum_ev w_k x_k m_ev + sum_pp w_k x_k m_pp + w_duels x_duels m_tot] / m_tot
  v_ev60  = sum_ev w_k x_k                        (goals / 60 of his 5v5 time)
Lineup value LV = sum over dressed skaters of v_all60 * m_tot / 60 = sum_k w_k LU_k.
Goalie value GV = w_goalie * LU_goalie (goals per 60 = per game).

Market-blind; DEV only (every frame asserted gid < 2018000000).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

D = lambda p: os.path.join(ROOT, "data", p)  # noqa: E731

SK_COMPS = ["ev_shot", "ev_fin", "ev_a1", "ev_a2", "ev_def", "pp_xg", "pp_ast", "duels"]
COMPS = SK_COMPS + ["goalie"]
STATE = {"ev_shot": "ev", "ev_fin": "ev", "ev_a1": "ev", "ev_a2": "ev", "ev_def": "ev",
         "pp_xg": "pp", "pp_ast": "pp", "duels": "all"}
PRIOR_W = {"ev_shot": 1.0, "ev_fin": 1.0, "ev_a1": 0.0, "ev_a2": 0.0, "ev_def": 1.0,
           "pp_xg": 1.0, "pp_ast": 0.0, "duels": 1.0, "goalie": 1.0}
PRIOR_TAU = {"ev_shot": 0.5, "ev_fin": 0.5, "ev_a1": 1.0, "ev_a2": 1.0, "ev_def": 0.5,
             "pp_xg": 0.5, "pp_ast": 1.0, "duels": 0.5, "goalie": 0.5}
CENTRE = ["ev_shot", "ev_a1", "ev_a2", "pp_xg", "pp_ast", "duels"]   # d_xga, fin are
CENTRE_HL = 30000.0                                                   # already relative
DEBUT_MIN = {"F": 10.0, "D": 14.0}
FIRST_FIT_SEASON = 20112012
SEASONS = [20102011, 20112012, 20122013, 20132014, 20142015, 20152016, 20162017,
           20172018]


def assert_dev(df, name=""):
    assert len(df) and int(df.gid.max()) < DEV_MAX_GID, f"TEST gid leaked {name}"
    return df


def load_panel():
    sk = assert_dev(pd.read_parquet(D("pv_nhl_compose_skater_games.parquet")), "sk")
    gk = assert_dev(pd.read_parquet(D("pv_nhl_compose_goalie_games.parquet")), "gk")
    tg = assert_dev(pd.read_parquet(D("pv_nhl_compose_team_games.parquet")), "tg")
    return sk, gk, tg


def raw_components(sk):
    """Uncentred per-60 rates and the minutes they apply to."""
    x = pd.DataFrame(index=sk.index)
    e = np.exp(sk.fin_mu.fillna(0.0).to_numpy())
    x["ev_shot"] = sk.ixg_ev_hat.to_numpy(float)
    x["ev_fin"] = sk.ixg_ev_hat.to_numpy(float) * (e - 1.0)
    x["ev_a1"] = sk.a1_ev_hat.to_numpy(float)
    x["ev_a2"] = sk.a2_ev_hat.to_numpy(float)
    x["ev_def"] = sk.d_xga.fillna(0.0).to_numpy(float)
    x["pp_xg"] = sk.ixg_pp_hat.to_numpy(float) * e
    x["pp_ast"] = (sk.a1_pp_hat + sk.a2_pp_hat).to_numpy(float)
    x["duels"] = sk.dd_g60.to_numpy(float)
    deb = sk.grp.map(DEBUT_MIN).to_numpy(float)
    m = pd.DataFrame(index=sk.index)
    m["ev"] = np.where(sk.exp_min_ev.notna(), sk.exp_min_ev, deb)
    m["pp"] = sk.exp_min_pp.fillna(0.0).to_numpy(float)
    m["all"] = sk.exp_toi_s.to_numpy(float) / 60.0
    return x, m


def centre_means(sk, x, m):
    """Walk-forward position-group means: minutes-weighted, decayed by player-games
    (half-life CENTRE_HL), using only strictly earlier dates."""
    lam = 0.5 ** (1.0 / CENTRE_HL)
    out = {c: np.zeros(len(sk)) for c in CENTRE}
    dates = sk.date.to_numpy()
    grp = sk.grp.to_numpy()
    order = np.argsort(dates, kind="stable")
    ud, starts = np.unique(dates[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    S = {(c, g): 0.0 for c in CENTRE for g in ("F", "D")}
    W = {(c, g): 0.0 for c in CENTRE for g in ("F", "D")}
    xv = {c: x[c].to_numpy() for c in CENTRE}
    mv = {c: m[STATE[c]].to_numpy() for c in CENTRE}
    for a, b in zip(starts, ends):
        idx = order[a:b]
        for g in ("F", "D"):
            sel = idx[grp[idx] == g]
            for c in CENTRE:
                mu = S[(c, g)] / W[(c, g)] if W[(c, g)] > 0 else np.nan
                out[c][sel] = mu
        # commit today's player-games (usable from the NEXT date on)
        n_today = b - a
        dec = lam ** n_today
        for g in ("F", "D"):
            sel = idx[grp[idx] == g]
            for c in CENTRE:
                ww = mv[c][sel]
                S[(c, g)] = S[(c, g)] * dec + float(np.sum(xv[c][sel] * ww))
                W[(c, g)] = W[(c, g)] * dec + float(np.sum(ww))
    for c in CENTRE:            # first date only: no earlier data -> its own mean is
        v = out[c]              # NOT used; fall back to 0 centring (warm-up season)
        v[np.isnan(v)] = 0.0
    return pd.DataFrame(out, index=sk.index)


def components(sk):
    x, m = raw_components(sk)
    cm = centre_means(sk, x, m)
    for c in CENTRE:
        x[c] = x[c] - cm[c]
    return x, m, cm


def goalie_component(gk, tg):
    st = gk[gk.g_start == 1][["gid", "side", "pid", "sav_mu", "sav_sd"]].copy()
    st = st.merge(tg[["gid", "side", "lg_xg_pre"]], on=["gid", "side"], how="left")
    st["goalie"] = st.lg_xg_pre.fillna(2.5) * (1.0 - np.exp(-st.sav_mu.fillna(0.0)))
    return st.rename(columns={"pid": "g_pid"})


def lineup(sk, x, m, st):
    """Per team-game lineup aggregates LU_k (goals / game, pre-game)."""
    cols = {}
    for c in SK_COMPS:
        cols["LU_" + c] = x[c].to_numpy() * m[STATE[c]].to_numpy() / 60.0
    L = pd.DataFrame(cols)
    L["gid"] = sk.gid.to_numpy()
    L["side"] = sk.side.to_numpy()
    L["n_sk"] = 1
    L = L.groupby(["gid", "side"], sort=True).sum().reset_index()
    L = L.merge(st[["gid", "side", "g_pid", "goalie"]], on=["gid", "side"], how="left")
    L = L.rename(columns={"goalie": "LU_goalie"})
    return L


def game_matrix(L, tg, gtype_reg_only=True):
    """One row per game: home-minus-away lineup differences + labels."""
    H = L[L.side == "H"].set_index("gid")
    A = L[L.side == "A"].set_index("gid")
    gids = np.intersect1d(H.index.to_numpy(), A.index.to_numpy())
    G = pd.DataFrame({"gid": gids})
    for c in COMPS:
        G["d_" + c] = (H.loc[gids, "LU_" + c].to_numpy() - A.loc[gids, "LU_" + c].to_numpy())
        G["h_" + c] = H.loc[gids, "LU_" + c].to_numpy()
        G["a_" + c] = A.loc[gids, "LU_" + c].to_numpy()
    th = tg[tg.side == "H"].set_index("gid")
    G["season"] = th.loc[gids, "season"].to_numpy()
    G["date"] = th.loc[gids, "date"].to_numpy()
    G["gtype"] = th.loc[gids, "type"].to_numpy()
    G["home"] = th.loc[gids, "home"].to_numpy()
    G["away"] = th.loc[gids, "away"].to_numpy()
    for c in ("gd_noen", "gf_noen", "ga_noen", "gd_s5", "gd_st", "gf_s5", "ga_s5"):
        G[c] = th.loc[gids, c].to_numpy(float)
    if gtype_reg_only:
        G = G[G.gtype == 2]
    return assert_dev(G.sort_values(["date", "gid"]).reset_index(drop=True), "G")


def ridge_fit(X, y, w0, tau, sigma2=None):
    """argmin ||y - b - Xw||^2 / s2 + sum_k (w_k - w0_k)^2 / tau_k^2 (b unpenalised)."""
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X])
    s2 = float(np.var(y)) if sigma2 is None else sigma2
    P = np.zeros((k + 1, k + 1))
    P[1:, 1:] = np.diag(1.0 / np.asarray(tau) ** 2)
    m0 = np.r_[0.0, w0]
    A = Z.T @ Z / s2 + P
    rhs = Z.T @ y / s2 + P @ m0
    beta = np.linalg.solve(A, rhs)
    cov = np.linalg.inv(A)
    return beta, np.sqrt(np.diag(cov))


def walk_forward_weights(G, comps=COMPS, target="gd_noen", prior=None, tau=None,
                         seasons=SEASONS):
    """weights for season s fit on regular-season games of FIRST_FIT_SEASON..s-1."""
    prior = PRIOR_W if prior is None else prior
    tau = PRIOR_TAU if tau is None else tau
    w0 = np.array([prior[c] for c in comps])
    t = np.array([tau[c] for c in comps])
    out = {}
    for s in seasons:
        tr = (G.season >= FIRST_FIT_SEASON) & (G.season < s)
        if s <= FIRST_FIT_SEASON or tr.sum() == 0:
            out[s] = {"b": float(G[G.season < s][target].mean()) if (G.season < s).any()
                      else 0.25, "w": dict(zip(comps, w0.tolist())), "n": 0,
                      "se": dict(zip(comps, t.tolist()))}
            continue
        X = G.loc[tr, ["d_" + c for c in comps]].to_numpy(float)
        y = G.loc[tr, target].to_numpy(float)
        beta, se = ridge_fit(X, y, w0, t)
        assert int(G.loc[tr, "season"].max()) < s          # walk-forward guard
        out[s] = {"b": float(beta[0]), "w": dict(zip(comps, beta[1:].tolist())),
                  "n": int(tr.sum()), "se": dict(zip(comps, se[1:].tolist()))}
    return out


def apply_weights(G, W, comps=COMPS, prefix="d_"):
    pred = np.zeros(len(G))
    for s, ws in W.items():
        m = (G.season == s).to_numpy()
        if not m.any():
            continue
        v = np.full(m.sum(), ws["b"])
        for c in comps:
            v = v + ws["w"][c] * G.loc[m, prefix + c].to_numpy(float)
        pred[m] = v
    return pred


def player_values(sk, x, m, W):
    """Per skater-game walk-forward values in goals / 60 using the season's weights."""
    ev = [c for c in SK_COMPS if STATE[c] == "ev"]
    pp = [c for c in SK_COMPS if STATE[c] == "pp"]
    season = sk.season.to_numpy()
    v_ev60 = np.zeros(len(sk))
    v_pp60 = np.zeros(len(sk))
    v_dd60 = np.zeros(len(sk))
    parts = {c: np.zeros(len(sk)) for c in SK_COMPS}
    for s, ws in W.items():
        msk = season == s
        for c in SK_COMPS:
            parts[c][msk] = ws["w"][c] * x[c].to_numpy()[msk]
    for c in ev:
        v_ev60 += parts[c]
    for c in pp:
        v_pp60 += parts[c]
    v_dd60 = parts["duels"]
    mtot = m["all"].to_numpy()
    per_game = (v_ev60 * m["ev"].to_numpy() + v_pp60 * m["pp"].to_numpy()
                + v_dd60 * mtot) / 60.0
    out = pd.DataFrame({"v_ev60": v_ev60, "v_pp60": v_pp60, "v_dd60": v_dd60,
                        "v_game": per_game, "v_all60": per_game / mtot * 60.0},
                       index=sk.index)
    for c in SK_COMPS:
        out["p_" + c] = parts[c]
    return out


# ------------------------------------------------------------------ goalie ---
# DECLARED variant G2 (experience prior), evaluated at the GOALS level before any
# game-level screen: a goalie's 1v1 saving logit is shrunk toward 0 (the average
# goalie) whatever his experience, so a thin-history backup reads as average. G2
# adds a walk-forward offset rho_b by experience bucket b of his career prior
# recalibrated xGA faced (event archive, all appearances, cumulative from 2010-11:
# left-censored for veterans in 2010-11, which is never scored):
#     rho_b(s) = sum(resid) / (sum(xGA) + K)    over starts of seasons 2011-12..s-1
# resid = realised goals saved above recalibrated xG minus the base prediction
# xGA_rc * (1 - exp(-mu)); K = 232 xGA (finishing_and_saving's DEV split-half EB
# constant for GSAx). Season 2011-12 (no earlier scored season): rho = 0.
# Adoption rule (declared): G2 replaces the base goalie value iff its walk-forward
# per-start squared-error improvement over 2012-13..2017-18 has a game-bootstrap
# 95% CI above zero.
G_BUCKETS = [0.0, 25.0, 75.0, 150.0, 300.0, np.inf]
G_K = 232.0


def goalie_experience(gk):
    g = gk.sort_values(["gi", "side", "pid"]).copy()
    # prior career xGA faced: strictly earlier games (a goalie plays once per game)
    g["cum_xrc"] = g.groupby("pid").fa_xrc.cumsum() - g.fa_xrc
    g["bucket"] = np.searchsorted(G_BUCKETS, g.cum_xrc.to_numpy(), side="right") - 1
    return g


def goalie_offsets(g, seasons=SEASONS):
    """rho[s][bucket] from starts of seasons FIRST_FIT_SEASON..s-1 (walk-forward)."""
    st = g[(g.g_start == 1) & (g.gtype == 2)].copy()
    st["pred"] = st.fa_xrc * (1.0 - np.exp(-st.sav_mu))
    st["real"] = st.fa_xrc - st.fa_g
    st["resid"] = st.real - st.pred
    rho = {}
    for s in seasons:
        tr = st[(st.season >= FIRST_FIT_SEASON) & (st.season < s)]
        r = {}
        for b in range(len(G_BUCKETS) - 1):
            sb = tr[tr.bucket == b]
            r[b] = float(sb.resid.sum() / (sb.fa_xrc.sum() + G_K)) if len(sb) else 0.0
        if len(tr):
            assert int(tr.season.max()) < s
        rho[s] = r
    return rho


def goalie_component_g2(gk, tg, rho):
    g = goalie_experience(gk)
    st = g[g.g_start == 1][["gid", "side", "pid", "season", "sav_mu", "sav_sd",
                            "bucket", "cum_xrc"]].copy()
    st = st.merge(tg[["gid", "side", "lg_xg_pre"]], on=["gid", "side"], how="left")
    off = np.array([rho.get(s, {}).get(b, 0.0) for s, b in zip(st.season, st.bucket)])
    st["rho"] = off
    st["goalie"] = st.lg_xg_pre.fillna(2.5) * (1.0 - np.exp(-(st.sav_mu.fillna(0.0) + off)))
    return st.rename(columns={"pid": "g_pid"})
