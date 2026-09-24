"""Player-level validation of the NHL finishing_and_saving component (DEV ONLY).

Reads the walk-forward outputs of phase0/pv_nhl_finishing_and_saving.py and
answers, on DEV seasons 2010-11..2017-18 only (scored 2011-12..2017-18):

  R  RELIABILITY -- split-half (odd/even appearances within a season) and
     year-over-year correlations, with n:
       goalies : raw GSAx/xGA (MoneyPuck xG) | GSAx vs recalibrated xG |
                 1v1 static ridge fit on each half, players only | 1v1 + team/rink
       shooters: the same four
  P  PREDICTIVE VALIDITY -- walk-forward, future GOALS (never xG):
       P1 shot-level prequential log-loss of goal/no-goal from pre-game states
          (the proper score): MoneyPuck xG | recalibrated | context | +goalie |
          +shooter | full; goalie baselines raw career GSAx and round-1 EB GSAx
          (K=400 xGA, d=0.998, season x0.85) plugged into the same baselines;
          shooter baseline EB goals-above-expected with the DEV split-half K
       P2 goalie-season: rating FROZEN at season start (and at Jan-1 for the rest of
          the season) vs realised GSAx/xGA over the frozen window
       P3 shooter-season: rating frozen at season start vs realised goals above xG
       P4 future goals per game: season-s goals/GP forecast at season start from
          past ixG/GP [xG-only assumption] vs past goals/GP vs ixG/GP x finishing
       P5 unit level: team-game realised (GF - xGF) and (xGA - GA) vs the pre-game
          lineup values; team goal DIFFERENTIAL on the shipped xG team rating + Elo
          +/- the lineup value
  F  FACE VALIDITY -- top/bottom goalies, shooters, rinks at the end of DEV
  W  WALK-FORWARD AUDIT -- outcomes of every shot on/after a cutoff date are
     permuted (and 1/7 flipped), the rating is rebuilt, and every pre-game value
     before (and on) the cutoff date must be bitwise identical.

Market-blind: no odds anywhere.  Nothing from a TEST season is loaded
(pv_nhl_io asserts gid < 2018000000; every frame is re-asserted here).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pv_nhl_finishing_and_saving as FS  # noqa: E402
from pv_nhl_io import DEV_MAX_GID  # noqa: E402

OUT = os.path.join(ROOT, "data/pv_nhl_finishing_and_saving_validation.json")
EVAL_FROM = FS.EVAL_FROM
sig = FS.sig


def lg(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def wcorr(x, y, w=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cx, cy = x - mx, y - my
    return float(np.sum(w * cx * cy) / np.sqrt(np.sum(w * cx * cx) * np.sum(w * cy * cy)))


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def ci_r(r, n):
    """Fisher-z 95% CI for a correlation."""
    if r is None or n < 4:
        return None
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return [round(float(np.tanh(z - 1.96 * se)), 3), round(float(np.tanh(z + 1.96 * se)), 3)]


def boot_sum(d, nb=4000, seed=7):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    bs = np.empty(nb)
    for i in range(0, nb, 500):
        idx = rng.integers(0, len(d), size=(min(500, nb - i), len(d)))
        bs[i:i + idx.shape[0]] = d[idx].sum(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ------------------------------------------------------ static 1v1 fits ---
def fit_static(df, cfg, ctx):
    """Ridge-logistic 1v1 on a shot subset: logit p = z_rc + a_shooter - b_goalie
    [+ o_team - d_team + k_rink if ctx], priors N(0, S0^2) from the DEV-tuned cfg.
    Returns the goalie and shooter effect Series."""
    n = len(df)
    rows = np.arange(n)
    blocks, precs = [], []
    out_idx = {}
    specs = [("sh", df.shooter.values, 1.0, None), ("go", df.goalie.values, -1.0, "G")]
    if ctx:
        specs += [("to", df.team.values, 1.0, "TO"), ("td", df.def_team.values, -1.0, "TD"),
                  ("rk", df.rink.values, 1.0, "RK")]
    off = 0
    for name, keys, sgn, g in specs:
        codes, uniq = pd.factorize(keys)
        blocks.append(sparse.csr_matrix((np.full(n, sgn), (rows, codes)), shape=(n, len(uniq))))
        if name == "sh":
            isd = df.drop_duplicates("shooter").set_index("shooter").is_d.reindex(uniq).values
            precs.append(np.where(isd.astype(bool), 1 / cfg["D"][0] ** 2, 1 / cfg["F"][0] ** 2))
        else:
            precs.append(np.full(len(uniq), 1 / cfg[g][0] ** 2))
        out_idx[name] = (off, uniq)
        off += len(uniq)
    X = sparse.hstack(blocks).tocsr()
    prec = np.concatenate(precs)
    z0 = df.z_rc.values
    y = df.y.values.astype(float)

    def f(w):
        z = z0 + X @ w
        L = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(prec * w * w)
        g = X.T @ (sig(z) - y) + prec * w
        return L, g

    w = minimize(f, np.zeros(off), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=1000, gtol=1e-7)).x
    o, u = out_idx["go"]
    go = pd.Series(w[o:o + len(u)], index=u)
    o, u = out_idx["sh"]
    sh = pd.Series(w[o:o + len(u)], index=u)
    return go, sh


def season_halves(sh, pg):
    ap = pg[["gi", "pid", "season"]].sort_values("gi").copy()
    ap["k"] = ap.groupby(["pid", "season"]).cumcount()
    par = pd.Series((ap.k % 2).values, index=pd.MultiIndex.from_arrays([ap.gi, ap.pid]))
    sh = sh.copy()
    sh["h_go"] = par.reindex(pd.MultiIndex.from_arrays([sh.gi, sh.goalie])).values
    sh["h_sh"] = par.reindex(pd.MultiIndex.from_arrays([sh.gi, sh.shooter])).values
    return sh


def _metrics(d, key, sgn, fits):
    agg = d.groupby(key).agg(x=("xg", "sum"), xr=("p_rc", "sum"), g=("y", "sum"))
    agg["raw"] = sgn * (agg.x - agg.g) / agg.x
    agg["rc"] = sgn * (agg.xr - agg.g) / agg.xr
    for nm, (go, shs) in fits.items():
        agg[nm] = (go if key == "goalie" else shs).reindex(agg.index)
    return agg


def _rel(m, mins, cols=("raw", "rc", "adj", "adj_ctx")):
    out = {}
    for mn in mins:
        d = m[(m.xr_a >= mn) & (m.xr_b >= mn)]
        w = 1.0 / (1.0 / d.xr_a + 1.0 / d.xr_b)
        res = {"n": int(len(d))}
        for c in cols:
            r = wcorr(d[f"{c}_a"], d[f"{c}_b"])
            res[c] = dict(r=r4(r), ci95=ci_r(r, len(d)), r_w=r4(wcorr(d[f"{c}_a"], d[f"{c}_b"], w)))
        out[f"min_xg_{mn}"] = res
    return out


def reliability(sh, pg, cfg):
    t0 = time.time()
    sh = season_halves(sh, pg)
    split = {"goalie": [], "shooter": []}
    seasons = sorted(sh.season.unique())
    for s in seasons:
        d = sh[sh.season == s]
        for key, hcol, sgn in (("goalie", "h_go", 1), ("shooter", "h_sh", -1)):
            halves = []
            for h in (0, 1):
                dh = d[d[hcol] == h]
                fits = {"adj": fit_static(dh, cfg, False), "adj_ctx": fit_static(dh, cfg, True)}
                halves.append(_metrics(dh, key, sgn, fits))
            m = halves[0].join(halves[1], lsuffix="_a", rsuffix="_b", how="inner")
            m["season"] = s
            split[key].append(m)
        print(f"  split-half {s} ({time.time()-t0:.0f}s)", flush=True)
    res = {"goalie": {}, "shooter": {}}
    for key, mins in (("goalie", (25, 50)), ("shooter", (3, 6))):
        res[key]["split_half"] = _rel(pd.concat(split[key]), mins)
    fits = {}
    for s in seasons:
        d = sh[sh.season == s]
        f = {"adj": fit_static(d, cfg, False), "adj_ctx": fit_static(d, cfg, True)}
        for key, sgn in (("goalie", 1), ("shooter", -1)):
            a = _metrics(d, key, sgn, f)
            if key == "goalie":
                tm = d.groupby("goalie").def_team.agg(lambda x: x.value_counts().index[0])
            else:
                tm = d.groupby("shooter").team.agg(lambda x: x.value_counts().index[0])
            a["team"] = tm.reindex(a.index)
            fits[(key, s)] = a
    for key, mins in (("goalie", (50, 100)), ("shooter", (5, 10))):
        rows = []
        for a_, b_ in zip(seasons[:-1], seasons[1:]):
            rows.append(fits[(key, a_)].join(fits[(key, b_)], lsuffix="_a", rsuffix="_b",
                                             how="inner"))
        m = pd.concat(rows)
        res[key]["year_over_year"] = _rel(m, mins)
        res[key]["year_over_year_changed_team"] = _rel(m[m.team_a != m.team_b], mins[:1])
        res[key]["year_over_year_same_team"] = _rel(m[m.team_a == m.team_b], mins[:1])
    # teammate goalies, same season: how much of "goalie GSAx" is the team?
    tg = []
    for s in seasons:
        a = fits[("goalie", s)]
        a = a[a.xr >= 40].reset_index().rename(columns={"index": "goalie"})
        p = a.merge(a, on="team", suffixes=("_a", "_b"))
        tg.append(p[p.goalie_a < p.goalie_b])
    p = pd.concat(tg)
    res["goalie"]["teammate_goalies_same_season"] = {
        "n_pairs": int(len(p)),
        **{c: dict(r=r4(wcorr(p[f"{c}_a"], p[f"{c}_b"])),
                   ci95=ci_r(wcorr(p[f"{c}_a"], p[f"{c}_b"]), len(p)))
           for c in ("raw", "rc", "adj", "adj_ctx")}}
    for key, mn in (("goalie", 25), ("shooter", 3)):
        m = pd.concat(split[key])
        d = m[(m.xr_a >= mn) & (m.xr_b >= mn)]
        r = wcorr(d.rc_a, d.rc_b)
        xh = float(np.median(np.r_[d.xr_a.values, d.xr_b.values]))
        res[key]["implied_EB_K_xg_from_split_half_rc"] = dict(
            r=r4(r), median_half_xg=round(xh, 1), K=round(xh * (1 - r) / r, 1) if r > 0 else None)
    return res


# ----------------------------------------------- walk-forward baselines ---
def goalie_baselines(pg):
    """Pre-game raw career GSAx/xGA and round-1 EB GSAx (construction G of
    bt_nhl_gmar_gl: N,X decayed 0.998/appearance, x0.85 per season, K=400, m=0) for
    every goalie row, from strictly earlier appearances, against three baselines:
    MoneyPuck xG (round 1), recalibrated xG, and the full context baseline."""
    g = pg[pg.grp == "G"].sort_values("gi").copy()
    bases = {"raw": g.fa_xg.values, "rc": g.fa_xrc.values, "ctx": g.fa_xctx.values}
    out = {f"{k}_{b}": np.zeros(len(g)) for k in ("eb", "cum") for b in bases}
    st = {}
    ga = g.fa_g.values
    for i, (pid, season, played) in enumerate(zip(g.pid.values, g.season.values,
                                                  g.fa_n.values > 0)):
        s = st.get(pid)
        if s is None:
            s = st[pid] = {"season": season, **{f"{k}{b}": 0.0 for k in ("N", "X", "cN", "cX")
                                                 for b in bases}}
        if s["season"] != season:
            for b in bases:
                s[f"N{b}"] *= 0.85
                s[f"X{b}"] *= 0.85
            s["season"] = season
        for b in bases:
            out[f"eb_{b}"][i] = s[f"N{b}"] / (s[f"X{b}"] + 400.0)
            out[f"cum_{b}"][i] = s[f"cN{b}"] / s[f"cX{b}"] if s[f"cX{b}"] > 0 else 0.0
        if played:
            for b, x in bases.items():
                s[f"N{b}"] = 0.998 * s[f"N{b}"] + x[i] - ga[i]
                s[f"X{b}"] = 0.998 * s[f"X{b}"] + x[i]
                s[f"cN{b}"] += x[i] - ga[i]
                s[f"cX{b}"] += x[i]
    for k, v in out.items():
        g[k] = v
    return g


def shooter_baselines(pg):
    s = pg[pg.grp != "G"].sort_values("gi").copy()
    for c, src in (("cum_x", "sh_xrc"), ("cum_xc", "sh_xctx"), ("cum_g", "sh_g"), ("cum_n", "sh_n")):
        s[c] = s.groupby("pid")[src].cumsum() - s[src]
    return s


# --------------------------------------------------------------- P1 shots ---
def shot_ll(sh, gb, sb, K_sh):
    y = sh.y.values.astype(float)
    zr = sh.z_rc.values
    zc = zr + sh.t_to.values + sh.t_td.values + sh.t_rk.values
    gk = pd.MultiIndex.from_arrays([sh.gi, sh.goalie])
    gm = gb.set_index(["gi", "pid"])

    def goff(col):
        r = np.clip(gm[col].reindex(gk).values, -0.6, 0.6)
        return np.log1p(-r)

    sk = pd.MultiIndex.from_arrays([sh.gi, sh.shooter])
    sm = sb.set_index(["gi", "pid"])
    cx = sm.cum_x.reindex(sk).values
    cxc = sm.cum_xc.reindex(sk).values
    cg = sm.cum_g.reindex(sk).values
    s_eb = np.log1p(np.clip((cg - cx) / (cx + K_sh), -0.9, 3.0))
    s_ebc = np.log1p(np.clip((cg - cxc) / (cxc + K_sh), -0.9, 3.0))
    s_raw = np.log1p(np.where(cx > 0, np.clip((cg - cx) / np.maximum(cx, 1e-9), -0.9, 3.0), 0))
    arms = {
        "moneypuck_xg": lg(sh.xg.values),
        "recalibrated (base)": zr,
        "rc+goalie raw career GSAx": zr + goff("cum_rc"),
        "rc+goalie EB GSAx (round-1, MoneyPuck xG)": zr + goff("eb_raw"),
        "rc+goalie EB GSAx (recal xG)": zr + goff("eb_rc"),
        "rc+goalie 1v1": zr + sh.t_go.values,
        "rc+shooter raw career": zr + s_raw,
        "rc+shooter EB": zr + s_eb,
        "rc+shooter 1v1": zr + sh.t_sh.values,
        "rc+shooter EB+goalie EB": zr + s_eb + goff("eb_rc"),
        "rc+shooter 1v1+goalie 1v1": zr + sh.t_sh.values + sh.t_go.values,
        "context (rc+team+rink)": zc,
        "context+goalie EB GSAx (ctx xG)": zc + goff("eb_ctx"),
        "context+goalie 1v1": zc + sh.t_go.values,
        "context+shooter EB (ctx xG)": zc + s_ebc,
        "context+shooter 1v1": zc + sh.t_sh.values,
        "context+EB+EB": zc + s_ebc + goff("eb_ctx"),
        "FULL 1v1 (context+shooter+goalie)": zc + sh.t_sh.values + sh.t_go.values,
    }
    L = {k: FS.ll(y, sig(z)) for k, z in arms.items()}
    gcode = pd.factorize(sh.gid.values)[0]
    seas = sh.season.values
    windows = {"all_dev_2011_18": seas >= EVAL_FROM,
               "tune_2011_15": (seas >= EVAL_FROM) & (seas <= FS.TUNE_TO),
               "check_2015_18": seas > FS.TUNE_TO}
    pairs = [("rc+goalie 1v1", "rc+goalie EB GSAx (recal xG)"),
             ("rc+shooter 1v1", "rc+shooter EB"),
             ("rc+shooter 1v1+goalie 1v1", "rc+shooter EB+goalie EB"),
             ("context+goalie 1v1", "context+goalie EB GSAx (ctx xG)"),
             ("context+goalie 1v1", "context (rc+team+rink)"),
             ("FULL 1v1 (context+shooter+goalie)", "context+EB+EB"),
             ("FULL 1v1 (context+shooter+goalie)", "context+shooter 1v1"),
             ("context (rc+team+rink)", "recalibrated (base)")]
    out = {}
    base = L["recalibrated (base)"]
    for wn, w in windows.items():
        gw = gcode[w]
        ng = np.bincount(gw)
        keep = ng > 0
        tab = {"_n_shots": int(w.sum()), "_n_games": int(keep.sum())}
        for k, l in L.items():
            d = np.bincount(gw, weights=(base - l)[w])[keep]
            ci = boot_sum(d)
            tab[k] = dict(ll=round(float(l[w].mean()), 7),
                          gain_vs_base_nats=round(float(d.sum()), 1),
                          ci95=[round(ci[0], 1), round(ci[1], 1)])
        for a, b in pairs:
            d = np.bincount(gw, weights=(L[b] - L[a])[w])[keep]
            ci = boot_sum(d)
            tab[f"PAIR {a}  vs  {b}"] = dict(gain_nats=round(float(d.sum()), 1),
                                             ci95=[round(ci[0], 1), round(ci[1], 1)])
        out[wn] = tab
    return out


# --------------------------------------------------- P2/P3 frozen checks ---
def goalie_frozen(gb, sh, cutoff=None):
    """Rating frozen at the goalie's first game of the season (cutoff=None) or at his
    first game on/after MM-DD `cutoff` in that season; realised GSAx/xGA over all his
    games from that game to the end of the season (rc baseline, and ctx baseline)."""
    g = gb[gb.season >= EVAL_FROM].copy()
    g["md"] = g.date.str[5:]
    if cutoff is not None:
        yr2 = g.season % 10000
        g = g[g.date >= (yr2.astype(str) + "-" + cutoff)]
    g = g.sort_values("gi")
    first = g.groupby(["pid", "season"]).head(1).set_index(["pid", "season"])
    agg = g[g.fa_n > 0].groupby(["pid", "season"]).agg(
        xr=("fa_xrc", "sum"), xc=("fa_xctx", "sum"), ga=("fa_g", "sum"), gp=("gi", "size"))
    t = agg.join(first[["mu", "eb_raw", "eb_rc", "eb_ctx", "cum_rc"]], how="inner")
    t["real_rc"] = (t.xr - t.ga) / t.xr
    t["real_ctx"] = (t.xc - t.ga) / t.xc
    t["p_1v1"] = 1 - np.exp(-t.mu)
    t["cum_rc"] = t.cum_rc.clip(-0.6, 0.6)
    out = {}
    for mn in (40, 80):
        d = t[t.xr >= mn]
        res = {"n_goalie_seasons": int(len(d))}
        for c in ("p_1v1", "eb_raw", "eb_rc", "eb_ctx", "cum_rc"):
            r = wcorr(d[c], d.real_rc, d.xr)
            res[c] = dict(r_w_vs_real_rc=r4(r), ci95=ci_r(r, len(d)),
                          r_w_vs_real_ctx=r4(wcorr(d[c], d.real_ctx, d.xr)),
                          slope_w=r4(np.polyfit(d[c], d.real_rc, 1, w=np.sqrt(d.xr))[0])
                          if d[c].std() > 0 else None,
                          mse_w_x1e4=r4(np.average((d.real_rc - d[c]) ** 2, weights=d.xr) * 1e4))
        res["mse_w_x1e4_predict_zero"] = r4(np.average(d.real_rc ** 2, weights=d.xr) * 1e4)
        out[f"min_xga_{mn}"] = res
    return out


def shooter_frozen(sh, pg, sb, K_sh):
    s = sh[sh.season >= EVAL_FROM]
    sk = pg[pg.grp != "G"].sort_values("gi")
    first = sk.groupby(["pid", "season"]).head(1).set_index(["pid", "season"]).mu
    fs = sb.sort_values("gi").groupby(["pid", "season"]).head(1).set_index(["pid", "season"])
    eb0 = ((fs.cum_g - fs.cum_x) / (fs.cum_x + K_sh)).clip(-0.9, 3.0)
    raw0 = np.where(fs.cum_x > 0, ((fs.cum_g - fs.cum_x) / fs.cum_x.clip(lower=1e-9)).clip(-0.9, 3.0), 0)
    raw0 = pd.Series(raw0, index=fs.index)
    key = pd.MultiIndex.from_arrays([s.shooter, s.season])
    mu0 = first.reindex(key).values
    zr = s.z_rc.values
    t = pd.DataFrame(dict(
        shooter=s.shooter.values, season=s.season.values, y=s.y.values, prc=s.p_rc.values,
        p_1v1=sig(zr + mu0) - s.p_rc.values,
        p_eb=s.p_rc.values * eb0.reindex(key).values,
        p_raw=s.p_rc.values * raw0.reindex(key).values))
    t["real"] = t.y - t.prc
    a = t.groupby(["shooter", "season"]).agg(n=("y", "size"), real=("real", "sum"),
                                             p_1v1=("p_1v1", "sum"), p_eb=("p_eb", "sum"),
                                             p_raw=("p_raw", "sum"))
    out = {}
    for mn in (50, 100, 150):
        d = a[a.n >= mn]
        res = {"n_player_seasons": int(len(d)),
               "rmse_goals_xg_only": r4(np.sqrt(np.mean(d.real ** 2)))}
        for c in ("p_1v1", "p_eb", "p_raw"):
            r = wcorr(d[c] / d.n, d.real / d.n, d.n)
            res[c] = dict(r_w=r4(r), ci95=ci_r(r, len(d)),
                          rmse_goals=r4(np.sqrt(np.mean((d.real - d[c]) ** 2))),
                          slope=r4(np.polyfit(d[c], d.real, 1)[0]))
        out[f"min_shots_{mn}"] = res
    return out


def future_goals(pg, K_sh):
    """P4: season-s goals per game forecast at season start from strictly earlier
    seasons (walk-forward at the season boundary)."""
    sk = pg[(pg.grp != "G")].sort_values("gi")
    ps = sk.groupby(["pid", "season"]).agg(gp=("gi", "size"), x=("sh_xrc", "sum"),
                                           g=("sh_g", "sum"), mu0=("mu", "first"),
                                           grp=("grp", "first")).reset_index()
    ps = ps.sort_values(["pid", "season"])
    rows = []
    for pid, d in ps.groupby("pid"):
        cx = cg = cgp = 0.0
        for r in d.itertuples(index=False):
            if cgp >= 20 and r.season >= EVAL_FROM and r.gp >= 40:
                rows.append(dict(pid=pid, season=r.season, grp=r.grp, gp=r.gp,
                                 g_pg=r.g / r.gp, xg_rate=cx / cgp, g_rate=cg / cgp,
                                 fin_eb=(cg - cx) / (cx + K_sh), mu0=r.mu0))
            cx = 0.7 * cx + r.x
            cg = 0.7 * cg + r.g
            cgp = 0.7 * cgp + r.gp
    t = pd.DataFrame(rows)
    preds = {"xG only (past ixG/GP)": t.xg_rate, "past goals/GP": t.g_rate,
             "ixG/GP x exp(mu 1v1)": t.xg_rate * np.exp(t.mu0),
             "ixG/GP x (1 + EB finishing)": t.xg_rate * (1 + t.fin_eb)}
    out = {"n_player_seasons": int(len(t)), "min_gp": 40, "min_prior_gp_decayed": 20}
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(t), size=(2000, len(t)))
    base = preds["xG only (past ixG/GP)"].values
    yv = t.g_pg.values
    for k, p in preds.items():
        p = p.values
        a = float(np.sum(p * yv) / np.sum(p * p))       # one through-origin scale per arm
        e = yv - a * p
        res = dict(r=r4(wcorr(p, yv)), rmse=r4(np.sqrt(np.mean(e ** 2))), scale=r4(a))
        if k != "xG only (past ixG/GP)":
            # paired bootstrap of the correlation gain over xG-only
            dr = []
            for ii in idx[:1000]:
                dr.append(np.corrcoef(p[ii], yv[ii])[0, 1] - np.corrcoef(base[ii], yv[ii])[0, 1])
            res["r_gain_vs_xg_only_ci95"] = [round(float(np.percentile(dr, 2.5)), 4),
                                             round(float(np.percentile(dr, 97.5)), 4)]
        out[k] = res
    for grp in ("F", "D"):
        m = (t.grp == grp).values
        out[f"r_by_group_{grp} (n={int(m.sum())})"] = {k: r4(wcorr(p.values[m], yv[m]))
                                                        for k, p in preds.items()}
    return out


# ------------------------------------------------------------ P5 unit ------
def unit_validity(tg):
    t = tg[tg.season >= EVAL_FROM]
    n = len(t)
    out = {"n_team_games": int(n)}
    for a, b in (("fin_pre", "fin_real"), ("sav_pre", "sav_real")):
        r = wcorr(t[a], t[b])
        out[f"corr_{a}_vs_{b}"] = dict(r=r4(r), ci95=ci_r(r, n),
                                        slope=r4(np.polyfit(t[a], t[b], 1)[0]))
    tot = t.fin_real + t.sav_real
    for a in ("fs_pre", "fs_env_pre"):
        r = wcorr(t[a], tot)
        out[f"corr_{a}_vs_fs_real"] = dict(r=r4(r), ci95=ci_r(r, n),
                                            slope=r4(np.polyfit(t[a], tot, 1)[0]))
    for c in ("fs_pre", "fin_pre", "sav_pre", "fs_env_pre"):
        out[f"sd_{c}_goals"] = r4(t[c].std())
    return out


def goal_diff_increment(tg):
    import nhl_depth_eval as Hd
    games = Hd.dev_games()
    xg = Hd.run_xg_arr(games, *Hd.SHIPPED_XG)
    pe = np.clip(Hd.run_elo_arr(games, *Hd.SHIPPED_ELO), 1e-9, 1 - 1e-9)
    gid = np.array([g["game_id"] for g in games])
    seas = np.array([g["season"] for g in games])
    h = tg[tg.side_h == 1].set_index("gid")
    a = tg[tg.side_h == 0].set_index("gid")
    ok = np.isin(gid, h.index) & np.isin(gid, a.index) & ~np.isnan(xg) & (seas >= EVAL_FROM)
    gd = (h.gf - a.gf).reindex(gid[ok]).values
    col = {c: (h[c] - a[c]).reindex(gid[ok]).values
           for c in ("fs_pre", "fin_pre", "sav_pre", "fs_env_pre")}
    X0 = np.column_stack([np.ones(ok.sum()), xg[ok], lg(pe[ok])])
    out = {"n_games": int(ok.sum()), "target": "home - away goals on unblocked shots at a goalie in net (excl. empty-net and shootout)",
           "columns": "const, shipped xG-Elo diff, shipped Elo logit, [extras]"}
    for name, extra in (("base", []), ("+fs_diff", ["fs_pre"]),
                        ("+fin_diff+sav_diff", ["fin_pre", "sav_pre"]),
                        ("+fs_env_diff (incl. team terms)", ["fs_env_pre"])):
        X = np.column_stack([X0] + [col[c] for c in extra]) if extra else X0
        w, *_ = np.linalg.lstsq(X, gd, rcond=None)
        e = gd - X @ w
        s2 = e @ e / (X.shape[0] - X.shape[1])
        cov = s2 * np.linalg.inv(X.T @ X)
        out[name] = dict(rmse=round(float(np.sqrt(np.mean(e ** 2))), 5),
                         coefs=[round(float(c), 4) for c in w],
                         t_stats=[round(float(c / np.sqrt(cov[i, i])), 2) for i, c in enumerate(w)])
    return out


# ------------------------------------------------------------- face ---------
def face(final, pg, names):
    nm = names.set_index("pid")
    tot = pg.groupby("pid").agg(fa_n=("fa_n", "sum"), fa_x=("fa_xrc", "sum"), fa_g=("fa_g", "sum"),
                                sh_n=("sh_n", "sum"), sh_x=("sh_xrc", "sum"), sh_g=("sh_g", "sum"),
                                last_season=("season", "max"))
    f = final[final.pid >= 0].set_index("pid").join(tot, how="inner")
    f["name"] = [f"{nm['first'].get(p, '')} {nm['last'].get(p, '')}".strip() for p in f.index]
    out = {}
    g = f[(f.grp == "G") & (f.fa_n >= 3000) & (f.last_season >= 20162017)].copy()
    g["dev_gsax_rc_per_xga"] = (g.fa_x - g.fa_g) / g.fa_x
    g["dev_sv_pct_unblocked"] = 1 - g.fa_g / g.fa_n
    cols = ["name", "mu", "sd", "fa_n", "dev_gsax_rc_per_xga"]
    out["goalies_n (>=3000 unblocked faced, active 2016-18)"] = int(len(g))
    out["goalies_top10"] = g.sort_values("mu", ascending=False)[cols].head(10).round(4).to_dict("records")
    out["goalies_bottom10"] = g.sort_values("mu")[cols].head(10).round(4).to_dict("records")
    out["goalies_corr_mu_vs_dev_career_gsax"] = r4(wcorr(g.mu, g.dev_gsax_rc_per_xga))
    for grp in ("F", "D"):
        s = f[(f.grp == grp) & (f.sh_n >= 600) & (f.last_season >= 20162017)].copy()
        s["dev_gax_per_xg"] = (s.sh_g - s.sh_x) / s.sh_x
        cols = ["name", "mu", "sd", "sh_n", "sh_g", "sh_x", "dev_gax_per_xg"]
        out[f"shooters_{grp}_n (>=600 shots, active 2016-18)"] = int(len(s))
        out[f"shooters_{grp}_top10"] = s.sort_values("mu", ascending=False)[cols].head(10).round(3).to_dict("records")
        out[f"shooters_{grp}_bottom10"] = s.sort_values("mu")[cols].head(10).round(3).to_dict("records")
    ctx = final[final.pid < 0].copy()
    for k in ("RK", "TO", "TD"):
        c = ctx[ctx.grp == k].sort_values("mu")
        out[f"{k}_low3"] = c[["entity", "mu"]].head(3).round(4).to_dict("records")
        out[f"{k}_high3"] = c[["entity", "mu"]].tail(3).round(4).to_dict("records")
    return out


# ----------------------------------------------------------- W audit -------
def wf_audit(cutoffs=("2012-02-01", "2015-01-15", "2017-12-01")):
    D = FS.prep(verbose=False)
    cfg = {g: tuple(v) for g, v in json.load(open(FS.OUT_PAR))["params"].items()}
    S0, RG, RS = FS._vec(cfg)
    base = FS.run(D, S0, RG, RS)
    gdate = D["gm"].date.values
    s_date = gdate[D["s"].gi.values]
    r_date = gdate[D["ent"].gi.values]
    res = {}
    for k, cut in enumerate(cutoffs):
        y2 = D["s_y"].copy()
        m = np.where(s_date >= cut)[0]
        rng = np.random.default_rng(100 + k)
        y2[m] = rng.permutation(y2[m])
        y2[m[::7]] = 1 - y2[m[::7]]
        alt = FS.run(D, S0, RG, RS, y=y2)
        rr = r_date <= cut
        ss = s_date <= cut
        same = (np.array_equal(base[0][rr], alt[0][rr]) and np.array_equal(base[1][rr], alt[1][rr])
                and np.array_equal(base[2][ss], alt[2][ss]) and np.array_equal(base[3][ss], alt[3][ss])
                and np.array_equal(base[4][ss], alt[4][ss]))
        changed = int((base[0][r_date > cut] != alt[0][r_date > cut]).sum())
        res[cut] = dict(values_through_cutoff_date_identical=bool(same),
                        n_entity_games_through_cutoff=int(rr.sum()),
                        n_later_values_changed=changed, pass_=bool(same and changed > 0))
    return res


# ------------------------------------------------------------------ main ---
def main():
    t0 = time.time()
    cfg = {g: tuple(v) for g, v in json.load(open(FS.OUT_PAR))["params"].items()}
    pg = pd.read_parquet(FS.OUT_PG)
    eg = pd.read_parquet(FS.OUT_EG)
    sh = pd.read_parquet(FS.OUT_SH)
    for df in (pg, eg, sh):
        assert int(df.gid.max()) < DEV_MAX_GID, "TEST row in a DEV output"
    names = pd.read_csv(os.path.join(ROOT, "data/pv_nhl_players.csv"))
    gm = pd.read_csv(os.path.join(ROOT, "data/nhl_games.csv"), usecols=["game_id", "home", "away"])
    gm = gm[gm.game_id < DEV_MAX_GID].set_index("game_id")
    sh["rink"] = gm.home.reindex(sh.gid).values
    sh["def_team"] = np.where(sh.is_home == 1, gm.away.reindex(sh.gid).values,
                              gm.home.reindex(sh.gid).values)
    grp = pg.drop_duplicates("pid").set_index("pid").grp
    sh["is_d"] = (sh.shooter.map(grp) == "D").values
    res = {"component": "nhl finishing_and_saving", "params": {k: list(v) for k, v in cfg.items()},
           "dev_only": True, "test_seasons_touched": 0,
           "n_dev_shots": int(len(sh)), "n_dev_player_games": int(len(pg))}

    print("R reliability ...", flush=True)
    res["R_reliability"] = reliability(sh, pg, cfg)
    K_sh = res["R_reliability"]["shooter"]["implied_EB_K_xg_from_split_half_rc"]["K"] or 40.0
    res["shooter_EB_K_used_xg"] = K_sh
    print(json.dumps(res["R_reliability"], indent=0)[:6000], flush=True)

    print("P ...", flush=True)
    gb = goalie_baselines(pg)
    sb = shooter_baselines(pg)
    res["P1_shot_level_prequential"] = shot_ll(sh, gb, sb, K_sh)
    res["P2_goalie_frozen_season_start"] = goalie_frozen(gb, sh)
    res["P2_goalie_frozen_jan1_rest_of_season"] = goalie_frozen(gb, sh, cutoff="01-01")
    res["P3_shooter_frozen_season_start"] = shooter_frozen(sh, pg, sb, K_sh)
    res["P4_future_goals_per_game"] = future_goals(pg, K_sh)
    tg = pd.read_csv(FS.OUT_TG)          # written by the builder (walk-forward)
    assert int(tg.gid.max()) < DEV_MAX_GID
    res["P5_unit"] = unit_validity(tg)
    res["P5_goal_diff_increment"] = goal_diff_increment(tg)
    for k in ("P2_goalie_frozen_season_start", "P2_goalie_frozen_jan1_rest_of_season",
              "P3_shooter_frozen_season_start", "P4_future_goals_per_game", "P5_unit",
              "P5_goal_diff_increment"):
        print(k, json.dumps(res[k], indent=0), flush=True)

    print("F face ...", flush=True)
    final = pd.read_parquet(FS.OUT_FIN)
    res["F_face_validity_end_of_DEV"] = face(final, pg, names)
    print("W audit ...", flush=True)
    res["W_walk_forward_audit"] = wf_audit()
    print(json.dumps(res["W_walk_forward_audit"], indent=0), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(f"wrote {OUT} ({res['seconds']}s)")


if __name__ == "__main__":
    main()
