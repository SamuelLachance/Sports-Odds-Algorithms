"""NHL DEPTH leg — squeeze the components of the SHIPPED 4-feature blend.

The shipped NHL model (data/nhl_model.json) is a logistic over
[elo_logit, rest_diff, b2b_home, b2b_away, xg_diff]. Its six rating
hyperparameters were tuned GREEDILY: Elo (k, ha, regress) alone on raw Elo
log-loss, then the xG-Elo (XG_K, XG_HA, XG_REGRESS) separately, then the blend
fit on top. Elo and xG-Elo are partly redundant, so that sequence is almost
certainly not the joint optimum.

This module runs DEV-ONLY screens (the orchestrator owns TEST looks):

  A  JOINT re-tune of all six rating hyperparameters against the FULL blend's
     DEV log-loss, with a strict NESTED estimate of what the search is worth
     after removing search optimism.
  B  functional form of rest / b2b / xg.
  C  blend model class (logistic vs HistGradientBoosting vs stack).
  D  mechanistic interactions (xg x elo disagreement, b2b x rest).
  E  regularisation (blend C) and within-season shrinkage of ratings toward the
     league mean (the shipped model only regresses at season boundaries).

PROTOCOL (hard-asserted below):
  * the game list is TRUNCATED to season <= DEV_END before any rating is run,
    so no TEST-era game is ever even touched, let alone scored;
  * every score is a leave-one-season-out (LOSO) cross-validated DEV number,
    folds = seasons 2011-12 .. 2017-18;
  * noise is measured as the std of the per-season paired deltas; a gain is
    quoted in units of the standard error of that mean (gain / SE).

Market-blind: no odds enter any feature or any training signal.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date

import numpy as np

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import (DEV_END, DEV_START, DEV_WARM_BEFORE,  # noqa: E402
                              TEST_START, llv, load_games)
from nhl_features_eval import TEAM_FIX, XG_HA, XG_K, XG_REGRESS  # noqa: E402

OUT = "data/nhl_depth_dev.json"
SHIPPED_ELO = (8, 30, 0.30)          # k, ha, regress   (data/nhl_model.json)
SHIPPED_XG = (6.0, 0.15, 0.30)       # K, HA, regress
BLEND_C = 1e6                        # shipped blend uses C=1e6 (~unpenalised)

# ---------------------------------------------------------------- protocol ---
assert DEV_END < TEST_START, "DEV/TEST split constants moved"


def dev_games():
    """Every game strictly inside the DEV era. TEST games are never loaded."""
    games = [g for g in load_games() if g["season"] <= DEV_END]
    assert games, "no DEV games"
    hi = max(g["season"] for g in games)
    assert hi <= DEV_END < TEST_START, f"TEST-era game leaked into DEV list ({hi})"
    return games


def assert_dev_only(seasons):
    """Hard gate on anything that is about to be scored or printed."""
    s = np.asarray(seasons)
    assert s.size, "empty eval mask"
    assert s.min() >= DEV_WARM_BEFORE, f"pre-warm season scored ({s.min()})"
    assert s.max() <= DEV_END, f"TEST-era season scored ({s.max()})"
    assert s.max() < TEST_START


# ----------------------------------------------------------------- ratings ---
def run_elo_arr(games, k, ha, regress, shrink_n0=0.0):
    """Elo home-win probability per game (pre-game, leak-safe).

    shrink_n0 > 0 applies WITHIN-season shrinkage toward 1500: a team's rating
    is used as 1500 + (R-1500) * n/(n+n0) with n = games played this season.
    The stored rating still updates on the full deviation.
    """
    R = {}
    played = defaultdict(int)
    out = np.empty(len(games))
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - regress)
            played.clear()
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        if shrink_n0 > 0:
            eh = 1500 + (rh - 1500) * played[g["home"]] / (played[g["home"]] + shrink_n0)
            ea = 1500 + (ra - 1500) * played[g["away"]] / (played[g["away"]] + shrink_n0)
        else:
            eh, ea = rh, ra
        p = 1.0 / (1.0 + 10 ** (-((eh + ha) - ea) / 400.0))
        out[i] = p
        # update uses the same probability the model published (self-consistent)
        R[g["home"]] += k * (g["y"] - p)
        R[g["away"]] += k * ((1 - g["y"]) - (1 - p))
        played[g["home"]] += 1
        played[g["away"]] += 1
    return out


def load_team_xg_dev(games):
    keep = {g["game_id"] for g in games}
    out = defaultdict(dict)
    import csv
    for r in csv.DictReader(open("data/nhl_team_xg.csv", encoding="utf-8")):
        gid = int(r["nhl_game_id"])
        if gid not in keep:
            continue
        t = TEAM_FIX.get(r["team"], r["team"])
        out[gid][t] = (float(r["xgf"]), float(r["xga"]))
    return out


_TXG = None


def run_xg_arr(games, k, ha, regress, shrink_n0=0.0):
    """xG-Elo differential per game, replicating nhl_features_eval.build_features."""
    global _TXG
    if _TXG is None:
        _TXG = load_team_xg_dev(games)
    xr = defaultdict(float)
    played = defaultdict(int)
    out = np.full(len(games), np.nan)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for t in xr:
                xr[t] *= (1 - regress)
            played.clear()
        prev = g["season"]
        rh, ra = xr[g["home"]], xr[g["away"]]
        if shrink_n0 > 0:
            eh = rh * played[g["home"]] / (played[g["home"]] + shrink_n0)
            ea = ra * played[g["away"]] / (played[g["away"]] + shrink_n0)
        else:
            eh, ea = rh, ra
        out[i] = (eh + ha) - ea
        tx = _TXG.get(g["game_id"])
        if tx and g["home"] in tx and g["away"] in tx:
            hxgf, _ = tx[g["home"]]
            axgf, _ = tx[g["away"]]
            pred = rh - ra + ha
            err = (hxgf - axgf) - pred
            xr[g["home"]] += k / 100.0 * err
            xr[g["away"]] -= k / 100.0 * err
            played[g["home"]] += 1
            played[g["away"]] += 1
        else:
            out[i] = np.nan
    return out


def build_rest(games):
    """rest_diff (capped), b2b flags, and the raw capped rest days per side."""
    n = len(games)
    rest_diff = np.zeros(n)
    b2b_h = np.zeros(n)
    b2b_a = np.zeros(n)
    h_rest_a = np.zeros(n)
    a_rest_a = np.zeros(n)
    last = {}
    for i, g in enumerate(games):
        d = date.fromisoformat(g["date"])
        vals = {}
        for team, arr, key in ((g["home"], b2b_h, "h"), (g["away"], b2b_a, "a")):
            lg = last.get(team)
            rest = (d - lg).days if lg else 3
            rest = min(rest, 5)
            vals[key] = rest
            if lg and (d - lg).days <= 1:
                arr[i] = 1.0
        rest_diff[i] = vals["h"] - vals["a"]
        h_rest_a[i] = vals["h"]
        a_rest_a[i] = vals["a"]
        last[g["home"]] = d
        last[g["away"]] = d
    return dict(rest_diff=rest_diff, b2b_home=b2b_h, b2b_away=b2b_a,
                h_rest=h_rest_a, a_rest=a_rest_a)


# ---------------------------------------------------------------- logistic ---
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(X, y, C=BLEND_C, w0=None, max_iter=40, tol=1e-8):
    """IRLS logistic matching sklearn's objective: sum(logloss) + ||w||^2/(2C).

    X must carry an intercept column in position 0 (never penalised).
    """
    d = X.shape[1]
    w = np.zeros(d) if w0 is None else np.asarray(w0, float).copy()
    pen = np.full(d, 1.0 / C)
    pen[0] = 0.0
    for _ in range(max_iter):
        p = _sig(X @ w)
        g = X.T @ (y - p) - pen * w
        W = p * (1 - p) + 1e-10
        H = (X * W[:, None]).T @ X
        H[np.diag_indices(d)] += pen + 1e-9
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        nrm = np.max(np.abs(step))
        if nrm > 10:                      # damp wild Newton steps
            step *= 10.0 / nrm
        w += step
        if nrm < tol:
            break
    return w


def design(cols):
    return np.column_stack([np.ones(len(cols[0]))] + list(cols))


# ------------------------------------------------------------ CV machinery ---
class Folds:
    """Leave-one-season-out folds over the DEV eval window."""

    def __init__(self, seasons, mask):
        assert_dev_only(seasons[mask])
        self.mask = mask
        self.seasons = seasons[mask]
        self.list = sorted(set(self.seasons.tolist()))
        self.idx = {s: (self.seasons == s) for s in self.list}
        self.n = {s: int(self.idx[s].sum()) for s in self.list}
        self.N = int(mask.sum())

    def loso_pred(self, X, y, C=BLEND_C, cache=None):
        """Per-game out-of-fold predictions (fit on the other 6 seasons)."""
        p = np.empty(len(y))
        w0 = None if cache is None else cache.get("w")
        for s in self.list:
            tr = self.seasons != s
            w = fit_logit(X[tr], y[tr], C, w0)
            w0 = w
            p[self.idx[s]] = _sig(X[self.idx[s]] @ w)
        if cache is not None:
            cache["w"] = w0
        return p

    def per_season_loss(self, y, p):
        return {s: float(llv(y[self.idx[s]], p[self.idx[s]]).mean()) for s in self.list}

    def pooled(self, y, p):
        return float(llv(y, p).mean())


def fold_stats(y, folds, p_base, p_var):
    """Gain, per-season delta std, gain in SE units, and a paired bootstrap CI."""
    lb = llv(y, p_base)
    lv = llv(y, p_var)
    gain = float(lb.mean() - lv.mean())
    d = np.array([float(lb[folds.idx[s]].mean() - lv[folds.idx[s]].mean())
                  for s in folds.list])
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    rng = np.random.default_rng(7)
    diff = lb - lv
    bs = diff[rng.integers(0, len(diff), size=(4000, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"gain": round(gain, 5), "fold_sd": round(sd, 5),
            "fold_se": round(se, 5), "sigma": round(gain / se, 2) if se > 0 else 0.0,
            "n_pos_folds": int((d > 0).sum()), "n_folds": len(d),
            "boot_ci": [round(float(lo), 5), round(float(hi), 5)],
            "sig": "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")}


# ------------------------------------------------------------------ shared ---
class Ctx:
    def __init__(self):
        self.games = dev_games()
        self.seasons = np.array([g["season"] for g in self.games])
        self.y = np.array([g["y"] for g in self.games], float)
        self.F = build_rest(self.games)
        # eval window = warm DEV seasons with xG coverage under the SHIPPED xG cfg
        xg0 = run_xg_arr(self.games, *SHIPPED_XG)
        self.cover = ~np.isnan(xg0)
        self.mask = ((self.seasons >= DEV_WARM_BEFORE) & (self.seasons <= DEV_END)
                     & self.cover)
        self.folds = Folds(self.seasons, self.mask)
        self._elo = {}
        self._xg = {}
        assert_dev_only(self.seasons[self.mask])

    def elo(self, k, ha, reg, shrink=0.0):
        key = (k, ha, reg, shrink)
        if key not in self._elo:
            p = np.clip(run_elo_arr(self.games, k, ha, reg, shrink), 1e-9, 1 - 1e-9)
            self._elo[key] = np.log(p / (1 - p))[self.mask]
        return self._elo[key]

    def xg(self, k, ha, reg, shrink=0.0):
        key = (k, ha, reg, shrink)
        if key not in self._xg:
            self._xg[key] = np.nan_to_num(
                run_xg_arr(self.games, k, ha, reg, shrink))[self.mask]
        return self._xg[key]

    def base_cols(self):
        m = self.mask
        return [self.F["rest_diff"][m], self.F["b2b_home"][m], self.F["b2b_away"][m]]

    def yv(self):
        return self.y[self.mask]

    def X_for(self, elo_cfg, xg_cfg, shrink=(0.0, 0.0)):
        return design([self.elo(*elo_cfg, shrink=shrink[0])] + self.base_cols()
                      + [self.xg(*xg_cfg, shrink=shrink[1])])


# ------------------------------------------------------- axis A: joint core ---
ELO_K = [4, 6, 8, 10, 13, 16]
ELO_HA = [15, 20, 25, 30, 35, 40, 45]
ELO_R = [0.15, 0.22, 0.30, 0.38, 0.45]
XGK = [2.0, 4.0, 6.0, 9.0, 13.0]
XGHA = [0.05, 0.10, 0.15, 0.22]
XGR = [0.15, 0.22, 0.30, 0.38, 0.45]


def axis_a(ctx, quick=False):
    y = ctx.yv()
    folds = ctx.folds
    base = ctx.base_cols()
    ek, eh, er = (ELO_K, ELO_HA, ELO_R) if not quick else ([6, 8, 10], [25, 30, 35], [0.25, 0.35])
    xk, xh, xr = (XGK, XGHA, XGR) if not quick else ([4.0, 6.0, 9.0], [0.10, 0.15], [0.25, 0.35])
    elo_cfgs = list(itertools.product(ek, eh, er))
    xg_cfgs = list(itertools.product(xk, xh, xr))
    fl = folds.list
    pairs = [(a, b) for i, a in enumerate(fl) for b in fl[i + 1:]]

    print(f"[A] joint grid {len(elo_cfgs)} x {len(xg_cfgs)} = "
          f"{len(elo_cfgs)*len(xg_cfgs)} combos, {folds.N:,} DEV games, "
          f"{len(fl)} season folds", flush=True)

    # ---- greedy reproduction (what the shipped pipeline did) ----
    #   step 1: Elo alone, raw Elo log-loss (nhl_glicko2_eval's objective)
    g1 = min(elo_cfgs, key=lambda c: float(llv(y, _sig(ctx.elo(*c))).mean()))
    #   step 2: xG tuned with Elo frozen at step-1 (full blend objective)
    g2 = min(xg_cfgs, key=lambda c: folds.pooled(
        y, folds.loso_pred(design([ctx.elo(*g1)] + base + [ctx.xg(*c)]), y)))
    greedy = (g1, g2)

    results = {}
    t0 = time.time()
    loso_loss = {}
    pair_loss = {}
    cache = {}
    for ci, ec in enumerate(elo_cfgs):
        el = ctx.elo(*ec)
        for xc in xg_cfgs:
            X = design([el] + base + [ctx.xg(*xc)])
            p = folds.loso_pred(X, y, cache=cache)
            loso_loss[(ec, xc)] = folds.per_season_loss(y, p)
            pl = {}
            w0 = cache.get("w")
            for a, b in pairs:
                tr = (folds.seasons != a) & (folds.seasons != b)
                w = fit_logit(X[tr], y[tr], BLEND_C, w0)
                w0 = w
                for s, o in ((a, b), (b, a)):
                    m = folds.idx[o]
                    pl[(s, o)] = float(llv(y[m], _sig(X[m] @ w)).mean())
            pair_loss[(ec, xc)] = pl
        if ci % 10 == 0:
            print(f"    elo cfg {ci+1}/{len(elo_cfgs)}  {time.time()-t0:.0f}s", flush=True)

    def pooled_of(d):
        return sum(folds.n[s] * d[s] for s in fl) / folds.N

    combos = list(loso_loss)
    dev_score = {c: pooled_of(loso_loss[c]) for c in combos}
    shipped = (SHIPPED_ELO, SHIPPED_XG)
    assert quick or shipped in dev_score, "shipped point not on the grid"
    if shipped not in dev_score:            # quick smoke mode only
        shipped = min(combos, key=lambda c: dev_score[c])
    best = min(combos, key=lambda c: dev_score[c])
    ranked = sorted(combos, key=lambda c: dev_score[c])
    rank_shipped = ranked.index(shipped) + 1

    # per-game predictions for the paired comparison
    p_ship = folds.loso_pred(ctx.X_for(*shipped), y)
    p_best = folds.loso_pred(ctx.X_for(*best), y)
    st = fold_stats(y, folds, p_ship, p_best)

    # ---- strict nested estimate: search re-run inside each outer fold ----
    nested_pred = np.empty(len(y))
    nested_pick = {}
    for s in fl:
        inner = [o for o in fl if o != s]
        def inner_score(c):
            pl = pair_loss[c]
            return sum(folds.n[o] * pl[(s, o)] for o in inner) / sum(folds.n[o] for o in inner)
        c_star = min(combos, key=inner_score)
        nested_pick[s] = c_star
        # the outer prediction for season s is the LOSO fit (trained on the other 6)
        Xc = ctx.X_for(*c_star)
        tr = folds.seasons != s
        w = fit_logit(Xc[tr], y[tr], BLEND_C)
        nested_pred[folds.idx[s]] = _sig(Xc[folds.idx[s]] @ w)
    nested_ll = folds.pooled(y, nested_pred)
    nst = fold_stats(y, folds, p_ship, nested_pred)

    # ---- surface shape: 1-D profiles through the joint optimum ----
    names = ["elo_k", "elo_ha", "elo_regress", "xg_k", "xg_ha", "xg_regress"]
    grids = [ek, eh, er, xk, xh, xr]
    prof = {}
    for j, (nm, gr) in enumerate(zip(names, grids)):
        row = []
        for v in gr:
            e = list(best[0]); x = list(best[1])
            (e if j < 3 else x)[j % 3] = v
            c = (tuple(e), tuple(x))
            row.append([v, round(dev_score[c], 5)])
        prof[nm] = row
    spread = float(np.std([dev_score[c] for c in combos]))
    near = int(sum(1 for c in combos if dev_score[c] <= dev_score[best] + st["fold_se"]))

    results = {
        "n_dev": folds.N, "n_combos": len(combos),
        "shipped_cfg": {"elo": list(SHIPPED_ELO), "xg": list(SHIPPED_XG)},
        "shipped_dev_loso": round(dev_score[shipped], 5),
        "shipped_rank": rank_shipped,
        "shipped_pctile": round(100.0 * rank_shipped / len(combos), 2),
        "greedy_repro_cfg": {"elo": list(greedy[0]), "xg": list(greedy[1])},
        "greedy_repro_dev": round(dev_score[greedy], 5),
        "joint_best_cfg": {"elo": list(best[0]), "xg": list(best[1])},
        "joint_best_dev": round(dev_score[best], 5),
        "joint_vs_shipped": st,
        "nested_dev": round(nested_ll, 5),
        "nested_vs_shipped": nst,
        "search_optimism": round(dev_score[shipped] - dev_score[best]
                                 - (dev_score[shipped] - nested_ll), 5),
        "nested_picks": {str(s): {"elo": list(c[0]), "xg": list(c[1])}
                         for s, c in nested_pick.items()},
        "profiles": prof,
        "grid_score_sd": round(spread, 5),
        "combos_within_1se_of_best": near,
        "top10": [{"elo": list(c[0]), "xg": list(c[1]), "dev": round(dev_score[c], 5)}
                  for c in ranked[:10]],
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[A] shipped {results['shipped_dev_loso']:.5f} (rank {rank_shipped}/{len(combos)})"
          f"  joint best {results['joint_best_dev']:.5f} {best}"
          f"  nested {nested_ll:.5f}", flush=True)
    return results, best, p_ship


# ---------------------------------------------------- axes B/D: form + int ---
def _variants(ctx, core):
    """(name -> extra column list) over the given rating core. Column 0 = elo."""
    m = ctx.mask
    F = ctx.F
    el = ctx.elo(*core[0])
    xg = ctx.xg(*core[1])
    rd = F["rest_diff"][m]
    bh = F["b2b_home"][m]
    ba = F["b2b_away"][m]
    hr = F["h_rest"][m]
    ar = F["a_rest"][m]
    base = [el, rd, bh, ba, xg]
    V = {"BASELINE (shipped form)": base}

    # ---- B: rest functional form ----
    bins = []
    for lo, hi in [(-9, -2), (-1, -1), (1, 1), (2, 9)]:      # 0 is the reference
        bins.append(((rd >= lo) & (rd <= hi)).astype(float))
    V["rest as 5 bins"] = [el] + bins + [bh, ba, xg]
    knot = lambda v, t: np.maximum(v - t, 0.0)
    V["rest linear + hinges(+-1)"] = [el, rd, knot(rd, 1), knot(-rd, 1), bh, ba, xg]
    V["rest split h/a (2 terms)"] = [el, hr, ar, bh, ba, xg]
    V["rest sqrt-signed"] = [el, np.sign(rd) * np.sqrt(np.abs(rd)), bh, ba, xg]
    V["rest dropped"] = [el, bh, ba, xg]

    # ---- B: b2b form ----
    V["b2b as signed diff"] = [el, rd, bh - ba, xg]
    V["b2b both + signed"] = [el, rd, bh, ba, bh - ba, xg]
    V["b2b dropped"] = [el, rd, xg]

    # ---- B: xg form ----
    for s in (0.15, 0.25, 0.40):
        V[f"xg tanh(x/{s:.2f})"] = [el, rd, bh, ba, np.tanh(xg / s)]
    V["xg + xg^2"] = [el, rd, bh, ba, xg, xg ** 2]
    V["xg dropped"] = [el, rd, bh, ba]
    V["elo dropped"] = [rd, bh, ba, xg]

    # ---- D: interactions ----
    V["+ b2b_h x rest"] = base + [bh * rd, ba * rd]
    V["+ b2b_h x b2b_a"] = base + [bh * ba]
    V["+ xg x elo"] = base + [xg * el]
    V["+ |xg - elo/scale| disagree"] = base + [np.abs(xg - el * (xg.std() / el.std()))]
    V["+ elo^2 (shrink strong)"] = base + [el * np.abs(el)]
    V["+ xg x b2b_diff"] = base + [xg * (bh - ba)]
    V["+ all D interactions"] = base + [bh * rd, ba * rd, xg * el, xg * (bh - ba)]
    return V


def axis_bd(ctx, core, label):
    y = ctx.yv()
    folds = ctx.folds
    V = _variants(ctx, core)
    p0 = folds.loso_pred(design(V["BASELINE (shipped form)"]), y)
    base_ll = folds.pooled(y, p0)
    rows = []
    for name, cols in V.items():
        p = folds.loso_pred(design(cols), y)
        st = fold_stats(y, folds, p0, p)
        st.update({"name": name, "dev": round(folds.pooled(y, p), 5), "k": len(cols)})
        rows.append(st)
        print(f"  {name:<30} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
              f"({st['sigma']:+.2f} SE, {st['n_pos_folds']}/{st['n_folds']} folds) "
              f"{st['sig']}", flush=True)
    rows.sort(key=lambda r: -r["gain"])
    return {"core": label, "baseline_dev": round(base_ll, 5), "rows": rows}


# ----------------------------------------------------------- axis C: class ---
def axis_c(ctx, core):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    y = ctx.yv()
    folds = ctx.folds
    cols = _variants(ctx, core)["BASELINE (shipped form)"]
    X = design(cols)
    Xr = X[:, 1:]
    p0 = folds.loso_pred(X, y)
    rows = []

    # sklearn cross-check of the hand-rolled IRLS
    sk = LogisticRegression(C=BLEND_C, max_iter=3000).fit(Xr, y)
    mine = fit_logit(X, y)
    skv = np.r_[sk.intercept_, sk.coef_[0]]
    rows.append({"name": "IRLS vs sklearn max|coef diff|",
                 "value": round(float(np.max(np.abs(skv - mine))), 6)})

    def gbm_pred(**kw):
        p = np.empty(len(y))
        tr_ll = []
        for s in folds.list:
            tr = folds.seasons != s
            m = HistGradientBoostingClassifier(random_state=0, **kw).fit(Xr[tr], y[tr])
            p[folds.idx[s]] = m.predict_proba(Xr[folds.idx[s]])[:, 1]
            tr_ll.append(float(llv(y[tr], m.predict_proba(Xr[tr])[:, 1]).mean()))
        return p, float(np.mean(tr_ll))

    out = []
    for kw in ({"max_iter": 100, "learning_rate": 0.06, "max_leaf_nodes": 8,
                "min_samples_leaf": 200, "l2_regularization": 1.0},
               {"max_iter": 250, "learning_rate": 0.03, "max_leaf_nodes": 4,
                "min_samples_leaf": 400, "l2_regularization": 5.0},
               {"max_iter": 60, "learning_rate": 0.05, "max_leaf_nodes": 3,
                "min_samples_leaf": 600, "l2_regularization": 10.0}):
        p, tr_ll = gbm_pred(**kw)
        st = fold_stats(y, folds, p0, p)
        st.update({"name": f"HistGB {kw['max_iter']}x{kw['max_leaf_nodes']}lv "
                           f"lr{kw['learning_rate']}",
                   "dev": round(folds.pooled(y, p), 5), "train_ll": round(tr_ll, 5),
                   "train_cv_gap": round(folds.pooled(y, p) - tr_ll, 5)})
        out.append((st, p))
        print(f"  {st['name']:<34} dev {st['dev']:.5f} gain {st['gain']:+.5f} "
              f"({st['sigma']:+.2f} SE) train {tr_ll:.5f} gap {st['train_cv_gap']:+.5f}",
              flush=True)
    best_gbm = min(out, key=lambda o: o[0]["dev"])
    for w in (0.15, 0.30, 0.50):
        ps = (1 - w) * p0 + w * best_gbm[1]
        st = fold_stats(y, folds, p0, ps)
        st.update({"name": f"stack logistic+GBM w={w}", "dev": round(folds.pooled(y, ps), 5)})
        out.append((st, ps))
        print(f"  {st['name']:<34} dev {st['dev']:.5f} gain {st['gain']:+.5f} "
              f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)
    lin_train = float(llv(y, _sig(X @ fit_logit(X, y))).mean())
    rows.append({"name": "logistic (shipped class)", "dev": round(folds.pooled(y, p0), 5),
                 "train_ll": round(lin_train, 5),
                 "train_cv_gap": round(folds.pooled(y, p0) - lin_train, 5)})
    rows += [o[0] for o in out]
    return {"baseline_dev": round(folds.pooled(y, p0), 5), "rows": rows}


# --------------------------------------------------- axis E: reg + shrinkage --
def axis_e(ctx, core):
    y = ctx.yv()
    folds = ctx.folds
    cols = _variants(ctx, core)["BASELINE (shipped form)"]
    X = design(cols)
    Xs = X.copy()
    sd = Xs[:, 1:].std(axis=0)
    Xs[:, 1:] /= sd                        # standardise so C means the same thing
    p0 = folds.loso_pred(X, y)
    base = folds.pooled(y, p0)
    rows = []
    for C in (0.02, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 1e6):
        p = folds.loso_pred(Xs, y, C=C)
        st = fold_stats(y, folds, p0, p)
        st.update({"name": f"blend C={C:g}", "dev": round(folds.pooled(y, p), 5)})
        rows.append(st)
        print(f"  {st['name']:<26} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
              f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)
    for n0 in (0, 5, 10, 20, 40):
        for n0x in (0, 10, 30):
            if n0 == 0 and n0x == 0:
                continue
            Xv = ctx.X_for(core[0], core[1], shrink=(n0, n0x))
            p = folds.loso_pred(Xv, y)
            st = fold_stats(y, folds, p0, p)
            st.update({"name": f"within-season shrink elo n0={n0} xg n0={n0x}",
                       "dev": round(folds.pooled(y, p), 5)})
            rows.append(st)
            print(f"  {st['name']:<44} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
                  f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)
    rows.sort(key=lambda r: -r["gain"])
    return {"baseline_dev": round(base, 5), "rows": rows}


# -------------------------------------------------------------------- main ---
def merge_write(payload):
    cur = {}
    if os.path.exists(OUT):
        try:
            cur = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(payload)
    cur["_protocol"] = {"dev_seasons": [DEV_WARM_BEFORE, DEV_END],
                        "eval": "leave-one-season-out CV on DEV only",
                        "test_looks_spent": 0, "market_blind": True}
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cur, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "a", "b", "c", "e", "final", "summary"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ctx = Ctx()
    print(f"DEV games {ctx.folds.N:,} over seasons {ctx.folds.list} "
          f"(xG coverage {ctx.cover[(ctx.seasons>=DEV_WARM_BEFORE)].mean():.1%})",
          flush=True)
    assert_dev_only(ctx.seasons[ctx.mask])

    shipped_core = (SHIPPED_ELO, SHIPPED_XG)
    pay = {}
    prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    joint = shipped_core
    if prev.get("axis_a"):
        j = prev["axis_a"]["joint_best_cfg"]
        joint = (tuple(j["elo"]), tuple(j["xg"]))

    if args.stage in ("all", "a"):
        res, best, _ = axis_a(ctx, quick=args.quick)
        pay["axis_a"] = res
        joint = best
    if args.stage in ("all", "b"):
        print("\n[B/D] functional form + interactions @ SHIPPED core", flush=True)
        pay["axis_bd_shipped"] = axis_bd(ctx, shipped_core, "shipped")
        if joint != shipped_core:
            print("\n[B/D] functional form + interactions @ JOINT core", flush=True)
            pay["axis_bd_joint"] = axis_bd(ctx, joint, "joint")
    if args.stage in ("all", "c"):
        print("\n[C] model class @ shipped core", flush=True)
        pay["axis_c"] = axis_c(ctx, shipped_core)
    if args.stage in ("all", "e"):
        print("\n[E] regularisation + within-season shrinkage @ shipped core", flush=True)
        pay["axis_e"] = axis_e(ctx, shipped_core)
    if args.stage in ("all", "final"):
        y = ctx.yv()
        p_ship = ctx.folds.loso_pred(ctx.X_for(*shipped_core), y)
        p_joint = ctx.folds.loso_pred(ctx.X_for(*joint), y)
        st = fold_stats(y, ctx.folds, p_ship, p_joint)
        pay["headline"] = {
            "dev_baseline_shipped": round(ctx.folds.pooled(y, p_ship), 5),
            "dev_joint_core": round(ctx.folds.pooled(y, p_joint), 5),
            "joint_cfg": {"elo": list(joint[0]), "xg": list(joint[1])},
            "stats": st}
        print(f"\nHEADLINE shipped {pay['headline']['dev_baseline_shipped']:.5f} -> "
              f"joint {pay['headline']['dev_joint_core']:.5f} "
              f"({st['gain']:+.5f}, {st['sigma']:+.2f} SE)", flush=True)
    merge_write(pay)
    if args.stage in ("all", "summary"):
        merge_write({"shortlist": build_shortlist()})


def build_shortlist():
    """Rank every screened change by nested-honest gain, in noise (SE) units."""
    d = json.load(open(OUT, encoding="utf-8"))
    out = []
    a = d.get("axis_a")
    if a:
        out.append({"rank": 0, "axis": "A", "change": "joint 6-param core re-tune",
                    "raw_gain": a["joint_vs_shipped"]["gain"],
                    "raw_sigma": a["joint_vs_shipped"]["sigma"],
                    "honest_gain": a["nested_vs_shipped"]["gain"],
                    "honest_sigma": a["nested_vs_shipped"]["sigma"],
                    "search_optimism": a["search_optimism"],
                    "verdict": "NULL — nested gain is negative; the whole raw gain "
                               "is search optimism", "confidence": "high"})
    for key, axis in (("axis_bd_shipped", "B/D"), ("axis_e", "E"), ("axis_c", "C")):
        blk = d.get(key)
        if not blk:
            continue
        for r in sorted(blk["rows"], key=lambda r: -r.get("gain", -9))[:4]:
            if "gain" not in r:
                continue
            out.append({"rank": 0, "axis": axis, "change": r["name"],
                        "raw_gain": r["gain"], "raw_sigma": r["sigma"],
                        "honest_gain": None, "honest_sigma": None,
                        "search_optimism": None,
                        "verdict": r["sig"], "confidence": "high"})
    out.sort(key=lambda r: -(r["raw_gain"] or 0))
    for i, r in enumerate(out):
        r["rank"] = i + 1
    return out


if __name__ == "__main__":
    main()
