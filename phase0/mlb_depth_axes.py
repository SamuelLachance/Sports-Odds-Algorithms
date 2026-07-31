"""Axes A-F of the MLB depth campaign. Driven by phase0/mlb_depth_eval.py.

DEV ONLY (2002-2015). Every objective is the pooled out-of-fold log loss of the
shipped FULL tier under 5 season-grouped folds; the only thing an axis changes is
how one (or more) of the six blend columns is produced, or how the blend itself
is fit.

Honesty machinery lives in `nested_over_grid` / `nested_procedure`: a raw best
cell is always paired with a nested estimate in which the winning cell is
re-chosen without ever seeing the fold it is scored on. raw - nested = the
search optimism, reported in every axis.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time

import numpy as np

import mlb_depth_eval as M
from mlb_depth_eval import (DEV_HI, DEV_LO, EPS, FEATS, SHIP_ELO, _sig,
                            fit_logit, llvec, oof_ll, pr, pred_logit, save_axis)

# ---------------------------------------------------------------- context
class Ctx:
    def __init__(self):
        self.ds = M.load_ds(quiet=True)
        ds = self.ds
        cols, extra, ok = ds.design()
        self.ok = ok
        self.y = ds.y[ok]
        self.seas = ds.season[ok]
        self.groups = ds.groups
        self.masks = [np.isin(self.seas, list(g)) for g in self.groups]
        self.nf = np.array([m.sum() for m in self.masks], float)
        self.cols = {k: v[ok] for k, v in cols.items()}
        self.extra = {k: v[ok] for k, v in extra.items()}
        assert self.seas.max() <= DEV_HI and self.seas.min() >= DEV_LO, "TEST leak"
        self.base_ll_vec = oof_ll(M.full_cols(self.cols), self.y, self.seas, self.groups)
        self.base_fold = self.fold_means(self.base_ll_vec)
        self.base_ll = float(self.base_ll_vec.mean())
        print(f"context: n={len(self.y):,} DEV {DEV_LO}-{DEV_HI}  "
              f"baseline full-tier OOF ll {self.base_ll:.6f}", flush=True)

    def fold_means(self, v):
        return np.array([float(v[m].mean()) for m in self.masks])

    def pooled(self, fold):
        return float((np.asarray(fold) * self.nf).sum() / self.nf.sum())

    def score(self, cols):
        """per-fold OOF mean log loss of a candidate column set."""
        v = oof_ll(cols, self.y, self.seas, self.groups)
        return self.fold_means(v), v


# ------------------------------------------------------------ honest tools
def nested_over_grid(ctx, fold_ll, label, params_list):
    """fold_ll: (ncell, nfold) of a DATA-INDEPENDENT candidate grid.

    raw    = pick the cell with the best pooled OOF ll, quote its pooled ll.
    nested = for each fold, pick the cell using only the OTHER folds, quote the
             held-out fold. Legitimate because the candidate SET does not depend
             on the data (a fixed grid / a fixed random draw)."""
    nf = ctx.nf
    pooled = (fold_ll * nf).sum(1) / nf.sum()
    j = int(np.argmin(pooled))
    raw_fold = fold_ll[j]
    picks, nested_fold = [], []
    for f in range(len(nf)):
        o = [g for g in range(len(nf)) if g != f]
        sc = (fold_ll[:, o] * nf[o]).sum(1) / nf[o].sum()
        k = int(np.argmin(sc))
        picks.append(k)
        nested_fold.append(fold_ll[k, f])
    nested_fold = np.array(nested_fold)
    return _pack(ctx, label, params_list[j], raw_fold, nested_fold,
                 [params_list[k] for k in picks], n_cells=len(pooled))


def _pack(ctx, label, best_params, raw_fold, nested_fold, picks, n_cells=None):
    b = ctx.base_fold
    dr = b - raw_fold
    dn = b - nested_fold
    return {
        "label": label, "n_cells": n_cells,
        "best_params": best_params,
        "base_ll": round(ctx.base_ll, 6),
        "raw_ll": round(ctx.pooled(raw_fold), 6),
        "raw_delta": round(ctx.pooled(b) - ctx.pooled(raw_fold), 6),
        "nested_ll": round(ctx.pooled(nested_fold), 6),
        "nested_delta": round(ctx.pooled(b) - ctx.pooled(nested_fold), 6),
        "optimism": round(ctx.pooled(nested_fold) - ctx.pooled(raw_fold), 6),
        "raw_fold_deltas": [round(x, 6) for x in dr],
        "nested_fold_deltas": [round(x, 6) for x in dn],
        "fold_sd_raw": round(float(np.std(dr, ddof=1)), 6),
        "fold_sd_nested": round(float(np.std(dn, ddof=1)), 6),
        "n_sd_nested": (round(float(dn.mean() / np.std(dn, ddof=1)), 2)
                        if np.std(dn, ddof=1) > 0 else None),
        "fold_picks": picks,
    }


def show(r):
    print(f"  {r['label']:<30} raw {r['raw_ll']:.6f} (d{r['raw_delta']:+.6f}) | "
          f"nested {r['nested_ll']:.6f} (d{r['nested_delta']:+.6f}, "
          f"{r['n_sd_nested']} fold-sd) | optimism {r['optimism']:+.6f}", flush=True)


def paired_ci(ctx, ll_vec, seed=7):
    d = ctx.base_ll_vec - ll_vec
    lo, hi = M.boot_ci(d, seed=seed)
    return {"delta": round(float(d.mean()), 6), "ci": [round(lo, 6), round(hi, 6)],
            "sig": "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")}


# =================================================================== AXIS A
A_RANGES = {
    "k_team": (1.0, 9.0), "hfa": (8.0, 48.0), "team_regress": (0.10, 0.95),
    "beta": (10.0, 80.0), "decay": (0.80, 0.995), "prior_ip": (15.0, 250.0),
    "season_decay": (0.0, 0.7),
}
A_ORDER = list(A_RANGES)


def a_eval(ctx, p):
    lf = ctx.ds.elo_lf(p)[ctx.ok]
    cols = [lf] + [ctx.cols[k] for k in FEATS]
    return ctx.score(cols)


def a_profile(ctx):
    """1-D profiles through the shipped point: plateau or knife edge?"""
    out = {}
    for k, (lo, hi) in A_RANGES.items():
        vals = np.linspace(lo, hi, 11)
        lls = []
        for v in vals:
            p = dict(SHIP_ELO); p[k] = float(v)
            fold, _ = a_eval(ctx, p)
            lls.append(ctx.pooled(fold))
        base = a_eval(ctx, dict(SHIP_ELO))[0]
        out[k] = {"grid": [round(float(v), 4) for v in vals],
                  "ll": [round(x, 6) for x in lls],
                  "ship": SHIP_ELO[k], "ship_ll": round(ctx.pooled(base), 6),
                  "best_on_axis": round(float(vals[int(np.argmin(lls))]), 4),
                  "range_ll": round(float(max(lls) - min(lls)), 6)}
        print(f"    {k:<13} ship {SHIP_ELO[k]:>9.4f}  1-D best {out[k]['best_on_axis']:>9.4f}  "
              f"ll spread over range {out[k]['range_ll']:.5f}", flush=True)
    return out


def a_coord_descent(ctx, start, obj, passes=2, npts=9, shrink=0.45):
    """Coordinate descent on an arbitrary fold-subset objective `obj(fold)->float`."""
    cur = dict(start)
    fold, _ = a_eval(ctx, cur)
    best = obj(fold)
    width = {k: (hi - lo) * 0.5 for k, (lo, hi) in A_RANGES.items()}
    n_cells = 1
    for _ in range(passes):
        for k in A_ORDER:
            lo, hi = A_RANGES[k]
            c = cur[k]
            grid = np.clip(np.linspace(c - width[k], c + width[k], npts), lo, hi)
            for v in np.unique(np.round(grid, 6)):
                if abs(v - c) < 1e-9:
                    continue
                p = dict(cur); p[k] = float(v)
                f, _ = a_eval(ctx, p)
                n_cells += 1
                s = obj(f)
                if s < best:
                    best, cur = s, p
            width[k] *= shrink
    return cur, best, n_cells


def axis_a(ctx, n_random=600, seed=11):
    t0 = time.time()
    print("  1-D profiles through the shipped point:", flush=True)
    prof = a_profile(ctx)

    rng = np.random.default_rng(seed)
    cells = [dict(SHIP_ELO)]
    for _ in range(n_random):
        cells.append({k: float(rng.uniform(*A_RANGES[k])) for k in A_ORDER})
    fold_ll = np.empty((len(cells), len(ctx.nf)))
    for i, p in enumerate(cells):
        fold_ll[i], _ = a_eval(ctx, p)
        if (i + 1) % 100 == 0:
            print(f"    random {i+1}/{len(cells)}  best so far "
                  f"{((fold_ll[:i+1]*ctx.nf).sum(1)/ctx.nf.sum()).min():.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    rand_res = nested_over_grid(ctx, fold_ll, "A random-search (joint, 7 params)", cells)
    show(rand_res)

    # full procedure = random search THEN coordinate descent; nested version runs
    # the whole procedure once per outer fold on the other four folds only.
    def pooled_on(fold, idx):
        idx = list(idx)
        return float((np.asarray(fold)[idx] * ctx.nf[idx]).sum() / ctx.nf[idx].sum())

    allf = list(range(len(ctx.nf)))
    seed_cell = cells[int(np.argmin((fold_ll * ctx.nf).sum(1) / ctx.nf.sum()))]
    full_p, _, nc = a_coord_descent(ctx, seed_cell, lambda f: pooled_on(f, allf))
    full_fold, full_vec = a_eval(ctx, full_p)
    print(f"    coordinate descent (raw, all folds): {nc} extra cells, "
          f"ll {ctx.pooled(full_fold):.6f} ({time.time()-t0:.0f}s)", flush=True)

    nested_fold, picks = [], []
    for f in allf:
        inner = [g for g in allf if g != f]
        sc = (fold_ll[:, inner] * ctx.nf[inner]).sum(1) / ctx.nf[inner].sum()
        s0 = cells[int(np.argmin(sc))]
        pf, _, _ = a_coord_descent(ctx, s0, lambda fl: pooled_on(fl, inner))
        fold, _ = a_eval(ctx, pf)
        nested_fold.append(fold[f])
        picks.append({k: round(v, 4) for k, v in pf.items()})
        print(f"    nested outer fold {f}: ll {fold[f]:.6f} "
              f"(base {ctx.base_fold[f]:.6f})  ({time.time()-t0:.0f}s)", flush=True)
    proc = _pack(ctx, "A random+coord-descent (full procedure)",
                 {k: round(v, 4) for k, v in full_p.items()},
                 full_fold, np.array(nested_fold), picks, n_cells=len(cells) + nc)
    show(proc)
    proc["paired_bootstrap_raw"] = paired_ci(ctx, full_vec)
    return {"profiles": prof, "random_grid": rand_res, "procedure": proc,
            "ship": SHIP_ELO, "seconds": round(time.time() - t0)}


# =================================================================== AXIS B
B_BETA = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4]
B_TAU = [0.0, 0.003, 0.01, 0.03, 0.1]
B_EPS = [0.0, 0.05, 0.15]
B_SHRINK = [0.0, 20.0, 50.0, 150.0, 500.0]
B_SIGK = [0.0, 0.25, 0.5]


def axis_b(ctx):
    t0 = time.time()
    ds = ctx.ds
    cells, fold_ll = [], []
    for br, tr, er in itertools.product(B_BETA, B_TAU, B_EPS):
        W = M.ts_walk(ds.P, beta_rel=br, tau_rel=tr, eps_rel=er)
        idx = {g: i for i, g in enumerate(W["gid"])}
        pos = np.array([idx.get(g, -1) for g in ds.gid])
        for sn, sk in itertools.product(B_SHRINK, B_SIGK):
            *_, edge = M.ts_edge_from(W, shrink_n=sn, sig_k=sk)
            e = np.where(pos >= 0, edge[np.clip(pos, 0, None)], np.nan)[ctx.ok]
            if not np.all(np.isfinite(e)):
                continue
            cols = [ctx.cols["lf"], e] + [ctx.cols[k] for k in FEATS if k != "ts"]
            f, _ = ctx.score(cols)
            cells.append({"beta_rel": br, "tau_rel": tr, "eps_rel": er,
                          "shrink_n": sn, "sig_k": sk})
            fold_ll.append(f)
        print(f"    walk beta_rel={br} tau_rel={tr} eps_rel={er} done "
              f"({time.time()-t0:.0f}s, {len(cells)} cells)", flush=True)
    fold_ll = np.array(fold_ll)
    res = nested_over_grid(ctx, fold_ll, "B TrueSkill internals (full joint grid)", cells)
    show(res)

    # marginals: best ll achievable at each value of each knob (diagnostic only)
    pooled = (fold_ll * ctx.nf).sum(1) / ctx.nf.sum()
    marg = {}
    for k in ("beta_rel", "tau_rel", "eps_rel", "shrink_n", "sig_k"):
        vals = sorted({c[k] for c in cells})
        marg[k] = {str(v): round(float(pooled[[i for i, c in enumerate(cells)
                                               if c[k] == v]].min()), 6) for v in vals}
        print(f"    best-ll by {k:<9} " +
              "  ".join(f"{v}:{marg[k][v]:.6f}" for v in marg[k]), flush=True)
    ship_i = [i for i, c in enumerate(cells)
              if c["beta_rel"] == 0.5 and c["tau_rel"] == 0.01 and c["eps_rel"] == 0.0
              and c["shrink_n"] == 0.0 and c["sig_k"] == 0.0]
    res["ship_cell_ll"] = round(float(pooled[ship_i[0]]), 6) if ship_i else None
    res["marginals"] = marg
    res["note"] = ("engine is scale-invariant: mu0 fixed at 25 and (sig0, beta, tau) "
                   "scaled together only rescales the edge, which is z-scored before "
                   "the blend. beta_rel=beta/sig0 and tau_rel=tau/sig0 span the space. "
                   "A plate appearance cannot be drawn, so eps_rel is a soft-margin on "
                   "the win update, not a draw rate.")
    res["seconds"] = round(time.time() - t0)
    return res


# =================================================================== AXIS C
def _Xfull(ctx):
    return np.column_stack(M.full_cols(ctx.cols))


def _hgb(params):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=0, **params)


C_HGB = [
    dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=1.0),
    dict(max_iter=200, learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=1.0),
    dict(max_iter=300, learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0),
    dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=0.0),
    dict(max_iter=60, learning_rate=0.05, max_leaf_nodes=3, min_samples_leaf=500, l2_regularization=10.0),
    dict(max_iter=400, learning_rate=0.01, max_leaf_nodes=3, min_samples_leaf=500, l2_regularization=10.0),
]
C_CB = [
    dict(iterations=300, depth=3, learning_rate=0.03, l2_leaf_reg=10.0),
    dict(iterations=600, depth=2, learning_rate=0.02, l2_leaf_reg=10.0),
    dict(iterations=300, depth=6, learning_rate=0.03, l2_leaf_reg=3.0),
]
C_LOGC = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e6]


def _cv_model(ctx, make, X):
    """5-fold OOF + in-sample train loss for a sklearn-style classifier."""
    y = ctx.y
    oof = np.empty(len(y))
    tr_ll = []
    for m in ctx.masks:
        tr = ~m
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        Z = (X - mu) / sd
        clf = make()
        clf.fit(Z[tr], y[tr])
        oof[m] = clf.predict_proba(Z[m])[:, 1]
        tr_ll.append(float(llvec(y[tr], clf.predict_proba(Z[tr])[:, 1]).mean()))
    v = llvec(y, oof)
    return ctx.fold_means(v), v, float(np.mean(tr_ll))


def axis_c(ctx):
    t0 = time.time()
    X = _Xfull(ctx)
    out = {"note": "same six features as the shipped blend; only the model class changes"}

    # (i) logistic with tuned L2
    cells, fl = [], []
    for C in C_LOGC:
        l2 = 1.0 / C if C < 1e5 else 1e-8
        v = oof_ll(M.full_cols(ctx.cols), ctx.y, ctx.seas, ctx.groups, l2=l2)
        cells.append({"C": C})
        fl.append(ctx.fold_means(v))
    out["logistic_C"] = nested_over_grid(ctx, np.array(fl), "C logistic tuned C", cells)
    show(out["logistic_C"])

    # (ii) HistGradientBoosting
    cells, fl, gaps = [], [], []
    for p in C_HGB:
        f, v, tr = _cv_model(ctx, lambda p=p: _hgb(p), X)
        cells.append(p); fl.append(f); gaps.append(tr)
        print(f"    hgb {p}  cv {ctx.pooled(f):.6f}  train {tr:.6f} "
              f"gap {ctx.pooled(f)-tr:+.6f}", flush=True)
    out["hgb"] = nested_over_grid(ctx, np.array(fl), "C HistGradientBoosting", cells)
    out["hgb"]["train_ll"] = [round(g, 6) for g in gaps]
    out["hgb"]["train_cv_gap_best"] = round(out["hgb"]["raw_ll"] - gaps[int(np.argmin(
        (np.array(fl) * ctx.nf).sum(1) / ctx.nf.sum()))], 6)
    show(out["hgb"])

    # (iii) CatBoost
    try:
        from catboost import CatBoostClassifier
        cells, fl, gaps = [], [], []
        for p in C_CB:
            def mk(p=p):
                return CatBoostClassifier(verbose=0, allow_writing_files=False,
                                          random_seed=0, **p)
            f, v, tr = _cv_model(ctx, mk, X)
            cells.append(p); fl.append(f); gaps.append(tr)
            print(f"    catboost {p}  cv {ctx.pooled(f):.6f}  train {tr:.6f} "
                  f"gap {ctx.pooled(f)-tr:+.6f}", flush=True)
        out["catboost"] = nested_over_grid(ctx, np.array(fl), "C CatBoost", cells)
        out["catboost"]["train_ll"] = [round(g, 6) for g in gaps]
        show(out["catboost"])
    except Exception as e:                       # pragma: no cover
        out["catboost"] = {"error": str(e)}

    # (iv) 2-model stack: logistic + the best GBM config, with the stacker fit on
    #      INNER-fold out-of-fold predictions only (no outer fold ever seen).
    p_hgb = out["hgb"]["best_params"]
    oof = np.empty(len(ctx.y))
    ws_log = []
    for fi, m in enumerate(ctx.masks):
        tr = ~m
        tri = np.where(tr)[0]
        pos = {g: i for i, g in enumerate(tri)}
        pl = np.empty(len(tri)); pg = np.empty(len(tri))
        for j, im in enumerate(ctx.masks):
            if j == fi:
                continue
            itr, ite = tr & ~im, tr & im
            Z = _std(X, itr)
            w = fit_logit(Z[itr], ctx.y[itr])
            g = _hgb(p_hgb).fit(Z[itr], ctx.y[itr])
            idx = [pos[i] for i in np.where(ite)[0]]
            pl[idx] = pred_logit(w, Z[ite])
            pg[idx] = g.predict_proba(Z[ite])[:, 1]
        ws = fit_logit(np.column_stack([_lg(pl), _lg(pg)]), ctx.y[tr])
        Z = _std(X, tr)
        w = fit_logit(Z[tr], ctx.y[tr])
        g = _hgb(p_hgb).fit(Z[tr], ctx.y[tr])
        Se = np.column_stack([_lg(pred_logit(w, Z[m])),
                              _lg(g.predict_proba(Z[m])[:, 1])])
        oof[m] = pred_logit(ws, Se)
        ws_log.append([round(float(x), 3) for x in ws])
        print(f"    stack fold {fi}: stacker weights {ws_log[-1]}", flush=True)
    v = llvec(ctx.y, oof)
    out["stack"] = M.summarize("C logistic+HGB stack", ctx.base_ll_vec, v,
                               ctx.seas, ctx.groups)
    out["stack"]["stacker_weights"] = ws_log
    pr(out["stack"])
    out["seconds"] = round(time.time() - t0)
    return out


def _lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _std(X, m):
    mu, sd = X[m].mean(0), X[m].std(0)
    sd[sd == 0] = 1
    return (X - mu) / sd


# =================================================================== AXIS D
def rcs(x, knots):
    """Restricted (natural) cubic spline basis, Harrell's parameterisation.
    Returns k-2 columns; column 0 (linear) is supplied separately by the caller."""
    k = len(knots)
    t = np.asarray(knots, float)
    out = []
    d = (t[-1] - t[0]) ** (2.0 / 3.0)

    def cu(u):
        return np.clip(u, 0, None) ** 3
    for j in range(k - 2):
        v = (cu(x - t[j]) - cu(x - t[k - 2]) * (t[-1] - t[j]) / (t[-1] - t[k - 2])
             + cu(x - t[-1]) * (t[k - 2] - t[j]) / (t[-1] - t[k - 2])) / d
        out.append(v)
    return np.column_stack(out)


def _spline_cols(ctx, name, nk, tr_mask):
    x = ctx.cols[name] if name in ctx.cols else None
    q = np.linspace(0.05, 0.95, nk)
    knots = np.quantile(x[tr_mask], q)
    if len(np.unique(knots)) < nk:
        return None
    return rcs(x, knots)


def _oof_with_extra(ctx, extra_fn, l2=1e-8):
    """OOF loss where extra columns are BUILT PER FOLD from training data only."""
    y, out = ctx.y, np.empty(len(ctx.y))
    base = M.full_cols(ctx.cols)
    for m in ctx.masks:
        tr = ~m
        ex = extra_fn(tr)
        X = np.column_stack(base + (list(ex) if ex is not None else []))
        Z = X.copy()
        mu, sd = X[tr, 1:].mean(0), X[tr, 1:].std(0)
        sd[sd == 0] = 1
        Z[:, 1:] = (X[:, 1:] - mu) / sd
        w = fit_logit(Z[tr], y[tr], l2=l2)
        out[m] = llvec(y[m], pred_logit(w, Z[m]))
    return out


D_KNOTS = [3, 4, 5]


def axis_d(ctx):
    t0 = time.time()
    out = {"note": "restricted cubic splines and 10-quantile bins on each continuous "
                   "blend column; knots/edges fit on the training folds only"}
    per = {}
    for name in ["lf"] + list(FEATS):
        cells, fl = [], []
        for nk in D_KNOTS:
            def fn(tr, name=name, nk=nk):
                return [_spline_cols(ctx, name, nk, tr)]
            v = _oof_with_extra(ctx, fn)
            cells.append({"feature": name, "knots": nk})
            fl.append(ctx.fold_means(v))

        def binfn(tr, name=name):
            x = ctx.cols[name]
            e = np.quantile(x[tr], np.linspace(0, 1, 11)[1:-1])
            b = np.digitize(x, e)
            return [np.column_stack([(b == i).astype(float) for i in range(1, 10)])]
        v = _oof_with_extra(ctx, binfn)
        cells.append({"feature": name, "bins": 10})
        fl.append(ctx.fold_means(v))
        r = nested_over_grid(ctx, np.array(fl), f"D nonlinear {name}", cells)
        per[name] = r
        show(r)
    # everything at once, at the best per-feature knot count
    def allfn(tr):
        cs = []
        for name in ["lf"] + list(FEATS):
            nk = per[name]["best_params"].get("knots", 4)
            c = _spline_cols(ctx, name, nk, tr)
            if c is not None:
                cs.append(c)
        return cs
    v = _oof_with_extra(ctx, allfn)
    out["all_splines"] = M.summarize("D all features splined", ctx.base_ll_vec, v,
                                     ctx.seas, ctx.groups)
    pr(out["all_splines"])
    out["per_feature"] = per
    out["seconds"] = round(time.time() - t0)
    return out


# =================================================================== AXIS E
def e_terms(ctx):
    """Each interaction is justified mechanistically. No shotgunning."""
    c, x = ctx.cols, ctx.extra

    def z(a):
        return (a - a.mean()) / a.std()
    si_h, si_a = z(x["si_h"]), z(x["si_a"])
    bp_h, bp_a = z(x["bp_h"]), z(x["bp_a"])
    T = {}
    # 1. a weak pen matters more behind a short (bad) starter: weight each side's
    #    bullpen by that side's own starter's badness (higher SIERA = worse = more
    #    relief innings). Signed to favour the home team like bp does.
    T["bp_x_own_starter"] = (si_a * bp_a) - (si_h * bp_h)
    # 2. run scoring is not additive: a strong lineup gains less against an ace,
    #    so the lineup edge should be discounted by the opposing-starter edge.
    T["ts_x_si"] = z(c["ts"]) * z(c["si"])
    # 3. team Elo already contains the offence; the lineup increment should be
    #    absorbed when the Elo gap is already extreme.
    T["lf_x_ts"] = z(c["lf"]) * z(c["ts"])
    T["absLf_x_ts"] = z(np.abs(c["lf"])) * z(c["ts"])
    # 4. the same absorption argument for the bullpen channel.
    T["lf_x_bp"] = z(c["lf"]) * z(c["bp"])
    # 5. power and on-base are complements -- a home run is worth more with
    #    runners on, so ISO should pay more for a high-OBP lineup.
    T["ts_x_pw"] = z(c["ts"]) * z(c["pw"])
    # 6. baserunning only converts when men reach base.
    T["ts_x_br"] = z(c["ts"]) * z(c["br"])
    # 7. xFIP and SIERA rate the same starter; where they disagree the pitcher
    #    signal is less reliable, so the starter channel should be shrunk.
    T["lf_x_si"] = z(c["lf"]) * z(c["si"])
    return T


def axis_e(ctx):
    t0 = time.time()
    T = e_terms(ctx)
    singles, cells, fl = {}, [], []
    for name, t in T.items():
        v = oof_ll(M.full_cols(ctx.cols) + [t], ctx.y, ctx.seas, ctx.groups)
        s = M.summarize(f"E +{name}", ctx.base_ll_vec, v, ctx.seas, ctx.groups)
        singles[name] = s
        pr(s)
        cells.append({"term": name}); fl.append(ctx.fold_means(v))
    cells.append({"term": "none"}); fl.append(ctx.base_fold)
    best = nested_over_grid(ctx, np.array(fl), "E best single interaction", cells)
    show(best)
    v = oof_ll(M.full_cols(ctx.cols) + list(T.values()), ctx.y, ctx.seas, ctx.groups)
    allr = M.summarize("E all 8 interactions", ctx.base_ll_vec, v, ctx.seas, ctx.groups)
    pr(allr)
    return {"singles": singles, "best_single_nested": best, "all_terms": allr,
            "seconds": round(time.time() - t0)}


# =================================================================== AXIS F
F_BP = [50.0, 100.0, 200.0, 400.0, 800.0]
F_PW = [40.0, 80.0, 150.0, 300.0, 600.0]
F_BR = [50.0, 100.0, 200.0, 400.0, 800.0]
F_SI = [30.0, 60.0, 120.0, 250.0, 500.0]
F_C = [1e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 1e6]


def axis_f(ctx):
    t0 = time.time()
    ds = ctx.ds
    out = {}

    # (i) blend L2
    cells, fl = [], []
    for C in F_C:
        l2 = 1.0 / C if C < 1e5 else 1e-8
        v = oof_ll(M.full_cols(ctx.cols), ctx.y, ctx.seas, ctx.groups, l2=l2)
        cells.append({"C": C}); fl.append(ctx.fold_means(v))
    out["blend_C"] = nested_over_grid(ctx, np.array(fl), "F blend L2 (C)", cells)
    show(out["blend_C"])

    # (ii) the four empirical-Bayes feature priors, JOINTLY (full 5^4 grid,
    #      free because the as-of raw counts are cached)
    cells, fl = [], []
    for bp, pw, br, si in itertools.product(F_BP, F_PW, F_BR, F_SI):
        P = ds.priors_features(bp_prior=bp, pw_prior=pw, br_prior=br, si_prior=si)
        cols = [ctx.cols["lf"], ctx.cols["ts"],
                (P["bp_a"] - P["bp_h"])[ctx.ok], (P["pw_h"] - P["pw_a"])[ctx.ok],
                (P["si_h"] - P["si_a"])[ctx.ok], (P["br_h"] - P["br_a"])[ctx.ok]]
        f, _ = ctx.score(cols)
        cells.append({"bp_prior_outs": bp, "pwr_prior_pa": pw,
                      "br_prior_pa": br, "siera_prior": si})
        fl.append(f)
    fl = np.array(fl)
    out["feature_priors"] = nested_over_grid(
        ctx, fl, "F EB priors (joint 5^4)", cells)
    show(out["feature_priors"])
    pooled = (fl * ctx.nf).sum(1) / ctx.nf.sum()
    marg = {}
    for k, vals in (("bp_prior_outs", F_BP), ("pwr_prior_pa", F_PW),
                    ("br_prior_pa", F_BR), ("siera_prior", F_SI)):
        marg[k] = {str(v): round(float(pooled[[i for i, c in enumerate(cells)
                                               if c[k] == v]].min()), 6) for v in vals}
        print(f"    best-ll by {k:<15} " +
              "  ".join(f"{v}:{marg[k][v]:.6f}" for v in marg[k]), flush=True)
    out["feature_priors"]["marginals"] = marg
    ship_i = [i for i, c in enumerate(cells) if c == {"bp_prior_outs": 200.0,
                                                      "pwr_prior_pa": 150.0,
                                                      "br_prior_pa": 200.0,
                                                      "siera_prior": 120.0}]
    out["feature_priors"]["ship_cell_ll"] = round(float(pooled[ship_i[0]]), 6) if ship_i else None
    out["seconds"] = round(time.time() - t0)
    return out


def axis_f2(ctx):
    """Degeneracy check for the axis-F prior sweep, plus a leave-one-feature-out
    ablation. If pushing an empirical-Bayes prior up just switches a feature off,
    the 'gain' is an ablation in disguise and must be read that way."""
    t0 = time.time()
    ds = ctx.ds
    out = {"note": "wide 1-D prior sweeps + leave-one-out ablation of the blend"}
    wide = {"bp_prior_outs": [50, 200, 800, 3200, 12800, 51200],
            "pwr_prior_pa": [40, 150, 600, 2400, 9600, 38400],
            "br_prior_pa": [50, 200, 800, 3200, 12800],
            "siera_prior": [30, 120, 500, 2000, 8000, 32000]}
    sweeps = {}
    for k, vals in wide.items():
        row = {}
        for v in vals:
            kw = {"bp_prior": M.BP_PRIOR_OUTS, "pw_prior": M.PWR_PRIOR_PA,
                  "br_prior": M.BR_PRIOR_PA, "si_prior": M.SIERA_PRIOR}
            kw[{"bp_prior_outs": "bp_prior", "pwr_prior_pa": "pw_prior",
                "br_prior_pa": "br_prior", "siera_prior": "si_prior"}[k]] = float(v)
            P = ds.priors_features(**kw)
            cols = [ctx.cols["lf"], ctx.cols["ts"],
                    (P["bp_a"] - P["bp_h"])[ctx.ok], (P["pw_h"] - P["pw_a"])[ctx.ok],
                    (P["si_h"] - P["si_a"])[ctx.ok], (P["br_h"] - P["br_a"])[ctx.ok]]
            f, _ = ctx.score(cols)
            row[str(v)] = round(ctx.pooled(f), 6)
        sweeps[k] = row
        print(f"    {k:<15} " + "  ".join(f"{a}:{b:.6f}" for a, b in row.items()), flush=True)
    out["wide_sweeps"] = sweeps

    abl = {}
    for drop in FEATS:
        cols = [ctx.cols["lf"]] + [ctx.cols[k] for k in FEATS if k != drop]
        v = oof_ll(cols, ctx.y, ctx.seas, ctx.groups)
        s = M.summarize(f"drop {drop}", ctx.base_ll_vec, v, ctx.seas, ctx.groups)
        abl[drop] = s
        pr(s)
    out["ablation"] = abl
    out["seconds"] = round(time.time() - t0)
    return out


def combo(ctx):
    """Stack the per-axis winners and see whether they add.

    NOTE ON HONESTY: each component was chosen on DEV, so the combined RAW delta
    carries the sum of the per-axis optimism. The number to believe is the sum of
    the per-axis NESTED deltas, and only if the axes are roughly additive -- which
    is exactly what this checks (raw combined vs raw sum of parts)."""
    t0 = time.time()
    rep = json.load(open(M.REPORT))
    ds = ctx.ds
    out = {"note": "components chosen on DEV; believe the nested column, not the raw one"}

    elo = rep.get("axis_a", {}).get("procedure", {}).get("best_params") or dict(SHIP_ELO)
    elo = {k: float(elo[k]) for k in SHIP_ELO}
    pri = rep.get("axis_f", {}).get("feature_priors", {}).get("best_params") or {}
    tsb = rep.get("axis_b", {}).get("best_params") or dict(SHIP_TS)

    def cols_for(use_elo, use_pri, use_ts):
        lf = ds.elo_lf(elo if use_elo else SHIP_ELO)[ctx.ok]
        if use_ts:
            W = ds.ts_walk_cached(tsb["beta_rel"], tsb["tau_rel"], tsb.get("eps_rel", 0.0))
            *_, e = M.ts_edge_from(W, tsb.get("shrink_n", 0.0), tsb.get("sig_k", 0.0))
            idx = {g: i for i, g in enumerate(W["gid"])}
            pos = np.array([idx.get(g, 0) for g in ds.gid])
            ts = e[pos][ctx.ok]
        else:
            ts = ctx.cols["ts"]
        if use_pri:
            P = ds.priors_features(bp_prior=pri["bp_prior_outs"], pw_prior=pri["pwr_prior_pa"],
                                   br_prior=pri["br_prior_pa"], si_prior=pri["siera_prior"])
            rest = [(P["bp_a"] - P["bp_h"])[ctx.ok], (P["pw_h"] - P["pw_a"])[ctx.ok],
                    (P["si_h"] - P["si_a"])[ctx.ok], (P["br_h"] - P["br_a"])[ctx.ok]]
        else:
            rest = [ctx.cols[k] for k in ("bp", "pw", "si", "br")]
        return [lf, ts] + rest

    combos = {"A only": (1, 0, 0), "F only": (0, 1, 0), "B only": (0, 0, 1),
              "A+F": (1, 1, 0), "A+B+F": (1, 1, 1)}
    res = {}
    for name, (a, f, b) in combos.items():
        v = oof_ll(cols_for(a, f, b), ctx.y, ctx.seas, ctx.groups)
        s = M.summarize(f"combo {name}", ctx.base_ll_vec, v, ctx.seas, ctx.groups)
        res[name] = s
        pr(s)
    out["combos"] = res
    out["nested_delta_sum"] = round(
        (rep.get("axis_a", {}).get("procedure", {}).get("nested_delta", 0.0))
        + (rep.get("axis_b", {}).get("nested_delta", 0.0))
        + (rep.get("axis_f", {}).get("feature_priors", {}).get("nested_delta", 0.0)), 6)
    out["components"] = {"elo": elo, "priors": pri, "ts": tsb}
    out["seconds"] = round(time.time() - t0)
    return out


def shortlist(ctx):
    """Assemble the ranked shortlist from what the axes actually measured.

    Ranking key is the NESTED delta in units of the fold-to-fold sd, because a
    raw best cell is worth nothing if re-choosing it without the scored fold
    gives it back."""
    rep = json.load(open(M.REPORT))

    def g(*path, default=None):
        cur = rep
        for p in path:
            cur = cur.get(p, {}) if isinstance(cur, dict) else {}
        return cur if cur != {} else default
    items = []
    for label, node, what in (
            ("B per-PA TrueSkill internals", g("axis_b"),
             "tau_rel 0.01->0.003, read-time shrink_n 0->150, sig_k 0->0.5, beta_rel 0.5->~0.7-1.0"),
            ("A FIP-Elo hyperparameters (joint)", g("axis_a", "procedure"),
             "team_regress 0.54->~0.20, decay 0.953->~0.988, season_decay 0.009->~0.6, beta up"),
            ("F empirical-Bayes feature priors", g("axis_f", "feature_priors"),
             "bp 200->800 outs, pwr 150->600 PA, siera 120->500 PA, br unchanged at 200"),
    ):
        if not node:
            continue
        items.append({"axis": label, "change": what,
                      "raw_delta": node.get("raw_delta"),
                      "nested_delta": node.get("nested_delta"),
                      "optimism": node.get("optimism"),
                      "n_sd_nested": node.get("n_sd_nested")})
    items.sort(key=lambda d: -(d["nested_delta"] or 0))
    dead = []
    for label, node in (("C blend model class (GBM/CatBoost/stack)", g("axis_c", "hgb")),
                        ("D functional form (splines/bins)", g("axis_d", "all_splines")),
                        ("E interactions", g("axis_e", "all_terms")),
                        ("F blend L2", g("axis_f", "blend_C"))):
        if node:
            dead.append({"axis": label,
                         "delta": node.get("nested_delta", node.get("delta")),
                         "verdict": "no effect or worse than the shipped linear blend"})
    out = {"live": items, "dead": dead,
           "combined": g("combo", "combos", default={}).get("A+B+F") if isinstance(
               g("combo", "combos"), dict) else None,
           "nested_delta_sum": g("combo", default={}).get("nested_delta_sum"),
           "caveat": ("nested CV inside DEV estimates transfer to another 2002-2015 "
                      "season block, not to 2016-2024; rule and run-environment drift "
                      "is a further, unmeasured shrinkage on any TEST look.")}
    for it in items:
        print(f"  LIVE  {it['axis']:<38} nested {it['nested_delta']:+.6f} "
              f"({it['n_sd_nested']} fold-sd, optimism {it['optimism']:+.6f})", flush=True)
    for it in dead:
        print(f"  DEAD  {it['axis']:<38} {it['delta']:+.6f}", flush=True)
    return out


# ==================================================================== driver
AXES = {"--combo": ("combo", combo), "--shortlist": ("shortlist", shortlist),
        "--axis-a": ("axis_a", axis_a), "--axis-b": ("axis_b", axis_b),
        "--axis-c": ("axis_c", axis_c), "--axis-d": ("axis_d", axis_d),
        "--axis-e": ("axis_e", axis_e), "--axis-f": ("axis_f", axis_f),
        "--axis-f2": ("axis_f2", axis_f2)}


def main(mode, argv):
    if mode == "--report":
        rep = json.load(open(M.REPORT))
        for k, v in rep.items():
            print(k, json.dumps(v)[:200])
        return
    key, fn = AXES[mode]
    ctx = Ctx()
    res = fn(ctx)
    save_axis(key, res)
    print(f"wrote {M.REPORT} [{key}]", flush=True)
