"""MLB DEPTH round 2 — axes G (calibration), H (training cadence), I (ensembling).

DEV ONLY (2002-2015). Reuses phase0/mlb_depth_eval.py: cached as-of state, the
Newton-IRLS blend fitter, the DEV-capped scorer and the season-grouped folds.
Every scoring path re-asserts the DEV cap; no TEST-era row is parsed into this
process (the game loader is hard-capped at year_max=2015 in the cache build).

Market-blind: the only inputs are Retrosheet game logs and parsed events. No odds
enter any feature, any weight, or any selection criterion.

Honesty protocol (unchanged from round 1)
  * every search reports the RAW best cell and a NESTED estimate in which the
    winning cell is re-chosen without ever seeing the fold it is scored on;
    raw - nested = the search optimism.
  * fold-to-fold noise = std of the five per-fold deltas; every gain is quoted in
    units of that std.
  * negatives are reported with the same precision as positives.

Objectives
  G, I  : pooled 5-fold season-grouped OOF log loss of the shipped FULL tier
          (baseline 0.676686) — the same objective round 1 used.
  H     : a WALK-FORWARD objective, because cadence is meaningless under a
          symmetric CV split. Seasons >= first are scored with a blend fitted
          only on strictly earlier games. Its own baseline is reported; it is NOT
          comparable to the 5-fold number and is never compared to it.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

import numpy as np  # noqa: E402

import mlb_depth_eval as M  # noqa: E402
import mlb_depth_axes as AX  # noqa: E402
from mlb_depth_eval import DEV_HI, DEV_LO, EPS, FEATS, _sig, llvec, oof_ll  # noqa: E402

REPORT2 = "data/mlb_depth2_dev.json"
GATE = 0.0020            # MLB pre-registered gate, round 1. Never lowered here.

TIERS = {"recal": ["lf"],
         "recbp": ["lf", "bp", "si"],
         "full": ["lf"] + list(FEATS)}


# ============================================================ small utilities
def _lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logit_w(X, y, w=None, l2=1e-8, iters=60):
    """Weighted Newton-IRLS logistic fit (intercept added). w=None -> unweighted,
    in which case it reproduces M.fit_logit exactly."""
    n, k = X.shape
    A = np.empty((n, k + 1))
    A[:, 0] = 1.0
    A[:, 1:] = X
    if w is None:
        w = np.ones(n)
    b = np.zeros(k + 1)
    for _ in range(iters):
        p = _sig(A @ b)
        g = A.T @ (w * (p - y)) + l2 * b
        s = np.clip(p * (1 - p), 1e-10, None) * w
        H = (A * s[:, None]).T @ A
        H[np.diag_indices_from(H)] += l2
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        b -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return b


def _pred(b, X):
    return _sig(b[0] + X @ b[1:])


def _stdz(X, tr):
    """Standardise every column EXCEPT column 0 (the Elo logit, which enters raw),
    using the training rows only. Mirrors M.oof_ll exactly."""
    Z = X.copy()
    if X.shape[1] > 1:
        mu = X[tr, 1:].mean(0)
        sd = X[tr, 1:].std(0)
        sd[sd == 0] = 1.0
        Z[:, 1:] = (X[:, 1:] - mu) / sd
    return Z


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def save2(key, payload):
    old = {}
    if os.path.isfile(REPORT2):
        old = json.load(open(REPORT2))
    old[key] = payload
    old["_meta"] = {"dev": f"{DEV_LO}-{DEV_HI}", "folds": M.N_FOLDS,
                    "objective_GI": "pooled 5-fold season-grouped OOF log loss, FULL tier",
                    "objective_H": "walk-forward per-season refit, FULL tier",
                    "gate": GATE, "market_blind": True,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = REPORT2 + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(old, fh, indent=1)
    os.replace(tmp, REPORT2)


# =========================================================== calibrators
def cal_platt(p_tr, y_tr):
    """Platt: logistic on logit(p). Returns a callable p -> p'."""
    b = fit_logit_w(_lg(p_tr)[:, None], y_tr)
    return lambda p: _sig(b[0] + b[1] * _lg(p)), {"b0": float(b[0]), "b1": float(b[1])}


def cal_beta(p_tr, y_tr):
    """Beta calibration (Kull et al. 2017): logistic on [log p, -log(1-p)]."""
    q = np.clip(p_tr, EPS, 1 - EPS)
    Xt = np.column_stack([np.log(q), -np.log(1 - q)])
    b = fit_logit_w(Xt, y_tr)

    def f(p):
        q = np.clip(p, EPS, 1 - EPS)
        return _sig(b[0] + b[1] * np.log(q) + b[2] * (-np.log(1 - q)))
    return f, {"b0": float(b[0]), "a": float(b[1]), "c": float(b[2])}


def cal_iso(p_tr, y_tr, clip=1e-3):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=clip, y_max=1 - clip)
    ir.fit(p_tr, y_tr)
    return (lambda p: np.clip(ir.predict(p), clip, 1 - clip)), {"n_knots": int(len(ir.X_thresholds_))}


CALS = {"platt": cal_platt, "beta": cal_beta, "isotonic": cal_iso}


# ==================================================================== AXIS G
def g_pre_blend(ctx):
    """(i) Recalibrate the FIP-Elo channel BEFORE it enters the blend.

    Shipped = logit(fip_elo) enters linearly and the blend refits its slope and
    intercept, i.e. Platt is already embedded in the blend. Any affine recal is
    therefore an exact algebraic null; the informative variants are the
    non-affine ones (isotonic, beta)."""
    y, seas, groups = ctx.y, ctx.seas, ctx.groups
    p_elo = _sig(ctx.cols["lf"])
    rest = [ctx.cols[k] for k in FEATS]
    out, cells, fl = {}, [], []
    for name in ("shipped", "platt", "beta", "isotonic"):
        v = np.empty(len(y))
        coefs = []
        for m in ctx.masks:
            tr = ~m
            if name == "shipped":
                lf = ctx.cols["lf"]
            else:
                f, info = CALS[name](p_elo[tr], y[tr])
                coefs.append(info)
                lf = _lg(f(p_elo))
            X = np.column_stack([lf] + rest)
            Z = _stdz(X, tr)
            b = fit_logit_w(Z[tr], y[tr])
            v[m] = llvec(y[m], _pred(b, Z[m]))
        s = M.summarize(f"G pre-blend {name}", ctx.base_ll_vec, v, seas, groups)
        s["fold_coefs"] = coefs
        out[name] = s
        M.pr(s)
        cells.append({"pre_cal": name})
        fl.append(ctx.fold_means(v))
    out["_nested_choice"] = AX.nested_over_grid(
        ctx, np.array(fl), "G pre-blend calibrator (4 cells)", cells)
    AX.show(out["_nested_choice"])
    out["note"] = ("shipped and platt are algebraically identical: a logistic blend "
                   "with an intercept absorbs any affine map of one of its columns. "
                   "The measured delta between them is pure numerical noise and is "
                   "the control for this screen.")
    return out


def g_post_blend(ctx):
    """(i, continued) Recalibrate the BLEND OUTPUT.

    Fitting a calibrator on the blend's own IN-SAMPLE predictions is the identity
    (the MLE score equations already force slope 1, intercept 0), so the only
    meaningful version fits the calibrator on INNER out-of-fold predictions —
    which makes it a shrinkage correction for the blend's in-sample optimism."""
    y, seas, groups = ctx.y, ctx.seas, ctx.groups
    X = np.column_stack(M.full_cols(ctx.cols))
    out, cells, fl = {}, [], []
    # inner-OOF predictions on each outer training set (built once, reused)
    inner = {}
    for fi, m in enumerate(ctx.masks):
        tr = ~m
        pin = np.full(len(y), np.nan)
        for j, im in enumerate(ctx.masks):
            if j == fi:
                continue
            itr, ite = tr & ~im, tr & im
            Z = _stdz(X, itr)
            b = fit_logit_w(Z[itr], y[itr])
            pin[ite] = _pred(b, Z[ite])
        Z = _stdz(X, tr)
        b = fit_logit_w(Z[tr], y[tr])
        inner[fi] = (tr, pin, _pred(b, Z[m]), _pred(b, Z[tr]))
    for name in ("none", "platt", "beta", "isotonic"):
        v = np.empty(len(y))
        coefs = []
        for fi, m in enumerate(ctx.masks):
            tr, pin, pte, ptr = inner[fi]
            if name == "none":
                v[m] = llvec(y[m], pte)
                continue
            f, info = CALS[name](pin[tr], y[tr])
            coefs.append(info)
            v[m] = llvec(y[m], f(pte))
        s = M.summarize(f"G post-blend {name}", ctx.base_ll_vec, v, seas, groups)
        s["fold_coefs"] = coefs
        out[name] = s
        M.pr(s)
        cells.append({"post_cal": name})
        fl.append(ctx.fold_means(v))
    # the in-sample identity, verified numerically
    fi = 0
    tr, pin, pte, ptr = inner[fi]
    b_in = fit_logit_w(_lg(ptr)[:, None], y[tr])
    out["in_sample_platt_identity"] = {"b0": round(float(b_in[0]), 8),
                                       "b1": round(float(b_in[1]), 8),
                                       "note": "fitting Platt on the blend's own training "
                                               "predictions returns (0,1) exactly — a "
                                               "post-hoc curve can only do work if it is "
                                               "fitted out of sample"}
    out["_nested_choice"] = AX.nested_over_grid(
        ctx, np.array(fl), "G post-blend calibrator (4 cells)", cells)
    AX.show(out["_nested_choice"])
    return out


def g_per_tier(ctx):
    """(ii) One shared recalibration curve for the Elo channel vs a per-tier one.

    Shipped: every tier refits its own (b0, b1) on logit(fip_elo) — i.e. three
    private Platt curves. Shared: fit Platt once on the training folds, then hold
    it as a fixed-coefficient offset while each tier fits only its own feature
    weights and intercept."""
    y, seas, groups = ctx.y, ctx.seas, ctx.groups
    p_elo = _sig(ctx.cols["lf"])
    out = {"per_tier_slopes": {}, "shared": {}, "note": ""}
    slopes = {t: [] for t in TIERS}
    for t, names in TIERS.items():
        for m in ctx.masks:
            tr = ~m
            X = np.column_stack([ctx.cols[k] for k in names])
            Z = _stdz(X, tr)
            b = fit_logit_w(Z[tr], y[tr])
            slopes[t].append(round(float(b[1]), 4))
        out["per_tier_slopes"][t] = slopes[t]
    shared_slope = []
    for m in ctx.masks:
        tr = ~m
        b = fit_logit_w(_lg(p_elo)[tr][:, None], y[tr])
        shared_slope.append(round(float(b[1]), 4))
    out["per_tier_slopes"]["shared_platt"] = shared_slope
    cells, fl = [], []
    for mode in ("per_tier", "shared_offset"):
        v = np.empty(len(y))
        for m in ctx.masks:
            tr = ~m
            X = np.column_stack(M.full_cols(ctx.cols))
            Z = _stdz(X, tr)
            if mode == "per_tier":
                b = fit_logit_w(Z[tr], y[tr])
                v[m] = llvec(y[m], _pred(b, Z[m]))
            else:
                bp = fit_logit_w(_lg(p_elo)[tr][:, None], y[tr])
                off = bp[0] + bp[1] * _lg(p_elo)
                Zr = Z[:, 1:]
                # the remaining columns are refit with the shared curve pinned in
                br = _fit_offset(Zr[tr], y[tr], off[tr])
                v[m] = llvec(y[m], _sig(off[m] + br[0] + Zr[m] @ br[1:]))
        s = M.summarize(f"G tier-cal {mode}", ctx.base_ll_vec, v, seas, groups)
        out["shared"][mode] = s
        M.pr(s)
        cells.append({"tier_cal": mode})
        fl.append(ctx.fold_means(v))
    out["_nested_choice"] = AX.nested_over_grid(
        ctx, np.array(fl), "G shared vs per-tier calibration", cells)
    AX.show(out["_nested_choice"])
    out["note"] = ("the shipped serve chain gives the recal tier its own Platt and lets "
                   "recbp/full refit their own Elo slope; this measures whether freezing "
                   "one marginal curve across all tiers costs anything.")
    return out


def _fit_offset(X, y, off, l2=1e-8, iters=60):
    """Logistic fit with a fixed per-row offset (coefficient pinned at 1)."""
    n, k = X.shape
    A = np.empty((n, k + 1))
    A[:, 0] = 1.0
    A[:, 1:] = X
    b = np.zeros(k + 1)
    for _ in range(iters):
        p = _sig(off + A @ b)
        g = A.T @ (p - y) + l2 * b
        s = np.clip(p * (1 - p), 1e-10, None)
        H = (A * s[:, None]).T @ A
        H[np.diag_indices_from(H)] += l2
        step = np.linalg.solve(H, g)
        b -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return b


def g_drift(ctx):
    """(iii) Is a curve fitted on 2002-2008 still right in 2013-2015?"""
    y, seas = ctx.y, ctx.seas
    p_elo = _sig(ctx.cols["lf"])
    out = {}

    # per-season Platt coefficients with analytic SEs
    traj = {}
    for s in sorted(set(seas.tolist())):
        m = seas == s
        Xs = _lg(p_elo)[m][:, None]
        b = fit_logit_w(Xs, y[m])
        A = np.column_stack([np.ones(m.sum()), Xs])
        p = _sig(A @ b)
        H = (A * (p * (1 - p))[:, None]).T @ A
        cov = np.linalg.inv(H)
        traj[int(s)] = {"n": int(m.sum()),
                        "b0": round(float(b[0]), 4), "se0": round(float(math.sqrt(cov[0, 0])), 4),
                        "b1": round(float(b[1]), 4), "se1": round(float(math.sqrt(cov[1, 1])), 4),
                        "home_rate": round(float(y[m].mean()), 4)}
    out["per_season_platt"] = traj
    b1s = np.array([traj[s]["b1"] for s in traj])
    se1 = np.array([traj[s]["se1"] for s in traj])
    b0s = np.array([traj[s]["b0"] for s in traj])
    se0 = np.array([traj[s]["se0"] for s in traj])
    # heterogeneity test: is the spread of yearly slopes bigger than sampling noise?
    for nm, v, se in (("slope", b1s, se1), ("intercept", b0s, se0)):
        w = 1.0 / se ** 2
        mubar = float((w * v).sum() / w.sum())
        Q = float((w * (v - mubar) ** 2).sum())
        df = len(v) - 1
        out[f"{nm}_heterogeneity"] = {
            "pooled": round(mubar, 4), "Q": round(Q, 2), "df": df,
            "excess_over_noise": round(Q / df, 2),
            "reading": ("Q/df >> 1 means the season-to-season curve really moves; "
                        "Q/df ~ 1 means the wobble is sampling noise")}

    # stale vs fresh calibration on the last three DEV seasons
    late = np.isin(seas, [2013, 2014, 2015])
    tests = {"fit_2002_2008": np.isin(seas, list(range(2002, 2009))),
             "fit_2009_2012": np.isin(seas, list(range(2009, 2013))),
             "fit_2010_2012": np.isin(seas, list(range(2010, 2013))),
             "fit_2012_only": seas == 2012,
             "fit_2013_2015_oracle": late}
    stale = {}
    for nm, trm in tests.items():
        row = {}
        # (a) Platt on the Elo channel only
        b = fit_logit_w(_lg(p_elo)[trm][:, None], y[trm])
        row["platt_ll_on_2013_2015"] = round(float(
            llvec(y[late], _sig(b[0] + b[1] * _lg(p_elo)[late])).mean()), 6)
        row["platt_b0b1"] = [round(float(b[0]), 4), round(float(b[1]), 4)]
        # (b) the whole 6-column blend
        X = np.column_stack(M.full_cols(ctx.cols))
        Z = _stdz(X, trm)
        bb = fit_logit_w(Z[trm], y[trm])
        row["full_blend_ll_on_2013_2015"] = round(float(
            llvec(y[late], _pred(bb, Z[late])).mean()), 6)
        row["n_train"] = int(trm.sum())
        stale[nm] = row
        print(f"    {nm:<22} n={row['n_train']:>6,}  platt {row['platt_ll_on_2013_2015']:.6f}"
              f"  full-blend {row['full_blend_ll_on_2013_2015']:.6f}"
              f"  (b0,b1)={row['platt_b0b1']}", flush=True)
    out["stale_vs_fresh"] = stale
    o = stale["fit_2013_2015_oracle"]
    out["staleness_cost"] = {
        "platt_2002_2008_minus_oracle": round(
            stale["fit_2002_2008"]["platt_ll_on_2013_2015"] - o["platt_ll_on_2013_2015"], 6),
        "full_2002_2008_minus_oracle": round(
            stale["fit_2002_2008"]["full_blend_ll_on_2013_2015"] - o["full_blend_ll_on_2013_2015"], 6),
        "note": ("the oracle is fitted IN SAMPLE on 2013-2015 so it is an optimistic "
                 "floor, not an achievable target; the gap it shows is an upper bound "
                 "on what perfect freshness could buy.")}
    return out


def g_deciles(ctx):
    """(iv) WHERE is the model miscalibrated? Reliability tables with CIs."""
    y, seas = ctx.y, ctx.seas
    X = np.column_stack(M.full_cols(ctx.cols))
    oof = np.empty(len(y))
    for m in ctx.masks:
        tr = ~m
        Z = _stdz(X, tr)
        b = fit_logit_w(Z[tr], y[tr])
        oof[m] = _pred(b, Z[m])
    p_raw = _sig(ctx.cols["lf"])

    def table(p, mask=None, nb=10, label=""):
        sel = np.ones(len(y), bool) if mask is None else mask
        pp, yy = p[sel], y[sel]
        e = np.quantile(pp, np.linspace(0, 1, nb + 1))
        e[0], e[-1] = -np.inf, np.inf
        rows = []
        ece = 0.0
        for i in range(nb):
            m = (pp > e[i]) & (pp <= e[i + 1])
            n = int(m.sum())
            if n == 0:
                continue
            k = float(yy[m].sum())
            pm = float(pp[m].mean())
            obs = k / n
            lo, hi = wilson(k, n)
            z = (obs - pm) / math.sqrt(max(pm * (1 - pm), 1e-9) / n)
            ece += n / len(pp) * abs(obs - pm)
            rows.append({"decile": i + 1, "n": n,
                         "p_lo": round(float(pp[m].min()), 4),
                         "p_hi": round(float(pp[m].max()), 4),
                         "pred": round(pm, 4), "obs": round(obs, 4),
                         "obs_ci": [round(lo, 4), round(hi, 4)],
                         "obs_minus_pred": round(obs - pm, 4), "z": round(z, 2)})
        # calibration slope/intercept of the whole curve
        b = fit_logit_w(_lg(pp)[:, None], yy)
        return {"label": label, "n": int(sel.sum()), "bins": rows,
                "ece": round(ece, 5),
                "cal_intercept": round(float(b[0]), 4), "cal_slope": round(float(b[1]), 4),
                "n_bins_outside_ci": sum(1 for r in rows
                                         if r["pred"] < r["obs_ci"][0] or r["pred"] > r["obs_ci"][1])}

    out = {"full_oof": table(oof, None, 10, "full-tier OOF, all DEV"),
           "raw_fip_elo": table(p_raw, None, 10, "raw FIP-Elo, uncalibrated"),
           "full_oof_20": table(oof, None, 20, "full-tier OOF, 20 bins"),
           "full_oof_2002_2008": table(oof, seas <= 2008, 10, "full-tier OOF, 2002-2008"),
           "full_oof_2009_2015": table(oof, seas >= 2009, 10, "full-tier OOF, 2009-2015"),
           }
    for k in ("full_oof", "raw_fip_elo"):
        t = out[k]
        print(f"    {t['label']}: slope {t['cal_slope']:.4f} intercept {t['cal_intercept']:+.4f} "
              f"ECE {t['ece']:.5f}  bins outside 95% CI {t['n_bins_outside_ci']}/10", flush=True)
        for r in t["bins"]:
            print(f"      d{r['decile']:>2} n={r['n']:>5} p[{r['p_lo']:.3f},{r['p_hi']:.3f}] "
                  f"pred {r['pred']:.4f} obs {r['obs']:.4f} "
                  f"[{r['obs_ci'][0]:.4f},{r['obs_ci'][1]:.4f}] "
                  f"d {r['obs_minus_pred']:+.4f} z {r['z']:+.2f}", flush=True)
    return out


def axis_g(ctx):
    t0 = time.time()
    print("  (i) pre-blend recalibration of the FIP-Elo channel", flush=True)
    pre = g_pre_blend(ctx)
    print("  (i) post-blend recalibration of the served probability", flush=True)
    post = g_post_blend(ctx)
    print("  (ii) shared vs per-tier calibration", flush=True)
    tier = g_per_tier(ctx)
    print("  (iii) calibration drift", flush=True)
    drift = g_drift(ctx)
    print("  (iv) probability-decile reliability tables", flush=True)
    dec = g_deciles(ctx)

    # axis-level nested choice over EVERY calibration variant screened
    grid = []
    for nm in ("shipped", "beta", "isotonic"):
        grid.append(ctx.base_fold - np.array(pre[nm]["fold_deltas"]))
    for nm in ("platt", "beta", "isotonic"):
        grid.append(ctx.base_fold - np.array(post[nm]["fold_deltas"]))
    labels = ([{"where": "pre", "cal": n} for n in ("shipped", "beta", "isotonic")] +
              [{"where": "post", "cal": n} for n in ("platt", "beta", "isotonic")])
    best = AX.nested_over_grid(ctx, np.array(grid), "G best calibration variant", labels)
    AX.show(best)
    return {"pre_blend": pre, "post_blend": post, "per_tier": tier,
            "drift": drift, "deciles": dec, "axis_best": best,
            "seconds": round(time.time() - t0)}


# ==================================================================== AXIS H
class WFCtx:
    """A walk-forward scoring context. Exposes the same interface AX._pack needs."""

    def __init__(self, ctx, first=2005):
        self.ctx = ctx
        self.first = first
        self.seas = ctx.seas
        self.y = ctx.y
        self.X = np.column_stack(M.full_cols(ctx.cols))
        assert self.seas.max() <= DEV_HI and self.seas.min() >= DEV_LO, "TEST leak"
        assert np.all(np.diff(self.seas) >= 0), "rows must be chronological"
        self.scored = self.seas >= first
        ss = sorted(set(self.seas[self.scored].tolist()))
        self.groups = [set(int(x) for x in g) for g in np.array_split(np.array(ss), M.N_FOLDS)]
        self.masks = [np.isin(self.seas, list(g)) & self.scored for g in self.groups]
        self.nf = np.array([m.sum() for m in self.masks], float)
        self.base_vec = self.run()
        self.base_fold = self.fold_means(self.base_vec)
        self.base_ll = self.pooled(self.base_fold)
        self.base_ll_vec = self.base_vec
        print(f"  walk-forward baseline (expanding window, refit each season, "
              f"first={first}): n={int(self.scored.sum()):,} ll {self.base_ll:.6f}", flush=True)

    def _sub(self, m):
        return m[self.scored]

    def fold_means(self, v):
        return np.array([float(v[self._sub(m)].mean()) for m in self.masks])

    def pooled(self, fold):
        return float((np.asarray(fold) * self.nf).sum() / self.nf.sum())

    def run(self, half_life=None, window=None, refit_games=None, refit_seasons=1,
            ret_pred=False):
        """Score every game in seasons >= first with a blend fitted only on games
        that had already been played. half_life/window are in SEASONS."""
        X, y, seas = self.X, self.y, self.seas
        n = len(y)
        pred = np.full(n, np.nan)
        cache = None
        last_fit = None
        for s in sorted(set(seas[self.scored].tolist())):
            te = np.where(seas == s)[0]
            hist = np.where(seas < s)[0]
            do_refit = last_fit is None or (s - last_fit) >= refit_seasons

            def fit(tr_idx, s_now):
                if window:
                    # keep the `window` most recent PRIOR seasons (plus, when an
                    # intra-season refit is in play, the current season to date)
                    tr_idx = tr_idx[seas[tr_idx] >= s_now - window]
                w = None
                if half_life:
                    w = 0.5 ** ((s_now - seas[tr_idx]) / half_life)
                mu = X[tr_idx][:, 1:].mean(0)
                sd = X[tr_idx][:, 1:].std(0)
                sd[sd == 0] = 1.0
                Z = X.copy()
                Z[:, 1:] = (X[:, 1:] - mu) / sd
                b = fit_logit_w(Z[tr_idx], y[tr_idx], w)
                return b, mu, sd

            if do_refit:
                cache = fit(hist, s)
                last_fit = s
            b, mu, sd = cache
            if refit_games:
                pos = 0
                while pos < len(te):
                    chunk = te[pos:pos + refit_games]
                    Z = (X[chunk][:, 1:] - mu) / sd
                    pred[chunk] = _sig(b[0] + X[chunk][:, 0] * b[1] + Z @ b[2:])
                    pos += refit_games
                    if pos < len(te) and do_refit:
                        b, mu, sd = fit(np.concatenate([hist, te[:pos]]), s)
            else:
                Z = (X[te][:, 1:] - mu) / sd
                pred[te] = _sig(b[0] + X[te][:, 0] * b[1] + Z @ b[2:])
        p = pred[self.scored]
        assert np.all(np.isfinite(p))
        return p if ret_pred else llvec(y[self.scored], p)


H_HL = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, None]
H_WIN = [1, 2, 3, 5, 8, None]
H_REFIT_G = [None, 100, 250, 500, 1000]
H_REFIT_S = [1, 2, 3]


def axis_h(ctx):
    t0 = time.time()
    W = WFCtx(ctx, first=2005)
    out = {"baseline_wf_ll": round(W.base_ll, 6),
           "baseline_wf_fold_ll": [round(x, 6) for x in W.base_fold],
           "n_scored": int(W.scored.sum()),
           "note": ("walk-forward objective: seasons 2005-2015 scored with a blend "
                    "fitted only on strictly earlier games. NOT comparable to the "
                    "5-fold OOF number used by axes G and I.")}

    # (a) exponential recency half-life (seasons)
    cells, fl = [], []
    for hl in H_HL:
        v = W.run(half_life=hl)
        cells.append({"half_life_seasons": hl})
        fl.append(W.fold_means(v))
        print(f"    half_life={str(hl):<5} wf ll {W.pooled(fl[-1]):.6f}", flush=True)
    out["half_life"] = AX.nested_over_grid(W, np.array(fl), "H exponential half-life", cells)
    AX.show(out["half_life"])

    # (b) rectangular window (last W seasons) -- also the warm-start-length knob
    cells, fl = [], []
    for w in H_WIN:
        v = W.run(window=w)
        cells.append({"window_seasons": w})
        fl.append(W.fold_means(v))
        print(f"    window={str(w):<5} wf ll {W.pooled(fl[-1]):.6f}", flush=True)
    out["window"] = AX.nested_over_grid(W, np.array(fl), "H rectangular window", cells)
    AX.show(out["window"])

    # (c) refit frequency: intra-season (every N games) and stale (every k seasons)
    cells, fl = [], []
    for rg in H_REFIT_G:
        v = W.run(refit_games=rg)
        cells.append({"refit_every_games": rg, "refit_every_seasons": 1})
        fl.append(W.fold_means(v))
        print(f"    refit_games={str(rg):<5} wf ll {W.pooled(fl[-1]):.6f}", flush=True)
    for rs in H_REFIT_S[1:]:
        v = W.run(refit_seasons=rs)
        cells.append({"refit_every_games": None, "refit_every_seasons": rs})
        fl.append(W.fold_means(v))
        print(f"    refit_seasons={rs}    wf ll {W.pooled(fl[-1]):.6f}", flush=True)
    out["refit_cadence"] = AX.nested_over_grid(W, np.array(fl), "H refit cadence", cells)
    AX.show(out["refit_cadence"])

    # (d) joint: half-life x window, the two knobs that actually interact
    cells, fl = [], []
    for hl, w in itertools.product(H_HL, H_WIN):
        v = W.run(half_life=hl, window=w)
        cells.append({"half_life_seasons": hl, "window_seasons": w})
        fl.append(W.fold_means(v))
    out["joint_hl_window"] = AX.nested_over_grid(
        W, np.array(fl), "H half-life x window (joint)", cells)
    AX.show(out["joint_hl_window"])

    # (e) warm-start: how many seasons of history before the blend is trustworthy?
    #     Scored span held FIXED at 2008-2015 so every variant is judged on the
    #     same games; the knob is how much history the fitter is allowed.
    ws = {}
    fixed = ctx.seas >= 2008
    for hist_cap in (1, 2, 3, 4, 6, None):
        v_all = W.run(window=hist_cap, ret_pred=True)
        sub = fixed[W.scored]
        ll = float(llvec(ctx.y[W.scored][sub], v_all[sub]).mean())
        ws[str(hist_cap)] = round(ll, 6)
        print(f"    history cap {str(hist_cap):<5} seasons -> ll on 2008-2015 {ll:.6f}", flush=True)
    out["warm_start_history_cap"] = {
        "ll_on_2008_2015": ws,
        "note": ("with the scored span fixed, 'warm-start length' and 'training "
                 "window' are the SAME knob; reported separately only to show the "
                 "identity explicitly.")}
    # per-season convergence of the walk-forward blend vs an all-DEV oracle fit
    conv = {}
    pw = W.run(ret_pred=True)
    Xf = W.X
    Zf = _stdz(Xf, np.ones(len(ctx.y), bool))
    bo = fit_logit_w(Zf, ctx.y)
    po = _pred(bo, Zf)
    for s in sorted(set(ctx.seas[W.scored].tolist())):
        m = (ctx.seas == s)[W.scored]
        conv[int(s)] = {"wf_ll": round(float(llvec(ctx.y[W.scored][m], pw[m]).mean()), 6),
                        "insample_oracle_ll": round(float(
                            llvec(ctx.y[ctx.seas == s], po[ctx.seas == s]).mean()), 6)}
    out["per_season_convergence"] = conv
    out["seconds"] = round(time.time() - t0)
    return out


# ==================================================================== AXIS I
def i_tier_average(ctx):
    """(i) Average the tier predictions instead of hard-selecting the full tier."""
    y, seas, groups = ctx.y, ctx.seas, ctx.groups
    P = {}
    for t, names in TIERS.items():
        X = np.column_stack([ctx.cols[k] for k in names])
        p = np.empty(len(y))
        for m in ctx.masks:
            tr = ~m
            Z = _stdz(X, tr)
            b = fit_logit_w(Z[tr], y[tr])
            p[m] = _pred(b, Z[m])
        P[t] = p
    out, cells, fl = {}, [], []
    variants = {
        "full_only (shipped)": lambda: P["full"],
        "mean_prob_3tier": lambda: (P["recal"] + P["recbp"] + P["full"]) / 3.0,
        "mean_logit_3tier": lambda: _sig((_lg(P["recal"]) + _lg(P["recbp"]) + _lg(P["full"])) / 3.0),
        "mean_logit_full_recbp": lambda: _sig((_lg(P["recbp"]) + _lg(P["full"])) / 2.0),
        "logit_0.75full_0.25recbp": lambda: _sig(0.75 * _lg(P["full"]) + 0.25 * _lg(P["recbp"])),
        "logit_0.9full_0.1recal": lambda: _sig(0.9 * _lg(P["full"]) + 0.1 * _lg(P["recal"])),
    }
    for nm, fn in variants.items():
        v = llvec(y, fn())
        s = M.summarize(f"I {nm}", ctx.base_ll_vec, v, seas, groups)
        out[nm] = s
        M.pr(s)
        cells.append({"variant": nm})
        fl.append(ctx.fold_means(v))
    out["_nested_choice"] = AX.nested_over_grid(ctx, np.array(fl), "I tier averaging", cells)
    AX.show(out["_nested_choice"])
    out["note"] = ("in DEV every scored game has a complete feature set, so tier "
                   "selection is degenerate (always 'full'); averaging tiers is "
                   "therefore a pure shrinkage device toward the simpler nested models.")
    return out


def i_bagging(ctx):
    """(ii) Seed-averaging. The shipped chain has NO stochastic component (the Elo
    and TrueSkill walks are deterministic given the fixed event order, and IRLS is
    deterministic), so there is nothing to seed-average. The closest legitimate
    analogue is bootstrap bagging of the blend, screened here instead."""
    y, seas, groups = ctx.y, ctx.seas, ctx.groups
    X = np.column_stack(M.full_cols(ctx.cols))
    out, cells, fl = {}, [], []
    for B, seed in ((10, 3), (50, 3), (200, 3), (200, 99)):
        p = np.empty(len(y))
        rng = np.random.default_rng(seed)
        for m in ctx.masks:
            tr = np.where(~m)[0]
            Z = _stdz(X, ~m)
            acc = np.zeros(int(m.sum()))
            for _ in range(B):
                idx = tr[rng.integers(0, len(tr), len(tr))]
                b = fit_logit_w(Z[idx], y[idx])
                acc += _pred(b, Z[m])
            p[m] = acc / B
        v = llvec(y, p)
        s = M.summarize(f"I bagged blend B={B} seed={seed}", ctx.base_ll_vec, v, seas, groups)
        out[f"B{B}_seed{seed}"] = s
        M.pr(s)
        cells.append({"bag_B": B, "seed": seed})
        fl.append(ctx.fold_means(v))
    out["_nested_choice"] = AX.nested_over_grid(ctx, np.array(fl), "I bagged blend", cells)
    AX.show(out["_nested_choice"])
    out["seed_audit"] = ("no stochasticity exists in the shipped chain: FIP-Elo and the "
                         "per-PA TrueSkill walk are deterministic given the fixed "
                         "Retrosheet event order, and the blend is fit by Newton-IRLS. "
                         "Seed-averaging has nothing to average; bagging is screened as "
                         "the nearest analogue.")
    return out


def i_recency_ensemble(ctx):
    """(iii) Average blends fitted on different recency windows (walk-forward)."""
    t0 = time.time()
    W = WFCtx(ctx, first=2005)
    preds = {}
    for hl in (1.0, 2.0, 3.0, 5.0, None):
        preds[str(hl)] = W.run(half_life=hl, ret_pred=True)
    for w in (3, 5, 8):
        preds[f"win{w}"] = W.run(window=w, ret_pred=True)
    yy = ctx.y[W.scored]
    sets = {
        "expanding only (baseline)": ["None"],
        "hl {3, None}": ["3.0", "None"],
        "hl {1,3,None}": ["1.0", "3.0", "None"],
        "hl {1,2,3,5,None}": ["1.0", "2.0", "3.0", "5.0", "None"],
        "win {3,5,8} + expanding": ["win3", "win5", "win8", "None"],
        "all 8": list(preds),
    }
    out, cells, fl = {}, [], []
    for nm, keys in sets.items():
        p = _sig(np.mean([_lg(preds[k]) for k in keys], axis=0))
        v = llvec(yy, p)
        fold = W.fold_means(v)
        d = W.base_fold - fold
        out[nm] = {"wf_ll": round(W.pooled(fold), 6),
                   "delta_vs_wf_baseline": round(W.pooled(W.base_fold) - W.pooled(fold), 6),
                   "fold_deltas": [round(float(x), 6) for x in d],
                   "fold_sd": round(float(np.std(d, ddof=1)), 6)}
        print(f"    {nm:<28} wf ll {out[nm]['wf_ll']:.6f} "
              f"d {out[nm]['delta_vs_wf_baseline']:+.6f}", flush=True)
        cells.append({"members": keys})
        fl.append(fold)
    out["_nested_choice"] = AX.nested_over_grid(W, np.array(fl), "I recency-window ensemble", cells)
    AX.show(out["_nested_choice"])
    out["seconds"] = round(time.time() - t0)
    return out


def axis_i(ctx):
    t0 = time.time()
    print("  (i) tier averaging", flush=True)
    ta = i_tier_average(ctx)
    print("  (ii) bagging / seed-averaging", flush=True)
    bg = i_bagging(ctx)
    print("  (iii) recency-window ensemble (walk-forward objective)", flush=True)
    re = i_recency_ensemble(ctx)
    grid, labels = [], []
    for nm in ta:
        if nm.startswith("_") or nm == "note":
            continue
        grid.append(ctx.base_fold - np.array(ta[nm]["fold_deltas"]))
        labels.append({"family": "tier_avg", "variant": nm})
    for nm in bg:
        if nm.startswith("_") or nm == "seed_audit":
            continue
        grid.append(ctx.base_fold - np.array(bg[nm]["fold_deltas"]))
        labels.append({"family": "bagging", "variant": nm})
    best = AX.nested_over_grid(ctx, np.array(grid), "I best ensembling variant (5-fold obj)", labels)
    AX.show(best)
    return {"tier_average": ta, "bagging": bg, "recency_ensemble": re,
            "axis_best": best, "seconds": round(time.time() - t0)}


# =================================================================== summary
def summary(_ctx=None):
    """Merge the three axes into one ranked, honest verdict table.

    Also decomposes the apparent calibration 'staleness cost' from axis G(iii):
    the fresh curve is fitted IN SAMPLE on the same games it is scored on, so it
    carries an automatic ~k/(2n) advantage. Subtracting that is the difference
    between measuring drift and measuring the oracle's cheat."""
    rep = json.load(open(REPORT2))
    G, H, I = rep["axis_g"], rep["axis_h"], rep["axis_i"]

    n_late = G["drift"]["stale_vs_fresh"]["fit_2013_2015_oracle"]["n_train"]
    aic = {}
    for nm, k, meas in (("platt", 2, G["drift"]["staleness_cost"]["platt_2002_2008_minus_oracle"]),
                        ("full_blend", 7, G["drift"]["staleness_cost"]["full_2002_2008_minus_oracle"])):
        exp = k / (2.0 * n_late)
        aic[nm] = {"k_params": k, "n_scored": n_late,
                   "measured_stale_minus_fresh": round(meas, 6),
                   "expected_from_in_sample_fit_alone": round(exp, 6),
                   "residual_true_drift": round(meas - exp, 6)}

    rows = []
    for axis, node, obj, noise in (
            ("G calibration", G["axis_best"], "5-fold OOF, full tier", G["axis_best"]["fold_sd_nested"]),
            ("H training cadence", H["refit_cadence"], "walk-forward 2005-2015", H["refit_cadence"]["fold_sd_nested"]),
            ("I ensembling", I["axis_best"], "5-fold OOF, full tier", I["axis_best"]["fold_sd_nested"])):
        nd = node["nested_delta"]
        rows.append({
            "axis": axis, "objective": obj,
            "best_candidate": node["best_params"],
            "raw_delta": node["raw_delta"], "nested_delta": nd,
            "optimism": node["optimism"],
            "fold_sd": noise,
            "nested_in_noise_units": (round(nd / noise, 2) if noise else None),
            "gate": GATE,
            "verdict": "ADOPT" if nd >= GATE else "NULL — not adopted, no TEST look spent",
        })
    out = {
        "rows": rows,
        "drift_decomposition": aic,
        "headline": ("axes G, H and I are all NULL. Every nested estimate is negative; "
                     "the largest raw gain on any axis is +0.000008 (H, intra-season "
                     "refit every 250 games), 250x below the +0.0020 gate and negative "
                     "once re-selected honestly."),
        "structural_findings": [
            "Platt calibration of the Elo channel is ALGEBRAICALLY the shipped blend: a "
            "logistic with an intercept absorbs any affine map of one of its columns "
            "exactly. Measured delta 0.000000 confirms it. The same identity means a "
            "post-hoc Platt fitted on the blend's own training predictions returns "
            "(b0,b1)=(0,1) to 8 decimals — it can only do work if fitted out of sample.",
            "There is NO calibration drift inside DEV. Per-season Platt slopes have "
            "Q=12.65 on 13 df (Q/df=0.97) and intercepts Q=13.92 on 13 df (Q/df=1.07): "
            "the season-to-season wobble is exactly sampling noise.",
            "The apparent 'staleness cost' of a 2002-2008 curve on 2013-2015 is the "
            "oracle's in-sample advantage, not drift: measured +0.000234 vs +0.000137 "
            "expected from fitting 2 parameters in sample on 7,290 games; +0.000537 vs "
            "+0.000480 expected for the 7-parameter blend. Residual true drift is "
            "under 0.0001 on both.",
            "Recency weighting is monotonically HARMFUL and the damage scales with how "
            "much data it throws away (half-life 0.5 seasons +0.000605 worse, window of "
            "1 season +0.001252 worse). With no drift to track, discarding history buys "
            "nothing and costs coefficient variance.",
            "Tier averaging is a strict loss: the full tier dominates, and averaging in "
            "the nested tiers costs up to +0.000525. In DEV every scored game has a "
            "complete feature set, so tier selection is degenerate and tier-averaging is "
            "pure shrinkage toward a worse model.",
            "Nothing in the shipped MLB chain is stochastic — FIP-Elo and the per-PA "
            "TrueSkill walk are deterministic given the fixed Retrosheet event order, "
            "and the blend is fit by Newton-IRLS — so seed-averaging has nothing to "
            "average. Bagging, the nearest analogue, is null-to-worse at every B.",
            "The blend is already well calibrated where it matters: 0 of 10 OOF "
            "probability deciles fall outside their 95% Wilson interval, ECE 0.0061, "
            "calibration slope 0.9930. The raw FIP-Elo it is built from is NOT: slope "
            "1.3148, 5 of 10 deciles outside CI, monotone under-dispersion peaking at "
            "+0.0365 (z=+4.43) in the top decile — repaired entirely by the single "
            "linear-in-logit term the blend already fits.",
        ],
        "caveat": ("nested CV inside DEV estimates transfer to another 2002-2015 season "
                   "block, not to 2016-2024. No TEST-era number was computed, printed or "
                   "looked at by this process."),
    }
    for r in rows:
        print(f"  {r['axis']:<20} raw {r['raw_delta']:+.6f}  nested {r['nested_delta']:+.6f}  "
              f"optimism {r['optimism']:+.6f}  ({r['nested_in_noise_units']} fold-sd)  "
              f"{r['verdict']}", flush=True)
    for nm, d in aic.items():
        print(f"  drift {nm:<11} measured {d['measured_stale_minus_fresh']:+.6f}  "
              f"in-sample-fit alone {d['expected_from_in_sample_fit_alone']:+.6f}  "
              f"residual {d['residual_true_drift']:+.6f}", flush=True)
    return out


# ==================================================================== driver
AXES2 = {"--axis-g": ("axis_g", axis_g),
         "--axis-h": ("axis_h", axis_h),
         "--axis-i": ("axis_i", axis_i),
         "--summary": ("summary", summary)}


def main(argv):
    mode = argv[0] if argv else "--axis-g"
    if mode == "--all":
        ctx = AX.Ctx()
        for m in ("--axis-g", "--axis-h", "--axis-i"):
            key, fn = AXES2[m]
            print(f"\n===== {key} =====", flush=True)
            save2(key, fn(ctx))
            print(f"wrote {REPORT2} [{key}]", flush=True)
        return
    key, fn = AXES2[mode]
    ctx = None if mode == "--summary" else AX.Ctx()
    save2(key, fn(ctx))
    print(f"wrote {REPORT2} [{key}]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
