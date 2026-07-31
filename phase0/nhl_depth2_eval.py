"""NHL DEPTH ROUND 2 — axes G (calibration), H (training cadence), I (ensembling).

DEV SCREENS ONLY.  Every scored row is hard-asserted to be inside the DEV era
(season <= DEV_END = 20172018); the TEST era (>= 20182019) is never loaded.

Reuses the Round-1 harness (phase0/nhl_depth_eval.py): the same Ctx, the same
LOSO folds over seasons 2011-12 .. 2017-18, the same IRLS logistic, the same
fold_stats.  Baseline = shipped 5-feature blend, DEV LOSO log-loss 0.67240.

Market-blind: no odds enter any feature or any training signal anywhere here.

  G  post-hoc calibration.  Verified first that NO calibration layer exists in
     data/nhl_model.json or phase0/nhl_serve.py.  Screens Platt / temperature /
     isotonic / beta / regional-hinge, fitted either on IN-SAMPLE train
     predictions (the "free lunch" case a logistic with an intercept already
     handles) or on genuinely OUT-OF-FOLD inner-LOSO train predictions, plus
     era-restricted calibration sets to test drift.  Emits the probability
     decile table with Wilson CIs.
  H  training cadence.  The shipped blend is fitted ONCE on all warm games.
     Screens walk-forward refitting (expanding / rolling window / season
     recency weights / in-season refits) against that single fit.
  I  ensembling.  Elo-only blend x full blend at many weights, plus averaging
     blends fitted under different recency windows.

Usage
  python -X utf8 phase0/nhl_depth2_eval.py --stage all
  python -X utf8 phase0/nhl_depth2_eval.py --stage g
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "phase0")
import nhl_depth_eval as R  # noqa: E402
from nhl_glicko2_eval import DEV_END, DEV_START, DEV_WARM_BEFORE, TEST_START, llv  # noqa: E402

OUT = "data/nhl_depth2_dev.json"
SHIP_ELO, SHIP_XG = R.SHIPPED_ELO, R.SHIPPED_XG
BLEND_C = R.BLEND_C
T0 = time.time()

assert DEV_END < TEST_START


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


# =========================================================== small utilities ==
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def irls(B, y, w=None, ridge=1e-8, iters=60):
    """Unpenalised (ridge only for conditioning) weighted logistic on design B."""
    B = np.asarray(B, float)
    n, d = B.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    beta = np.zeros(d)
    for _ in range(iters):
        p = _sig(B @ beta)
        g = B.T @ (w * (y - p)) - ridge * beta
        W = w * p * (1 - p) + 1e-12
        H = (B * W[:, None]).T @ B
        H[np.diag_indices(d)] += ridge + 1e-10
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        nrm = np.max(np.abs(step))
        if nrm > 10:
            step *= 10.0 / nrm
        beta += step
        if nrm < 1e-9:
            break
    return beta


# --------------------------------------------------------- calibrator zoo ----
def _basis(kind, p):
    z = _logit(p)
    o = np.ones(len(z))
    if kind == "platt":
        return np.column_stack([o, z])
    if kind == "temp":                       # slope only, no intercept
        return z[:, None]
    if kind == "beta":
        q = np.clip(p, 1e-9, 1 - 1e-9)
        return np.column_stack([o, np.log(q), -np.log(1 - q)])
    if kind.startswith("hinge@"):
        k = float(kind.split("@")[1])
        return np.column_stack([o, z, np.maximum(z - k, 0.0), np.maximum(-z - k, 0.0)])
    if kind.startswith("bend@"):             # slope change above/below 0 only
        k = float(kind.split("@")[1])
        return np.column_stack([o, z, np.maximum(np.abs(z) - k, 0.0) * np.sign(z)])
    raise ValueError(kind)


def calibrate(kind, p_fit, y_fit, p_app, w=None):
    """Fit calibrator `kind` on (p_fit, y_fit); return calibrated p_app."""
    if kind == "identity":
        return np.asarray(p_app, float)
    if kind == "iso":
        from sklearn.isotonic import IsotonicRegression
        m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        m.fit(np.asarray(p_fit, float), np.asarray(y_fit, float), sample_weight=w)
        return np.clip(m.predict(np.asarray(p_app, float)), 1e-4, 1 - 1e-4)
    b = irls(_basis(kind, p_fit), np.asarray(y_fit, float), w)
    return np.clip(_sig(_basis(kind, p_app) @ b), 1e-6, 1 - 1e-6)


CAL_KINDS = ["identity", "platt", "temp", "iso", "beta",
             "hinge@0.4", "hinge@0.9", "bend@0.25", "bend@0.5"]


# ------------------------------------------------------------- decile table --
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def decile_table(p, y, nbin=10):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    order = np.argsort(p)
    edges = np.array_split(order, nbin)
    rows = []
    for i, ix in enumerate(edges):
        n = len(ix)
        pred = float(p[ix].mean())
        obs = float(y[ix].mean())
        lo, hi = wilson(float(y[ix].sum()), n)
        rows.append({"decile": i + 1, "p_lo": round(float(p[ix].min()), 4),
                     "p_hi": round(float(p[ix].max()), 4), "n": n,
                     "pred": round(pred, 4), "obs": round(obs, 4),
                     "obs_ci95": [round(lo, 4), round(hi, 4)],
                     "resid": round(obs - pred, 4),
                     "flag": "MISCAL" if (pred < lo or pred > hi) else ""})
    return rows


def cal_slope(p, y):
    """y ~ a + b*logit(p).  b=1,a=0 is perfect calibration-in-the-large+slope."""
    b = irls(np.column_stack([np.ones(len(p)), _logit(p)]), np.asarray(y, float))
    return {"intercept": round(float(b[0]), 4), "slope": round(float(b[1]), 4)}


# ------------------------------------------------------------- nested select -
def nested_select(per_map, nmap, base_name, pool):
    """LOSO over the eval seasons: pick the best pool member WITHOUT the fold it
    is scored on.  Returns raw best, nested estimate and the optimism gap."""
    seas = sorted(nmap)
    tot = sum(nmap[s] for s in seas)

    def pooled(nm, ss):
        return sum(nmap[s] * per_map[nm][s] for s in ss) / sum(nmap[s] for s in ss)

    held, picks = 0.0, {}
    for s in seas:
        others = [o for o in seas if o != s]
        best = min(pool, key=lambda nm: pooled(nm, others))
        picks[str(s)] = best
        held += nmap[s] * per_map[best][s]
    nested = held / tot
    base = pooled(base_name, seas)
    raw_best = min(pool, key=lambda nm: pooled(nm, seas))
    raw = pooled(raw_best, seas)
    return {"raw_best": raw_best, "raw_best_ll": round(raw, 5),
            "raw_gain": round(base - raw, 5),
            "nested_ll": round(nested, 5), "nested_gain": round(base - nested, 5),
            "optimism": round((base - raw) - (base - nested), 5),
            "loso_picks": picks}


# =============================================================== context =====
class Env:
    """Round-1 Ctx plus a WIDE (pre-DEV-warm) matrix used only for TRAINING in
    the walk-forward cadence screens."""

    def __init__(self):
        self.ctx = R.Ctx()
        c = self.ctx
        self.folds = c.folds
        self.y = c.yv()
        self.X = c.X_for(SHIP_ELO, SHIP_XG)
        self.s = c.seasons[c.mask]
        R.assert_dev_only(self.s)
        # elo-only design (drop the xG column) for axis I
        self.X_elo = self.X[:, :-1]

        wide = ((c.seasons >= DEV_START) & (c.seasons <= DEV_END) & c.cover)
        assert c.seasons[wide].max() <= DEV_END < TEST_START, "TEST leaked into wide mask"
        saved = c.mask
        c._elo.clear(); c._xg.clear()
        c.mask = wide
        self.Xw = c.X_for(SHIP_ELO, SHIP_XG)
        self.yw = c.y[wide]
        self.sw = c.seasons[wide]
        c.mask = saved
        c._elo.clear(); c._xg.clear()
        assert self.sw.max() <= DEV_END and self.sw.min() >= DEV_START
        # eval rows inside the wide matrix
        self.ev = self.sw >= DEV_WARM_BEFORE
        assert int(self.ev.sum()) == len(self.y), "wide eval subset != Round-1 eval set"
        assert np.allclose(self.yw[self.ev], self.y)
        assert np.allclose(self.Xw[self.ev], self.X, atol=1e-9), "wide rebuild mismatch"
        # within-season game order (for in-season refit cadence)
        self.wseq = np.zeros(len(self.sw), int)
        for ss in sorted(set(self.sw.tolist())):
            m = self.sw == ss
            self.wseq[m] = np.arange(int(m.sum()))
        self.wtot = {ss: int((self.sw == ss).sum()) for ss in sorted(set(self.sw.tolist()))}
        self.nmap = {int(s): self.folds.n[s] for s in self.folds.list}
        self.fl = [int(s) for s in self.folds.list]
        print(f"[env] eval {len(self.y):,} games / {len(self.fl)} seasons "
              f"{self.fl[0]}..{self.fl[-1]}; wide train pool {len(self.yw):,} "
              f"from {self.sw.min()}  [{time.time()-T0:.0f}s]", flush=True)

    # ---- per-season scoring helpers ----
    def per_season(self, p):
        return {int(s): float(llv(self.y[self.s == s], p[self.s == s]).mean())
                for s in self.fl}

    def pooled(self, p):
        return float(llv(self.y, p).mean())

    def report(self, name, p, p_base):
        st = R.fold_stats(self.y, self.folds, p_base, p)
        st.update({"name": name, "dev": round(self.pooled(p), 5)})
        return st

    def loso_pred(self, X=None):
        X = self.X if X is None else X
        p = np.empty(len(self.y))
        for s in self.fl:
            tr = self.s != s
            w = R.fit_logit(X[tr], self.y[tr], BLEND_C)
            p[self.s == s] = _sig(X[self.s == s] @ w)
        return p

    def inner_oof(self, s, X=None):
        """Out-of-fold predictions for the 6 TRAINING seasons of outer fold s."""
        X = self.X if X is None else X
        tr = self.s != s
        p = np.full(len(self.y), np.nan)
        for u in self.fl:
            if u == s:
                continue
            m2 = (self.s != s) & (self.s != u)
            w = R.fit_logit(X[m2], self.y[m2], BLEND_C)
            mu = self.s == u
            p[mu] = _sig(X[mu] @ w)
        assert not np.isnan(p[tr]).any()
        return p, tr


# ================================== AXIS G ===================================
def axis_g(env):
    y, fl, s = env.y, env.fl, env.s
    p_base = env.loso_pred()
    base_per = env.per_season(p_base)
    out = {"no_existing_calibration_layer": {
        "checked": ["data/nhl_model.json", "phase0/nhl_serve.py",
                    "phase0/nhl_weekly.py"],
        "finding": "data/nhl_model.json holds only elo_cfg/glicko_cfg/xg_cfg and a "
                   "5-coefficient blend; no calibration map is stored or applied "
                   "anywhere in the serving path. Claim VERIFIED."},
        "baseline_dev": round(env.pooled(p_base), 5)}

    # ---- diagnostics FIRST (this is where the interesting question lives) ----
    out["decile_table_uncalibrated"] = decile_table(p_base, y)
    out["calibration_slope_overall"] = cal_slope(p_base, y)
    out["calibration_slope_by_season"] = {
        str(ss): cal_slope(p_base[s == ss], y[s == ss]) for ss in fl}
    # regional: is miscalibration concentrated in the tails or at the coin-flip?
    z = _logit(p_base)
    reg = {}
    for nm, m in (("underdog_home  p<0.45", p_base < 0.45),
                  ("coinflip 0.45-0.55", (p_base >= 0.45) & (p_base <= 0.55)),
                  ("fav 0.55-0.65", (p_base > 0.55) & (p_base <= 0.65)),
                  ("heavy fav p>0.65", p_base > 0.65)):
        if m.sum() > 30:
            lo, hi = wilson(float(y[m].sum()), int(m.sum()))
            reg[nm] = {"n": int(m.sum()), "pred": round(float(p_base[m].mean()), 4),
                       "obs": round(float(y[m].mean()), 4),
                       "obs_ci95": [round(lo, 4), round(hi, 4)],
                       "resid": round(float(y[m].mean() - p_base[m].mean()), 4)}
    out["regional_residuals"] = reg
    # likelihood-ratio style check: does a regional basis beat plain Platt
    # IN-SAMPLE on the pooled OOF predictions?  (upper bound on what any
    # regional recalibration could ever recover)
    lrt = {}
    for kind in ("platt", "hinge@0.4", "hinge@0.9", "bend@0.25", "beta", "iso"):
        pc = calibrate(kind, p_base, y, p_base)
        lrt[kind] = round(env.pooled(p_base) - env.pooled(pc), 5)
    out["in_sample_upper_bound_gain_by_family"] = {
        "note": "fit and scored on the SAME pooled OOF predictions -> an "
                "unattainable ceiling, quoted to bound the whole axis",
        "gain": lrt}

    # ---- honest screens -------------------------------------------------
    per_map, rows, keep_p = {"identity": base_per}, [], {}
    for kind in CAL_KINDS:
        if kind == "identity":
            continue
        for mode in ("insample", "oof", "oof_last3", "oof_last2"):
            nm = f"{kind} | {mode}"
            p = np.empty(len(y))
            ok = True
            for ss in fl:
                te = s == ss
                if mode == "insample":
                    tr = s != ss
                    w = R.fit_logit(env.X[tr], y[tr], BLEND_C)
                    pf = _sig(env.X[tr] @ w)
                    yf, wf = y[tr], None
                    pa = _sig(env.X[te] @ w)
                else:
                    poof, tr = env.inner_oof(ss)
                    if mode == "oof_last3":
                        keep = tr & np.isin(s, [o for o in fl if o != ss][-3:])
                    elif mode == "oof_last2":
                        keep = tr & np.isin(s, [o for o in fl if o != ss][-2:])
                    else:
                        keep = tr
                    pf, yf, wf = poof[keep], y[keep], None
                    w = R.fit_logit(env.X[tr], y[tr], BLEND_C)
                    pa = _sig(env.X[te] @ w)
                try:
                    p[te] = calibrate(kind, pf, yf, pa, wf)
                except Exception as exc:                    # noqa: BLE001
                    ok = False
                    print(f"  {nm}: FAILED {exc}", flush=True)
                    break
            if not ok:
                continue
            per_map[nm] = env.per_season(p)
            keep_p[nm] = p
            st = env.report(nm, p, p_base)
            rows.append(st)
            print(f"  {nm:<28} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
                  f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)
    rows.sort(key=lambda r: -r["gain"])
    out["rows"] = rows
    out["nested"] = nested_select(per_map, env.nmap, "identity", list(per_map))
    top = rows[0]["name"]
    out["decile_table_after_best_honest_calibrator"] = {
        "calibrator": top, "table": decile_table(keep_p[top], y),
        "calibration_slope": cal_slope(keep_p[top], y)}
    return out, per_map, p_base


# ================================== AXIS H ===================================
def axis_h(env, p_base):
    y, fl, s = env.y, env.fl, env.s
    yw, sw, Xw, ev = env.yw, env.sw, env.Xw, env.ev
    wide_seasons = sorted(set(sw.tolist()))
    out = {"baseline_dev": round(env.pooled(p_base), 5),
           "baseline_protocol": "blend fitted on the OTHER 6 eval seasons (LOSO) — "
                                "the DEV analogue of the shipped 'fit once on all "
                                "warm games'; it uses FUTURE seasons, so any "
                                "walk-forward variant is handicapped by data volume "
                                "as well as by cadence."}

    def fit_pred(tr_mask, te_mask, w=None):
        assert sw[tr_mask].max() <= DEV_END and sw[te_mask].max() <= DEV_END
        ww = R.fit_logit(Xw[tr_mask], yw[tr_mask], BLEND_C) if w is None else \
            _wfit(Xw[tr_mask], yw[tr_mask], w)
        return _sig(Xw[te_mask] @ ww)

    def _wfit(X, yy, w):
        """weighted IRLS with the same (unpenalised) objective as fit_logit."""
        return irls(X, yy, w)

    variants = {}

    def add(nm, p_wide):
        p = p_wide[ev]
        assert not np.isnan(p).any()
        variants[nm] = p

    # --- expanding walk-forward ---
    p = np.full(len(yw), np.nan)
    for ss in fl:
        tr = sw < ss
        te = sw == ss
        p[te] = fit_pred(tr, te)
    add("H1 walk-forward expanding", p)

    # --- rolling window of k seasons ---
    for k in (1, 2, 3, 4):
        p = np.full(len(yw), np.nan)
        for ss in fl:
            prev = [u for u in wide_seasons if u < ss][-k:]
            tr = np.isin(sw, prev)
            te = sw == ss
            p[te] = fit_pred(tr, te)
        add(f"H2 rolling window {k} season(s)", p)

    # --- expanding + season recency half-life ---
    for hl in (1.0, 2.0, 3.0, 5.0):
        p = np.full(len(yw), np.nan)
        for ss in fl:
            tr = sw < ss
            te = sw == ss
            age = np.array([wide_seasons.index(ss) - wide_seasons.index(u)
                            for u in sw[tr]], float)
            p[te] = fit_pred(tr, te, w=0.5 ** ((age - 1) / hl))
        add(f"H3 expanding + season half-life {hl:g}", p)

    # --- frozen: one fit on everything before an anchor, then never refit -----
    # NOTE: scored ONLY on seasons at/after the anchor, so the frozen fit can
    # never have seen a season it is graded on.  Because the eval set differs
    # per anchor, this lives in its own restricted block rather than the main
    # pool (mixing eval sets would make the nested selection meaningless).
    frozen = {}
    for anchor in (fl[1], fl[3], fl[4]):
        tr = sw < anchor
        assert sw[tr].max() < anchor, "frozen fit saw a season it is graded on"
        w = R.fit_logit(Xw[tr], yw[tr], BLEND_C)
        pf = _sig(Xw[ev] @ w)
        keep = [ssx for ssx in fl if ssx >= anchor]
        m = np.isin(s, keep)
        nn = {ssx: env.nmap[ssx] for ssx in keep}
        tt = sum(nn.values())
        frozen[f"frozen on seasons < {anchor} (scored {keep[0]}..{keep[-1]})"] = {
            "n_train_seasons": int(len(set(sw[tr].tolist()))),
            "scored_seasons": keep, "n_scored": int(m.sum()),
            "frozen_ll": round(float(llv(y[m], pf[m]).mean()), 5),
            "loso_baseline_ll_same_rows": round(float(llv(y[m], p_base[m]).mean()), 5),
            "walkforward_expanding_ll_same_rows": round(
                float(llv(y[m], variants["H1 walk-forward expanding"][m]).mean()), 5),
            "gain_vs_loso_baseline": round(
                float(llv(y[m], p_base[m]).mean() - llv(y[m], pf[m]).mean()), 5),
            "per_season_delta": {str(ssx): round(float(
                llv(y[s == ssx], p_base[s == ssx]).mean()
                - llv(y[s == ssx], pf[s == ssx]).mean()), 5) for ssx in keep},
            "_n_weighted_check": tt}
    out["H4_frozen_restricted"] = frozen

    # --- in-season refits (a cadence INCREASE over the shipped annual fit) ---
    for q in (2, 4, 8):
        p = np.full(len(yw), np.nan)
        for ss in fl:
            tot = env.wtot[ss]
            bounds = [int(round(tot * i / q)) for i in range(q + 1)]
            for i in range(q):
                lo, hi = bounds[i], bounds[i + 1]
                te = (sw == ss) & (env.wseq >= lo) & (env.wseq < hi)
                if not te.any():
                    continue
                tr = (sw < ss) | ((sw == ss) & (env.wseq < lo))
                p[te] = fit_pred(tr, te)
        add(f"H5 expanding + {q} in-season refits", p)

    # --- LOSO-with-recency (keeps the baseline's data volume, adds weighting) --
    for hl in (1.0, 2.0, 3.0):
        p = np.empty(len(y))
        for ss in fl:
            tr = s != ss
            age = np.abs(np.array([fl.index(u) for u in s[tr]], float) - fl.index(ss))
            w = 0.5 ** ((age - 1) / hl)
            b = irls(env.X[tr], y[tr], w)
            p[s == ss] = _sig(env.X[s == ss] @ b)
        variants[f"H6 LOSO + |season distance| half-life {hl:g}"] = p

    rows, per_map = [], {"BASELINE (single/LOSO fit)": env.per_season(p_base)}
    for nm, p in variants.items():
        per_map[nm] = env.per_season(p)
        st = env.report(nm, p, p_base)
        rows.append(st)
        print(f"  {nm:<44} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
              f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)
    rows.sort(key=lambda r: -r["gain"])
    out["rows"] = rows
    out["nested"] = nested_select(per_map, env.nmap, "BASELINE (single/LOSO fit)",
                                  list(per_map))
    # restricted comparison: last 4 folds, where every walk-forward variant has
    # >= 3 training seasons -> isolates CADENCE from DATA VOLUME
    late = fl[-4:]
    nlate = {ssx: env.nmap[ssx] for ssx in late}
    tot = sum(nlate.values())
    out["late4_only"] = {
        "seasons": late,
        "scores": {nm: round(sum(nlate[ssx] * per_map[nm][ssx] for ssx in late) / tot, 5)
                   for nm in per_map}}
    # mechanism: how much do the blend coefficients actually move fold to fold?
    W = []
    for ss in fl:
        tr = s != ss
        W.append(R.fit_logit(env.X[tr], y[tr], BLEND_C))
    W = np.array(W)
    names = ["intercept", "elo_logit", "rest", "b2b_home", "b2b_away", "xg"]
    out["coefficient_stability_across_folds"] = {
        n: {"mean": round(float(W[:, i].mean()), 4),
            "sd": round(float(W[:, i].std(ddof=1)), 4),
            "cv": round(float(W[:, i].std(ddof=1) / abs(W[:, i].mean())), 3)
            if abs(W[:, i].mean()) > 1e-9 else None}
        for i, n in enumerate(names)}
    return out, per_map


# ================================== AXIS I ===================================
def axis_i(env, p_base, h_per_map):
    y, fl, s = env.y, env.fl, env.s
    out = {"baseline_dev": round(env.pooled(p_base), 5)}
    p_elo = env.loso_pred(env.X_elo)
    out["elo_only_blend_dev"] = round(env.pooled(p_elo), 5)
    rows, per_map = [], {"BASELINE (full blend)": env.per_season(p_base)}

    for w in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        for space, f in (("logit", lambda a, b, w=w: _sig((1 - w) * _logit(a) + w * _logit(b))),
                         ("prob", lambda a, b, w=w: (1 - w) * a + w * b)):
            nm = f"I1 full x elo-only w={w:g} ({space})"
            p = f(p_base, p_elo)
            per_map[nm] = env.per_season(p)
            st = env.report(nm, p, p_base)
            rows.append(st)

    # --- ensemble over recency windows (uses the axis-H variant predictions) ---
    # rebuild the member predictions we need
    yw, sw, Xw, ev = env.yw, env.sw, env.Xw, env.ev
    wide_seasons = sorted(set(sw.tolist()))
    members = {}
    for k in (1, 2, 3, 4, 99):
        p = np.full(len(yw), np.nan)
        for ss in fl:
            prev = [u for u in wide_seasons if u < ss][-k:]
            tr = np.isin(sw, prev)
            te = sw == ss
            assert sw[tr].max() <= DEV_END
            w_ = R.fit_logit(Xw[tr], yw[tr], BLEND_C)
            p[te] = _sig(Xw[te] @ w_)
        members[k] = p[ev]
    for combo in ((1, 2, 3), (2, 3, 4), (1, 2, 3, 4, 99), (3, 4, 99), (2, 4, 99)):
        nm = "I2 avg recency-window blends " + "+".join(str(c) for c in combo)
        p = np.mean([members[c] for c in combo], axis=0)
        per_map[nm] = env.per_season(p)
        st = env.report(nm, p, p_base)
        rows.append(st)
    # and the same members averaged in logit space
    for combo in ((1, 2, 3), (1, 2, 3, 4, 99)):
        nm = "I2L avg (logit) recency-window blends " + "+".join(str(c) for c in combo)
        p = _sig(np.mean([_logit(members[c]) for c in combo], axis=0))
        per_map[nm] = env.per_season(p)
        st = env.report(nm, p, p_base)
        rows.append(st)
    # --- shrink toward the base rate (a 1-parameter "ensemble with a constant") -
    pbar = float(y.mean())
    for w in (0.02, 0.05, 0.1):
        nm = f"I3 shrink toward base rate {pbar:.3f} w={w:g}"
        p = (1 - w) * p_base + w * pbar
        per_map[nm] = env.per_season(p)
        st = env.report(nm, p, p_base)
        rows.append(st)

    for st in sorted(rows, key=lambda r: -r["gain"]):
        print(f"  {st['name']:<46} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
              f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)
    rows.sort(key=lambda r: -r["gain"])
    out["rows"] = rows
    out["nested"] = nested_select(per_map, env.nmap, "BASELINE (full blend)",
                                  list(per_map))
    return out, per_map


# =================================== main ====================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "g", "h", "i"])
    args = ap.parse_args()
    env = Env()
    pay = {}
    if os.path.exists(OUT):
        try:
            pay = json.load(open(OUT, encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            pay = {}
    p_base = env.loso_pred()
    print(f"\nBASELINE DEV LOSO log-loss {env.pooled(p_base):.5f} "
          f"(Round-1 headline 0.67240)", flush=True)

    if args.stage in ("all", "g"):
        print("\n[G] calibration", flush=True)
        g, _, p_base = axis_g(env)
        pay["axis_G_calibration"] = g
    hper = {}
    if args.stage in ("all", "h"):
        print("\n[H] training cadence", flush=True)
        h, hper = axis_h(env, p_base)
        pay["axis_H_cadence"] = h
    if args.stage in ("all", "i"):
        print("\n[I] ensembling", flush=True)
        i, _ = axis_i(env, p_base, hper)
        pay["axis_I_ensembling"] = i

    per0 = env.per_season(p_base)
    vals = np.array([per0[s] for s in env.fl])
    pay["_fold_noise"] = {
        "per_season_baseline_ll": {str(s): round(per0[s], 5) for s in env.fl},
        "sd_across_folds": round(float(vals.std(ddof=1)), 5),
        "se_of_mean": round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 5),
        "note": "1 UNPAIRED noise unit. Paired comparisons are far tighter and "
                "carry their own fold_se in every row below."}
    pay["_protocol"] = {
        "league": "NHL", "round": 2, "axes": "G calibration, H cadence, I ensembling",
        "dev_eval_seasons": env.fl, "n_dev_games": len(env.y),
        "train_pool_from_season": int(env.sw.min()),
        "test_era_never_touched": [TEST_START, 20252026],
        "eval": "leave-one-season-out CV on DEV only",
        "baseline_dev_ll": round(env.pooled(p_base), 5),
        "pre_registered_gate": 0.0005,
        "test_looks_spent": 0, "market_blind": True,
        "generated": time.strftime("%Y-%m-%d")}
    atomic_json(pay, OUT)
    print(f"wrote {OUT}  [{time.time()-T0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
