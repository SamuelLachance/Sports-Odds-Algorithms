"""NFL DEPTH ROUND 2 — axes G (calibration), H (training cadence), I (ensembling).

DEV SCREENS ONLY.  The main split's TEST era is season >= 2016 and is NEVER
loaded into any mask here: every training and scoring mask is hard-asserted to
sit strictly below MAIN_TEST_ERA.  The orchestrator owns TEST looks.

Harness: the shipped main-split model — 14 features, walk-forward yearly refit,
recency half-life 3 seasons, ridge C=100, trained from 1999, scored 2006-2015
(2,670 games, 10 season folds).  Round-1 baseline DEV log-loss 0.62292.
Round 1 already screened half-life x C jointly (null, shipped point on the
plateau), so axis H here is refit FREQUENCY and TRAINING-WINDOW length only.

Market-blind: no odds enter any feature or any training signal anywhere here.

  G  post-hoc calibration.  Verified first that NO calibration layer exists in
     the serving path (phase0/nfl_site_data.py stores a display-only decile
     table; nothing transforms the model probability).  Screens Platt /
     temperature / isotonic / beta / regional hinges, fitted on IN-SAMPLE train
     predictions vs genuinely OUT-OF-SAMPLE inner walk-forward predictions,
     plus era-restricted calibration sets to test drift.  Emits the decile
     observed-vs-predicted table with Wilson CIs.
  H  refit frequency (annual vs every-k-seasons vs frozen vs IN-SEASON refits)
     and training-window / warm-start length.
  I  averaging blends fitted under different recency half-lives and windows.

Usage
  python -X utf8 phase0/nfl_depth2_eval.py --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "phase0")
import nfl_depth_eval as R  # noqa: E402

OUT = "data/nfl_depth2_dev.json"
T0 = time.time()

games = R.games
seasons = R.seasons
y = R.y
llv = R.llv
MAIN_LO, MAIN_HI, MAIN_TEST_ERA = R.MAIN_LO, R.MAIN_HI, R.MAIN_TEST_ERA
HL_BASE, C_BASE = R.HL_BASE, R.C_BASE
TRAIN_FROM = 1999
assert MAIN_HI < MAIN_TEST_ERA

WEEK = np.array([g["week"] for g in games], int)
SCORED = (seasons >= MAIN_LO) & (seasons <= MAIN_HI)
assert seasons[SCORED].max() < MAIN_TEST_ERA, "scored mask leaked into TEST"
FOLDS = list(range(MAIN_LO, MAIN_HI + 1))


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


def irls(B, yy, w=None, ridge=1e-8, iters=60):
    B = np.asarray(B, float)
    n, d = B.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    beta = np.zeros(d)
    for _ in range(iters):
        p = _sig(B @ beta)
        g = B.T @ (w * (yy - p)) - ridge * beta
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


def _basis(kind, p):
    z = _logit(p)
    o = np.ones(len(z))
    if kind == "platt":
        return np.column_stack([o, z])
    if kind == "temp":
        return z[:, None]
    if kind == "beta":
        q = np.clip(p, 1e-9, 1 - 1e-9)
        return np.column_stack([o, np.log(q), -np.log(1 - q)])
    if kind.startswith("hinge@"):
        k = float(kind.split("@")[1])
        return np.column_stack([o, z, np.maximum(z - k, 0.0), np.maximum(-z - k, 0.0)])
    if kind.startswith("bend@"):
        k = float(kind.split("@")[1])
        return np.column_stack([o, z, np.maximum(np.abs(z) - k, 0.0) * np.sign(z)])
    raise ValueError(kind)


def calibrate(kind, p_fit, y_fit, p_app, w=None):
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
             "hinge@0.4", "hinge@0.9", "bend@0.25", "bend@0.6"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def decile_table(p, yy, nbin=10):
    p = np.asarray(p, float)
    yy = np.asarray(yy, float)
    order = np.argsort(p)
    rows = []
    for i, ix in enumerate(np.array_split(order, nbin)):
        n = len(ix)
        lo, hi = wilson(float(yy[ix].sum()), n)   # ties count as half a win
        rows.append({"decile": i + 1, "p_lo": round(float(p[ix].min()), 4),
                     "p_hi": round(float(p[ix].max()), 4), "n": n,
                     "pred": round(float(p[ix].mean()), 4),
                     "obs": round(float(yy[ix].mean()), 4),
                     "obs_ci95": [round(lo, 4), round(hi, 4)],
                     "resid": round(float(yy[ix].mean() - p[ix].mean()), 4),
                     "flag": "MISCAL" if (p[ix].mean() < lo or p[ix].mean() > hi) else ""})
    return rows


def cal_slope(p, yy):
    b = irls(np.column_stack([np.ones(len(p)), _logit(p)]), np.asarray(yy, float))
    return {"intercept": round(float(b[0]), 4), "slope": round(float(b[1]), 4)}


# --------------------------------------------------------- fold statistics ---
def per_season(p, sv, yv):
    return {int(s): float(llv(yv[sv == s], p[sv == s]).mean()) for s in FOLDS}


def fold_stats(p_base, p, sv, yv, seed=7):
    lb, lv = llv(yv, p_base), llv(yv, p)
    gain = float(lb.mean() - lv.mean())
    d = np.array([float(lb[sv == s].mean() - lv[sv == s].mean()) for s in FOLDS])
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    rng = np.random.default_rng(seed)
    diff = lb - lv
    bs = diff[rng.integers(0, len(diff), size=(4000, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"dev": round(float(lv.mean()), 5), "gain": round(gain, 5),
            "fold_sd": round(sd, 5), "fold_se": round(se, 5),
            "sigma": round(gain / se, 2) if se > 0 else 0.0,
            "n_pos_folds": int((d > 0).sum()), "n_folds": len(d),
            "per_season_delta": [round(x, 5) for x in d],
            "boot_ci": [round(float(lo), 5), round(float(hi), 5)],
            "sig": "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")}


def nested_select(per_map, base_name, pool, nmap):
    seas = FOLDS
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


# ================================== harness ==================================
class Env:
    def __init__(self):
        self.X, _ = R.build_main()
        assert self.X.shape[0] == len(games)
        self.sv = seasons[SCORED]
        self.yv = y[SCORED]
        self.nmap = {int(s): int((self.sv == s).sum()) for s in FOLDS}
        assert self.sv.max() < MAIN_TEST_ERA
        print(f"[env] scored {SCORED.sum():,} games, seasons {FOLDS[0]}..{FOLDS[-1]}, "
              f"train pool from {TRAIN_FROM}  [{time.time()-T0:.0f}s]", flush=True)

    # --- one logistic fit on an explicit mask -----------------------------
    def fit(self, tr_mask, anchor, HL=HL_BASE, C=C_BASE):
        from sklearn.linear_model import LogisticRegression
        tr = tr_mask & (y != 0.5)
        assert tr.any(), "empty train mask"
        assert seasons[tr].max() < MAIN_TEST_ERA, "TRAIN mask leaked into TEST era"
        w = (0.5 ** ((anchor - 1 - seasons[tr]) / HL) if np.isfinite(HL)
             else np.ones(int(tr.sum())))
        return LogisticRegression(C=C, max_iter=5000).fit(self.X[tr], y[tr],
                                                          sample_weight=w)

    def pred(self, m, mask):
        return m.predict_proba(self.X[mask])[:, 1]

    # --- shipped protocol: annual refit, expanding window from TRAIN_FROM ---
    def wf(self, HL=HL_BASE, C=C_BASE, refit_every=1, window=None, mid=0):
        """window: keep only the trailing `window` seasons of training data.
        refit_every k: coefficients frozen for k seasons at a time.
        mid m: m extra in-season refits that fold the current season's already
        played games into the training set."""
        p = np.full(len(games), np.nan)
        for s in FOLDS:
            assert s < MAIN_TEST_ERA
            s_fit = MAIN_LO + refit_every * ((s - MAIN_LO) // refit_every)
            lo = TRAIN_FROM if window is None else max(TRAIN_FROM, s_fit - window)
            base_tr = (seasons >= lo) & (seasons < s_fit)
            if mid <= 0:
                m = self.fit(base_tr, s_fit, HL, C)
                te = seasons == s
                assert seasons[te].max() < MAIN_TEST_ERA
                p[te] = self.pred(m, te)
            else:
                assert refit_every == 1, "in-season refit needs annual cadence"
                wks = np.unique(WEEK[seasons == s])
                cuts = [wks[int(round(len(wks) * i / (mid + 1)))] for i in range(1, mid + 1)]
                bounds = [-1] + list(cuts) + [10 ** 6]
                for i in range(len(bounds) - 1):
                    te = (seasons == s) & (WEEK > bounds[i]) & (WEEK <= bounds[i + 1])
                    if not te.any():
                        continue
                    tr = base_tr | ((seasons == s) & (WEEK <= bounds[i]))
                    assert seasons[tr].max() < MAIN_TEST_ERA
                    m = self.fit(tr, s, HL, C)
                    p[te] = self.pred(m, te)
        out = p[SCORED]
        assert not np.isnan(out).any()
        return out

    # --- inner walk-forward predictions of the TRAIN side of fold s --------
    def inner_oof(self, s, H, HL=HL_BASE, C=C_BASE):
        """Genuinely out-of-sample predictions for the H seasons before s,
        each produced by a model trained only on seasons before it."""
        lo = max(TRAIN_FROM + 2, s - H)
        p = np.full(len(games), np.nan)
        for u in range(lo, s):
            m = self.fit((seasons >= TRAIN_FROM) & (seasons < u), u, HL, C)
            te = seasons == u
            p[te] = self.pred(m, te)
        keep = (seasons >= lo) & (seasons < s)
        assert not np.isnan(p[keep]).any()
        assert seasons[keep].max() < s
        return p, keep


# ================================== AXIS G ===================================
def axis_g(env, p_base):
    sv, yv = env.sv, env.yv
    out = {"no_existing_calibration_layer": {
        "checked": ["phase0/nfl_site_data.py", "phase0/nfl_model_spec.py",
                    "phase0/nfl_final_board.py"],
        "finding": "the only 'calibration' key in the repo is a DISPLAY table of "
                   "hit-rate-by-bucket written into the site payload; no map is "
                   "fitted, stored or applied to a model probability anywhere. "
                   "Claim VERIFIED."},
        "baseline_dev": round(float(llv(yv, p_base).mean()), 5)}

    out["decile_table_uncalibrated"] = decile_table(p_base, yv)
    out["calibration_slope_overall"] = cal_slope(p_base, yv)
    out["calibration_slope_by_season"] = {
        str(s): cal_slope(p_base[sv == s], yv[sv == s]) for s in FOLDS}
    out["calibration_slope_by_era"] = {
        "2006-2010": cal_slope(p_base[sv <= 2010], yv[sv <= 2010]),
        "2011-2015": cal_slope(p_base[sv >= 2011], yv[sv >= 2011])}
    reg = {}
    for nm, m in (("big home dog p<0.35", p_base < 0.35),
                  ("home dog 0.35-0.45", (p_base >= 0.35) & (p_base < 0.45)),
                  ("coin-flip 0.45-0.55", (p_base >= 0.45) & (p_base <= 0.55)),
                  ("home fav 0.55-0.65", (p_base > 0.55) & (p_base <= 0.65)),
                  ("home fav 0.65-0.75", (p_base > 0.65) & (p_base <= 0.75)),
                  ("heavy home fav p>0.75", p_base > 0.75)):
        if m.sum() > 30:
            lo, hi = wilson(float(yv[m].sum()), int(m.sum()))
            reg[nm] = {"n": int(m.sum()), "pred": round(float(p_base[m].mean()), 4),
                       "obs": round(float(yv[m].mean()), 4),
                       "obs_ci95": [round(lo, 4), round(hi, 4)],
                       "resid": round(float(yv[m].mean() - p_base[m].mean()), 4)}
    out["regional_residuals"] = reg
    lrt = {}
    for kind in ("platt", "hinge@0.4", "hinge@0.9", "bend@0.25", "beta", "iso"):
        pc = calibrate(kind, p_base, yv, p_base)
        lrt[kind] = round(float(llv(yv, p_base).mean() - llv(yv, pc).mean()), 5)
    out["in_sample_upper_bound_gain_by_family"] = {
        "note": "fitted AND scored on the same pooled out-of-fold predictions -> "
                "an unattainable ceiling, quoted to bound the entire axis",
        "gain": lrt}

    per_map = {"identity": per_season(p_base, sv, yv)}
    keep_p = {}
    rows = []
    modes = [("insample", None), ("oof_H3", 3), ("oof_H5", 5), ("oof_H10", 10)]
    for kind in CAL_KINDS:
        if kind == "identity":
            continue
        for mode, H in modes:
            nm = f"{kind} | {mode}"
            p = np.full(len(games), np.nan)
            for s in FOLDS:
                te = seasons == s
                tr = (seasons >= TRAIN_FROM) & (seasons < s)
                m = env.fit(tr, s)
                pa = env.pred(m, te)
                if mode == "insample":
                    keep = tr & (y != 0.5)
                    pf, yf = env.pred(m, keep), y[keep]
                    wf = 0.5 ** ((s - 1 - seasons[keep]) / HL_BASE)
                else:
                    poof, keep = env.inner_oof(s, H)
                    pf, yf, wf = poof[keep], y[keep], None
                p[te] = calibrate(kind, pf, yf, pa, wf)
            pv = p[SCORED]
            assert not np.isnan(pv).any()
            per_map[nm] = per_season(pv, sv, yv)
            keep_p[nm] = pv
            st = fold_stats(p_base, pv, sv, yv)
            st["name"] = nm
            rows.append(st)
            print(f"  {nm:<26} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
                  f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)
    rows.sort(key=lambda r: -r["gain"])
    out["rows"] = rows
    out["nested"] = nested_select(per_map, "identity", list(per_map), env.nmap)
    top = rows[0]["name"]
    out["decile_table_after_best_honest_calibrator"] = {
        "calibrator": top, "table": decile_table(keep_p[top], yv),
        "calibration_slope": cal_slope(keep_p[top], yv)}
    return out


# ================================== AXIS H ===================================
def axis_h(env, p_base):
    sv, yv = env.sv, env.yv
    out = {"baseline_dev": round(float(llv(yv, p_base).mean()), 5),
           "baseline_protocol": "annual refit, expanding window from 1999, "
                                "recency half-life 3 seasons, C=100",
           "note": "half-life x C was screened jointly in Round 1 (null); this "
                   "axis holds HL=3, C=100 fixed and moves only CADENCE and "
                   "WINDOW LENGTH."}
    per_map = {"BASELINE annual refit / all history": per_season(p_base, sv, yv)}
    rows = []

    def add(nm, p):
        per_map[nm] = per_season(p, sv, yv)
        st = fold_stats(p_base, p, sv, yv)
        st["name"] = nm
        rows.append(st)
        print(f"  {nm:<46} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
              f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)

    for k in (2, 3, 5, 10):
        add(f"H1 refit every {k} season(s)", env.wf(refit_every=k))
    for W in (3, 5, 7, 10, 12):
        add(f"H2 training window = trailing {W} season(s)", env.wf(window=W))
    for m in (1, 3, 8):
        add(f"H3 annual + {m} in-season refit(s)", env.wf(mid=m))
    # window x flat weighting: does a hard window do what the half-life does?
    for W in (3, 5, 7):
        add(f"H4 window {W} + NO recency weight", env.wf(window=W, HL=float("inf")))
    # cold-start: how much does the pre-2006 warm history actually buy?
    for lo in (2000, 2002, 2004):
        p = np.full(len(games), np.nan)
        for s in FOLDS:
            tr = (seasons >= lo) & (seasons < s)
            mm = env.fit(tr, s)
            te = seasons == s
            p[te] = env.pred(mm, te)
        add(f"H5 warm-start from {lo} instead of 1999", p[SCORED])

    rows.sort(key=lambda r: -r["gain"])
    out["rows"] = rows
    out["nested"] = nested_select(per_map, "BASELINE annual refit / all history",
                                  list(per_map), env.nmap)
    # mechanism: coefficient drift across the 10 annual refits
    W = []
    for s in FOLDS:
        mm = env.fit((seasons >= TRAIN_FROM) & (seasons < s), s)
        W.append(np.r_[mm.intercept_, mm.coef_[0]])
    W = np.array(W)
    out["coefficient_stability_across_annual_refits"] = {
        f"col{i}" if i else "intercept": {
            "mean": round(float(W[:, i].mean()), 4),
            "sd": round(float(W[:, i].std(ddof=1)), 4),
            "range": [round(float(W[:, i].min()), 4), round(float(W[:, i].max()), 4)]}
        for i in range(W.shape[1])}
    out["effective_train_size"] = {
        str(s): round(float((0.5 ** ((s - 1 - seasons[(seasons >= TRAIN_FROM)
                                                      & (seasons < s)
                                                      & (y != 0.5)]) / HL_BASE)).sum()), 1)
        for s in FOLDS}
    return out


# ================================== AXIS I ===================================
def axis_i(env, p_base):
    sv, yv = env.sv, env.yv
    out = {"baseline_dev": round(float(llv(yv, p_base).mean()), 5)}
    per_map = {"BASELINE (HL=3)": per_season(p_base, sv, yv)}
    rows = []

    def add(nm, p):
        per_map[nm] = per_season(p, sv, yv)
        st = fold_stats(p_base, p, sv, yv)
        st["name"] = nm
        rows.append(st)

    hl_members = {}
    for hl in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, float("inf")):
        hl_members[hl] = env.wf(HL=hl)
    win_members = {}
    for W in (3, 5, 7, 10, None):
        win_members[W] = env.wf(window=W)

    combos = [(1.0, 3.0), (1.5, 3.0, 6.0), (1.0, 2.0, 3.0, 4.0, 6.0),
              (3.0, float("inf")), (1.0, 3.0, float("inf")),
              (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, float("inf"))]
    for c in combos:
        tag = "+".join("inf" if not np.isfinite(h) else f"{h:g}" for h in c)
        add(f"I1 avg HL {{{tag}}} (prob)", np.mean([hl_members[h] for h in c], axis=0))
        add(f"I1L avg HL {{{tag}}} (logit)",
            _sig(np.mean([_logit(hl_members[h]) for h in c], axis=0)))
    for c in [(3, 7, None), (5, 10, None), (3, 5, 7, 10, None)]:
        tag = "+".join("all" if w is None else str(w) for w in c)
        add(f"I2 avg window {{{tag}}} (prob)",
            np.mean([win_members[w] for w in c], axis=0))
    # cross-family: HL-ensemble x window-ensemble
    add("I3 avg {HL 1,3,inf} + {win 3,7,all} (prob)",
        np.mean([hl_members[h] for h in (1.0, 3.0, float("inf"))]
                + [win_members[w] for w in (3, 7, None)], axis=0))
    # shrink toward the base rate (1-parameter ensemble with a constant)
    pbar = float(yv.mean())
    for w in (0.02, 0.05, 0.1):
        add(f"I4 shrink toward base rate {pbar:.3f} w={w:g}", (1 - w) * p_base + w * pbar)
    # C-ensemble (regularisation diversity) as a control
    c_members = {c: env.wf(C=c) for c in (1.0, 10.0, 100.0, 1000.0)}
    add("I5 avg C {1,10,100,1000} (prob)", np.mean(list(c_members.values()), axis=0))

    for st in sorted(rows, key=lambda r: -r["gain"]):
        print(f"  {st['name']:<48} dev {st['dev']:.5f}  gain {st['gain']:+.5f} "
              f"({st['sigma']:+.2f} SE) {st['sig']}", flush=True)
    rows.sort(key=lambda r: -r["gain"])
    out["rows"] = rows
    out["nested"] = nested_select(per_map, "BASELINE (HL=3)", list(per_map), env.nmap)
    out["member_solo_dev"] = {
        **{f"HL={h:g}" if np.isfinite(h) else "HL=inf":
           round(float(llv(yv, p).mean()), 5) for h, p in hl_members.items()},
        **{f"window={w}": round(float(llv(yv, p).mean()), 5)
           for w, p in win_members.items()}}
    # how correlated are the members?  (an ensemble can only help if they differ)
    ms = list(hl_members.values())
    cc = np.corrcoef(np.array([_logit(m) for m in ms]))
    out["member_logit_correlation_min"] = round(float(cc[np.triu_indices(len(ms), 1)].min()), 5)
    return out


# ==================== RATINGS-ONLY HARNESS: calibration only =================
def axis_g_ratings_only():
    """Calibration screen on the SECOND NFL model (ratings-only, DEV 2017-2021).

    No engine walk is re-run.  Round 1 stored the per-game log-loss vector of
    the base config; llv is exactly invertible for y in {0,1}, so the shipped
    walk-forward out-of-fold probabilities are recovered exactly from it.  Ties
    (y=0.5) are not invertible and are dropped from this block only.
    """
    import os
    import tempfile
    pd_ = os.environ.get("NFL_DEPTH_PARTS",
                         os.path.join(tempfile.gettempdir(), "nfl_depth_parts"))
    zp = os.path.join(pd_, "axisA.npz")
    if not os.path.exists(zp):
        return {"skipped": "Round-1 axis-A store not present"}
    v = np.load(zp)["base"].astype(np.float64)
    sd_ = R.SEAS_D
    yd = R.Y_D
    sel = (sd_ >= R.SEL_LO) & (sd_ <= R.SEL_HI)
    assert sd_[sel].max() < R.RO_TEST_ERA, "ratings-only mask leaked into TEST"
    assert int(sel.sum()) == len(v), "stored vector does not match the DEV window"
    ys, ss = yd[sel], sd_[sel]
    keep = ys != 0.5
    yv = ys[keep]
    sv = ss[keep]
    lv = v[keep]
    p = np.where(yv == 1.0, np.exp(-lv), 1.0 - np.exp(-lv))
    assert np.allclose(-(yv * np.log(p) + (1 - yv) * np.log(1 - p)), lv, atol=1e-6)
    fl = sorted(set(sv.tolist()))
    nmap = {int(s): int((sv == s).sum()) for s in fl}
    base_ll = float(llv(yv, p).mean())
    out = {"n": int(len(yv)), "seasons": fl, "n_ties_dropped": int((~keep).sum()),
           "baseline_dev_ll_ties_dropped": round(base_ll, 5),
           "recovered_from": "Round-1 stored per-game log-loss (exact inversion)",
           "decile_table_uncalibrated": decile_table(p, yv),
           "calibration_slope_overall": cal_slope(p, yv),
           "calibration_slope_by_season": {
               str(s): cal_slope(p[sv == s], yv[sv == s]) for s in fl}}
    ub = {}
    for kind in ("platt", "hinge@0.4", "hinge@0.9", "beta", "iso"):
        ub[kind] = round(base_ll - float(llv(yv, calibrate(kind, p, yv, p)).mean()), 5)
    out["in_sample_upper_bound_gain_by_family"] = {
        "note": "fitted AND scored on the same pooled OOF predictions — an "
                "unattainable ceiling for the whole axis",
        "gain": ub}
    # honest: calibrator fitted on the out-of-fold predictions of PRIOR seasons
    per_map = {"identity": {int(s): float(llv(yv[sv == s], p[sv == s]).mean())
                            for s in fl}}
    rows = []
    for kind in ("platt", "temp", "iso", "beta", "hinge@0.4"):
        pc = p.copy()
        for s in fl[1:]:
            prior = sv < s
            assert sv[prior].max() < s
            pc[sv == s] = calibrate(kind, p[prior], yv[prior], p[sv == s])
        nm = f"{kind} | fitted on prior DEV seasons"
        per_map[nm] = {int(s): float(llv(yv[sv == s], pc[sv == s]).mean()) for s in fl}
        gain = base_ll - float(llv(yv, pc).mean())
        d = np.array([per_map["identity"][s] - per_map[nm][s] for s in fl])
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        rows.append({"name": nm, "dev": round(float(llv(yv, pc).mean()), 5),
                     "gain": round(gain, 5), "fold_sd": round(float(d.std(ddof=1)), 5),
                     "fold_se": round(se, 5),
                     "sigma": round(gain / se, 2) if se > 0 else 0.0,
                     "per_season_delta": [round(x, 5) for x in d],
                     "scored_2018_2021_only": round(
                         float(llv(yv[sv > fl[0]], pc[sv > fl[0]]).mean()), 5)})
        print(f"  [RO] {nm:<40} dev {rows[-1]['dev']:.5f} "
              f"gain {gain:+.5f} ({rows[-1]['sigma']:+.2f} SE)", flush=True)
    rows.sort(key=lambda r: -r["gain"])
    out["rows"] = rows
    seas = fl
    tot = sum(nmap[s] for s in seas)

    def pooled(nm, ssx):
        return sum(nmap[s] * per_map[nm][s] for s in ssx) / sum(nmap[s] for s in ssx)

    held, picks = 0.0, {}
    for s in seas:
        others = [o for o in seas if o != s]
        best = min(per_map, key=lambda nm: pooled(nm, others))
        picks[str(s)] = best
        held += nmap[s] * per_map[best][s]
    raw_best = min(per_map, key=lambda nm: pooled(nm, seas))
    out["nested"] = {"raw_best": raw_best,
                     "raw_gain": round(pooled("identity", seas) - pooled(raw_best, seas), 5),
                     "nested_ll": round(held / tot, 5),
                     "nested_gain": round(pooled("identity", seas) - held / tot, 5),
                     "loso_picks": picks}
    vals = np.array([per_map["identity"][s] for s in fl])
    out["fold_noise_unpaired"] = {
        "per_season_baseline_ll": {str(s): round(per_map["identity"][s], 5) for s in fl},
        "sd_across_folds": round(float(vals.std(ddof=1)), 5),
        "se_of_mean": round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 5)}
    return out


# =================================== main ====================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "g", "h", "i", "ro"])
    args = ap.parse_args()
    env = Env()
    p_base = env.wf()
    ll0 = float(llv(env.yv, p_base).mean())
    print(f"\nBASELINE main-split DEV log-loss {ll0:.5f} (Round-1 0.62292)", flush=True)

    pay = {}
    if os.path.exists(OUT):
        try:
            pay = json.load(open(OUT, encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            pay = {}
    if args.stage in ("all", "g"):
        print("\n[G] calibration", flush=True)
        pay["axis_G_calibration"] = axis_g(env, p_base)
    if args.stage in ("all", "h"):
        print("\n[H] training cadence", flush=True)
        pay["axis_H_cadence"] = axis_h(env, p_base)
    if args.stage in ("all", "i"):
        print("\n[I] ensembling", flush=True)
        pay["axis_I_ensembling"] = axis_i(env, p_base)
    if args.stage in ("all", "ro"):
        print("\n[G-ratings-only] calibration on the second NFL model", flush=True)
        pay["axis_G_calibration_ratings_only"] = axis_g_ratings_only()
    per0 = per_season(p_base, env.sv, env.yv)
    vals = np.array([per0[s] for s in FOLDS])
    pay["_fold_noise"] = {
        "per_season_baseline_ll": {str(s): round(per0[s], 5) for s in FOLDS},
        "sd_across_folds": round(float(vals.std(ddof=1)), 5),
        "se_of_mean": round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 5),
        "note": "1 UNPAIRED noise unit. Paired comparisons are far tighter and "
                "carry their own fold_se in every row below.",
        "baseline_run_to_run_jitter": {
            "observed_across_processes": [0.622953, 0.622962, 0.622980],
            "cause": "build_main() accumulates league EPA rates over dict iteration "
                     "order, so the 14-feature matrix reproduces only to ~3e-5 in "
                     "log-loss from process to process; WITHIN a process every "
                     "comparison below is paired against the baseline computed in "
                     "that same process, so the jitter cancels exactly",
            "size_vs_unpaired_se": "~1/200 of one unpaired fold SE — immaterial, but "
                                   "it is why this run reads ~0.62296 where Round 1 "
                                   "recorded 0.62292"}}
    pay["_protocol"] = {
        "league": "NFL", "round": 2, "harness": "main split (14 features)",
        "dev_train_from": TRAIN_FROM, "dev_scored": [MAIN_LO, MAIN_HI],
        "n_dev_games": int(SCORED.sum()),
        "test_era_never_touched": [MAIN_TEST_ERA, 2025],
        "baseline_dev_ll": round(ll0, 5), "pre_registered_gate": 0.0020,
        "test_looks_spent": 0, "market_blind": True,
        "generated": time.strftime("%Y-%m-%d")}
    atomic_json(pay, OUT)
    print(f"wrote {OUT}  [{time.time()-T0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
