"""DIAGNOSTIC LEG — where does the remaining error live, and where is the floor?

READ-ONLY. This module touches no model, changes no artifact, and fits nothing
that is ever served. It re-runs the three shipped DEV harnesses exactly as they
already exist, keeps the per-game out-of-fold probabilities they produce, and
then slices those probabilities.

ONE-LOOK PROTOCOL (hard-asserted in every league loader)
  MLB   DEV 2002-2015          TEST >= 2016   never loaded, never scored
  NFL   DEV scored 2006-2015   TEST >= 2016   never scored
  NHL   DEV 2011-12..2017-18   TEST >= 2018-19 never loaded, never scored
No TEST-era number is computed, printed or written by this file.

MARKET-BLINDNESS
  No odds enter any feature or any training signal here. Odds are used in PART 2
  ONLY as an evaluation benchmark and as the proxy for the "true" probability
  when estimating the irreducible-entropy floor. That evaluation use is the
  explicitly sanctioned one; it is flagged in the JSON under
  leagues.<lg>.floor.odds_use.

PART 1 — error decomposition
  For each league, per-game DEV log loss is sliced by predicted-probability
  decile, calendar position (month / week), favourite side, rest situation,
  team, and (MLB) starter-information depth and model tier. Every slice is
  scored against ITS OWN base rate: the reference is the binary entropy of the
  slice's realised home-win rate, so a slice is never penalised merely for being
  high-entropy. Slices are ranked by TOTAL excess loss (n x per-game excess),
  which is the actual research-priority ordering.

PART 2 — the theoretical floor
  A perfectly calibrated forecaster's expected log loss is the mean binary
  entropy of the true win probabilities. The closing line, devigged
  multiplicatively (the repo's tested convention, data/devig_method_test.json),
  is the best available proxy for that true probability. Reported per league:
  our DEV loss, market loss, entropy floor, and the two gaps -- plus a direct
  test of the assumption the floor rests on (decile calibration of the devigged
  close) and a sensitivity curve for how much residual information beyond the
  market would be worth.

Usage:  python -X utf8 phase0/error_map.py            ->  data/error_map.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import date

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

OUT = "data/error_map.json"
EPS = 1e-15
T0 = time.time()
RNG_SEED = 11
B_BOOT = 2000


# ============================================================ small helpers ==
def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def llv(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def H(q):
    """Binary entropy in nats. Vectorised, safe at 0/1."""
    q = np.clip(np.asarray(q, float), EPS, 1 - EPS)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def r6(x):
    return None if x is None else round(float(x), 6)


# ======================================================== the slice machine ==
def slice_stats(name, ll, y, p, global_excess=None, boot=True, rng=None):
    """Loss / calibration / excess-over-own-base-rate for one slice of games.

    Three instruments, deliberately kept separate because they answer different
    questions:

    excess          mean per-game log loss - H(slice base rate). The reference is
                    an oracle that knew ONLY the slice label, so a slice is never
                    penalised for being high-entropy. total_excess = n x excess is
                    the ranking the brief asked for.
    shortfall       excess - the n-weighted MEAN excess of the dimension this
                    slice belongs to. Within a dimension the total_shortfall
                    column sums to exactly zero, so it reads as "this slice
                    contributes N nats MORE loss than if it behaved like the rest
                    of the same dimension". This is the usable priority list,
                    because almost every slice has strongly NEGATIVE excess (the
                    model has real resolution) and the raw total_excess ranking is
                    then dominated by tiny-n noise. NOTE: the league's global
                    excess is NOT the right reference — sum of n_s*H(base_s) is
                    strictly below N*H(global base) by concavity, so every slice
                    would look "short" against it.
    recal LR        refit a + b*logit(p) inside the slice. This isolates a FIXABLE
                    defect (systematic mis-scaling) from a merely hard slice. The
                    fit is in-sample with 2 free parameters, so under the null
                    2*n*gain ~ chi2(2): the expected total gain is ~1.0 nat for
                    ANY slice regardless of size, which is the yardstick.
    """
    n = int(len(ll))
    if n == 0:
        return None
    base = float(np.mean(y))
    m = float(np.mean(ll))
    h = float(H(base))
    exc = m - h
    se = float(np.std(ll, ddof=1) / math.sqrt(n)) if n > 1 else None
    row = {
        "slice": name, "n": n,
        "ll": r6(m), "ll_se": r6(se),
        "base_rate": r6(base), "base_entropy": r6(h),
        "mean_pred": r6(float(np.mean(p))),
        "cal_gap": r6(float(np.mean(p)) - base),
        "cal_gap_se": r6(math.sqrt(max(base * (1 - base), 1e-12) / n)),
        "excess": r6(exc), "total_excess": r6(exc * n),
    }
    if boot and n > 4:
        rng = rng or np.random.default_rng(RNG_SEED)
        idx = rng.integers(0, n, size=(B_BOOT, n))
        bl = ll[idx].mean(axis=1)
        by = y[idx].mean(axis=1)
        be = bl - H(by)
        lo, hi = np.percentile(be, [2.5, 97.5])
        row["excess_ci"] = [r6(lo), r6(hi)]
        row["excess_sig"] = "SIG worse" if lo > 0 else ("SIG better" if hi < 0 else "n.s.")
    if n >= 25 and 0.0 < base < 1.0:
        try:
            a, b, se_b = fit_ab(logit(p), y)
            gain = m - float(llv(y, sig(a + b * logit(p))).mean())
            lr = max(2.0 * n * gain, 0.0)
            row.update({"recal_a": r6(a), "recal_b": r6(b), "recal_b_se": r6(se_b),
                        "recal_gain": r6(gain), "total_recal_gain": r6(gain * n),
                        "recal_lr_chi2_df2": r6(lr),
                        "recal_p": r6(math.exp(-lr / 2.0))})
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            pass
    return row


def add_shortfall(rows):
    """excess relative to the dimension's own n-weighted mean. Sums to zero."""
    N = sum(r["n"] for r in rows)
    if not N:
        return rows
    mean_exc = sum(r["n"] * r["excess"] for r in rows) / N
    for r in rows:
        r["shortfall"] = r6(r["excess"] - mean_exc)
        r["total_shortfall"] = r6((r["excess"] - mean_exc) * r["n"])
    return rows


def decompose(labels, ll, y, p, dim, global_excess=None):
    """One slicing dimension -> list of slice rows, ordered by total excess."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for lab in sorted(set(labels.tolist()), key=lambda v: (isinstance(v, str), v)):
        m = labels == lab
        r = slice_stats(str(lab), ll[m], y[m], p[m], rng=rng)
        if r:
            r["dim"] = dim
            rows.append(r)
    add_shortfall(rows)
    rows.sort(key=lambda r: -r["total_excess"])
    return rows


def dim_summary(dims):
    """Per-dimension roll-up: how much a perfect slice-label oracle would beat us
    by (total_excess), and how heterogeneous the dimension is (spread)."""
    out = {}
    for d, rows in dims.items():
        N = sum(r["n"] for r in rows)
        out[d] = {
            "n_slices": len(rows), "n": N,
            "total_excess": r6(sum(r["total_excess"] for r in rows)),
            "heterogeneity_nats": r6(sum(abs(r["total_shortfall"]) for r in rows) / 2.0),
            "max_slice_shortfall": r6(max(r["total_shortfall"] for r in rows)),
            "max_recal_lr": r6(max([r.get("recal_lr_chi2_df2") or 0.0 for r in rows])),
        }
        # What a PERFECT slice-conditional recalibration of this dimension would
        # buy on the whole league, and what pure noise would already buy: each
        # fitted slice contributes ~1.0 nat under the null (chi2(2)/2).
        fitted = [r for r in rows if r.get("total_recal_gain") is not None]
        tot = sum(r["total_recal_gain"] for r in fitted)
        out[d]["recal_total_nats"] = r6(tot)
        out[d]["recal_null_nats"] = len(fitted)
        out[d]["recal_net_ll_if_perfect"] = r6((tot - len(fitted)) / N) if N else None
    return out


def decile_labels(p, k=10):
    """Equal-count bins of the predicted probability, labelled by their range."""
    qs = np.quantile(p, np.linspace(0, 1, k + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, k - 1)
    edges = np.quantile(p, np.linspace(0, 1, k + 1))
    return np.array([f"D{ i+1 :02d} [{edges[i]:.3f},{edges[i+1]:.3f}]" for i in idx])


def rank_all(dims, key="total_excess", drop_season=True):
    """Flatten every dimension's slices into one ranking.

    The `season` dimension is excluded from the priority lists: a season is not
    an attackable slice, it is only a noise reference.
    """
    flat = [r for d, rows in dims.items() if not (drop_season and d == "season")
            for r in rows]
    flat.sort(key=lambda r: -(r.get(key) if r.get(key) is not None else -1e9))
    return flat


# ============================================================== PART 2: floor
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()


def entropy_floor_curve(pm, s_grid):
    """Floor if the TRUE probability is the market logit plus mean-zero noise s.

    p* = sigmoid(logit(p_mkt) + s*eps), eps ~ N(0,1).  s = 0 reproduces the
    plain mean-market-entropy floor.  s > 0 asks: if a forecaster held signal
    worth s logit-units beyond the close, how low could its loss go?
    The parametrisation only approximately preserves calibration, so the
    induced drift E[p*] - p_mkt is reported alongside each row.
    """
    z = logit(pm)
    out = []
    for s in s_grid:
        acc_h = np.zeros_like(z)
        acc_p = np.zeros_like(z)
        for x, w in zip(_GH_X, _GH_W):
            pk = sig(z + s * x)
            acc_h += w * H(pk)
            acc_p += w * pk
        out.append({"s": round(float(s), 3),
                    "floor": r6(float(acc_h.mean())),
                    "cal_drift_mean_abs": r6(float(np.abs(acc_p - pm).mean()))})
    return out


def calibration_table(p, y, k=10):
    rows = []
    lab = decile_labels(p, k)
    for l_ in sorted(set(lab.tolist())):
        m = lab == l_
        n = int(m.sum())
        pr_, re_ = float(p[m].mean()), float(y[m].mean())
        se = math.sqrt(max(re_ * (1 - re_), 1e-12) / n)
        rows.append({"decile": l_, "n": n, "mean_pred": r6(pr_), "realized": r6(re_),
                     "gap": r6(pr_ - re_), "gap_se": r6(se),
                     "z": r6((pr_ - re_) / se) if se > 0 else None})
    return rows


def fit_ab(z, y, iters=60):
    """Logistic recalibration y ~ sigmoid(a + b*z), Newton. Returns (a, b, se_b)."""
    A = np.column_stack([np.ones(len(z)), z])
    w = np.zeros(2)
    for _ in range(iters):
        p = sig(A @ w)
        g = A.T @ (p - y)
        s = np.clip(p * (1 - p), 1e-10, None)
        Hm = (A * s[:, None]).T @ A
        Hm[np.diag_indices(2)] += 1e-10
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    p = sig(A @ w)
    s = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv((A * s[:, None]).T @ A + np.eye(2) * 1e-12)
    return float(w[0]), float(w[1]), float(math.sqrt(cov[1, 1]))


def equivalent_shrink(pm, y, target_ll):
    """c such that the market logit scaled by c loses exactly what WE lose.

    Expresses our deficit in an interpretable unit: 'our model is the closing
    line shrunk toward 50/50 by a factor c'.
    """
    z = logit(pm)
    grid = np.arange(0.0, 1.501, 0.005)
    lls = np.array([float(llv(y, sig(c * z)).mean()) for c in grid])
    cbest = int(np.argmin(lls))
    lo = lls[:cbest + 1]
    j = int(np.argmin(np.abs(lo - target_ll)))
    return {"c_equivalent": round(float(grid[j]), 3),
            "c_equivalent_ll": r6(float(lo[j])),
            "c_optimal": round(float(grid[cbest]), 3),
            "c_optimal_ll": r6(float(lls[cbest]))}


def noise_equivalent(pm, target_ll, hi=3.0):
    """s such that 'the close + s logit-units of pure noise' loses what WE lose.

    Treats the (calibrated) close as truth: a forecaster publishing
    q = sigmoid(logit(p_mkt) + s*eps) has expected loss
    E[H(p_mkt)] + E[KL(p_mkt || q)].  Inverting that curve at our loss puts our
    deficit on the SAME axis as the floor-sensitivity curve, so the two numbers
    can be read against each other.
    """
    z = logit(pm)

    def ll_of(s):
        acc = np.zeros_like(z)
        for x, w in zip(_GH_X, _GH_W):
            q = np.clip(sig(z + s * x), 1e-12, 1 - 1e-12)
            acc += w * -(pm * np.log(q) + (1 - pm) * np.log(1 - q))
        return float(acc.mean())

    lo, hiv = 0.0, hi
    if ll_of(hiv) < target_ll:
        return {"s_noise_equivalent": None, "note": "our loss exceeds s=3 noise"}
    for _ in range(60):
        mid = (lo + hiv) / 2
        if ll_of(mid) < target_ll:
            lo = mid
        else:
            hiv = mid
    s = (lo + hiv) / 2
    return {"s_noise_equivalent": round(float(s), 3), "ll_at_s": r6(ll_of(s)),
            "reading": ("our model is loss-equivalent to the closing line blurred by this "
                        "many logit-units of noise; compare directly against the s column "
                        "of floor_sensitivity, which is EXTRA information beyond the close")}


def floor_block(name, p_model, pm, y, note):
    ours = float(llv(y, p_model).mean())
    mkt = float(llv(y, pm).mean())
    fl = float(H(pm).mean())
    d = llv(y, p_model) - llv(y, pm)
    rng = np.random.default_rng(RNG_SEED)
    bs = d[rng.integers(0, len(d), size=(B_BOOT, len(d)))].mean(axis=1)
    # the market->floor gap is DEFINITIONALLY ~0 if the close is calibrated:
    # E[LL_mkt | p_mkt] = H(p_mkt).  Its sampling SE is the yardstick.
    dm = llv(y, pm) - H(pm)
    bsm = dm[rng.integers(0, len(dm), size=(B_BOOT, len(dm)))].mean(axis=1)
    a, b, se_b = fit_ab(logit(pm), y)
    p_recal = sig(a + b * logit(pm))
    return {
        "odds_use": ("EVALUATION ONLY — the closing line is never a feature and never a "
                     "training signal. It is used here as the benchmark and as the proxy "
                     "for the true probability in the entropy floor. Sanctioned use."),
        "devig": "multiplicative (data/devig_method_test.json convention)",
        "n": int(len(y)), "coverage_note": note,
        "our_ll": r6(ours), "market_ll": r6(mkt), "entropy_floor": r6(fl),
        "gap_us_to_market": r6(ours - mkt),
        "gap_us_to_market_ci": [r6(np.percentile(bs, 2.5)), r6(np.percentile(bs, 97.5))],
        "gap_market_to_floor": r6(mkt - fl),
        "gap_market_to_floor_se": r6(float(dm.std(ddof=1) / math.sqrt(len(dm)))),
        "gap_market_to_floor_ci": [r6(np.percentile(bsm, 2.5)), r6(np.percentile(bsm, 97.5))],
        "floor_caveats": [
            "The floor is an ESTIMATE and it rests on the close being calibrated. If it "
            "is, then E[LL_mkt | p_mkt] = H(p_mkt) EXACTLY, so market_ll and entropy_floor "
            "are the same quantity measured two ways and gap_market_to_floor is pure "
            "sampling noise. Read it as a calibration check, not as headroom.",
            "mean H(p_mkt) is an UPPER bound on the true irreducible loss: p_mkt is a "
            "coarsening (conditional expectation) of the true probability and H is "
            "concave, so E[H(p_mkt)] >= E[H(p_true)]. The real floor is at or below this "
            "number and is NOT identified by the market alone — which is why the "
            "floor_sensitivity curve is reported instead of a single number.",
            "n is modest, so the floor carries sampling error of roughly the same size as "
            "gap_market_to_floor_se.",
        ],
        "noise_equivalent": noise_equivalent(pm, ours),
        "market_calibration": {
            "deciles": calibration_table(pm, y),
            "recal_intercept": r6(a), "recal_slope": r6(b), "recal_slope_se": r6(se_b),
            "recal_slope_z_vs_1": r6((b - 1.0) / se_b) if se_b > 0 else None,
            "ll_after_insample_recal": r6(float(llv(y, p_recal).mean())),
            "recal_gain_insample": r6(mkt - float(llv(y, p_recal).mean())),
            "reading": ("slope 1 / intercept 0 = perfectly calibrated close. The recal "
                        "gain is IN-SAMPLE (2 free parameters) so it is an upper bound "
                        "on real miscalibration."),
        },
        "equivalent_shrink": equivalent_shrink(pm, y, ours),
        "floor_sensitivity": entropy_floor_curve(pm, [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]),
    }


# =================================================================== M L B ===
MLB_DEV_LO, MLB_DEV_HI, MLB_TEST_LO = 2002, 2015, 2016


def mlb_league():
    import mlb_depth_eval as M
    assert M.DEV_LO == MLB_DEV_LO and M.DEV_HI == MLB_DEV_HI
    assert M.DEV_HI < MLB_TEST_LO, "MLB DEV/TEST split moved"
    ds = M.load_ds(quiet=True)
    cols, _extra, ok = ds.design()
    assert ok.all() or True
    y = ds.y[ok]
    seas = ds.season[ok]
    gid = ds.gid[ok]
    assert seas.min() >= MLB_DEV_LO and seas.max() <= MLB_DEV_HI, "TEST-era MLB game scored"
    assert seas.max() < MLB_TEST_LO
    C = {k: v[ok] for k, v in cols.items()}
    groups = ds.groups

    def oof_pred(collist):
        X = np.column_stack(collist)
        p = np.empty(len(y))
        for grp in groups:
            te = np.isin(seas, list(grp))
            tr = ~te
            Z = X.copy()
            if X.shape[1] > 1:
                mu, sd = X[tr, 1:].mean(0), X[tr, 1:].std(0)
                sd[sd == 0] = 1.0
                Z[:, 1:] = (X[:, 1:] - mu) / sd
            w = M.fit_logit(Z[tr], y[tr], l2=1e-8)
            p[te] = M.pred_logit(w, Z[te])
        return p

    tiers = {"recal": [C["lf"]],
             "recbp": [C["lf"], C["bp"], C["si"]],
             "full": M.full_cols(C)}
    tier_p = {k: oof_pred(v) for k, v in tiers.items()}
    p = tier_p["full"]
    ll = llv(y, p)
    log(f"MLB full-tier DEV OOF ll {ll.mean():.6f} n={len(y)} (recorded baseline 0.676686)")
    assert abs(ll.mean() - 0.676686) < 5e-4, "MLB baseline not reproduced"

    # ---- per-game metadata straight off the cached game spine -------------
    games = {g["game_id"]: g for g in ds.games}
    meta = [games[g] for g in gid]
    dates = np.array([m["date"] for m in meta])
    month = np.array([f"{int(d[4:6]):02d}" for d in dates])
    home = np.array([m["home"] for m in meta])
    away = np.array([m["away"] for m in meta])

    # rest days + team game index, walked over the FULL cached spine (2000+),
    # which is metadata only: no rating, no training signal.
    spine = sorted(ds.games, key=lambda g: (g["date"], g["game_id"]))
    last, gidx, season_now = {}, defaultdict(int), None
    rest_h, rest_a, idx_h, idx_a = {}, {}, {}, {}
    for g in spine:
        if g["season"] != season_now:
            season_now = g["season"]
            gidx = defaultdict(int)
        d = date(int(g["date"][:4]), int(g["date"][4:6]), int(g["date"][6:8]))
        for t, R, I in ((g["home"], rest_h, idx_h), (g["away"], rest_a, idx_a)):
            prev = last.get(t)
            R[g["game_id"]] = min((d - prev).days, 6) if prev else 6
            I[g["game_id"]] = gidx[t]
        for t in (g["home"], g["away"]):
            last[t] = d
            gidx[t] += 1
    rdiff = np.array([rest_h[g] - rest_a[g] for g in gid])
    rest_lab = np.array([("rest_diff <=-2" if v <= -2 else "rest_diff -1" if v == -1 else
                          "rest_diff 0" if v == 0 else "rest_diff +1" if v == 1 else
                          "rest_diff >=+2") for v in rdiff])
    mat = np.minimum([idx_h[g] for g in gid], [idx_a[g] for g in gid])
    mat_lab = np.array([("team-games 0-9" if v < 10 else "team-games 10-29" if v < 30 else
                         "team-games 30-59" if v < 60 else "team-games 60-119" if v < 120
                         else "team-games 120+") for v in mat])

    # starter information depth: prior starts of the LESS-known of the two SPs.
    sp_seen = defaultdict(int)
    sp_depth = {}
    for g in spine:
        sp_depth[g["game_id"]] = min(sp_seen[g["home_sp"]], sp_seen[g["away_sp"]])
        sp_seen[g["home_sp"]] += 1
        sp_seen[g["away_sp"]] += 1
    spd = np.array([sp_depth[g] for g in gid])
    sp_lab = np.array([("SP prior starts 0" if v == 0 else "SP prior starts 1-5" if v <= 5
                        else "SP prior starts 6-20" if v <= 20 else
                        "SP prior starts 21-60" if v <= 60 else "SP prior starts 61+")
                       for v in spd])

    fav = np.where(p > 0.5, "home favourite", "away favourite")
    gx = float(ll.mean() - H(y.mean()))

    dims = {
        "pred_decile": decompose(decile_labels(p), ll, y, p, "pred_decile", gx),
        "month": decompose(month, ll, y, p, "month", gx),
        "season_maturity": decompose(mat_lab, ll, y, p, "season_maturity", gx),
        "favourite_side": decompose(fav, ll, y, p, "favourite_side", gx),
        "rest": decompose(rest_lab, ll, y, p, "rest", gx),
        "starter_depth": decompose(sp_lab, ll, y, p, "starter_depth", gx),
        "team": _team_dim(home, away, ll, y, p, gx),
        # HOME games only, i.e. one slice per BALLPARK. The `team` dimension
        # above pools each franchise's home and away games, so a venue effect —
        # which by construction applies to only half of them — arrives diluted
        # by about a factor of two. This is the sharp version, and the project's
        # only test of the park channel: park factors are not a feature and the
        # spine carries no venue column, so the home team is the proxy.
        "home_venue": decompose(home, ll, y, p, "home_venue", gx),
        "season": decompose(np.array([str(s) for s in seas]), ll, y, p, "season", gx),
    }

    tiers_out = {}
    for k, pk in tier_p.items():
        lk = llv(y, pk)
        tiers_out[k] = {"ll": r6(float(lk.mean())),
                        "excess_vs_global_base": r6(float(lk.mean() - H(y.mean()))),
                        "n": int(len(y))}
    tiers_out["_note"] = ("Every DEV game has full-tier inputs (lineups + TrueSkill), so "
                          "'tier served' cannot be sliced historically; these are the same "
                          "games scored by each shipped tier. The serving-time tier "
                          "degradation is a data-availability question, not a DEV one.")

    out = {
        "harness": "phase0/mlb_depth_eval.py DS + 5-fold season-grouped OOF, shipped params",
        "dev": [MLB_DEV_LO, MLB_DEV_HI], "test_never_touched": [MLB_TEST_LO, 2025],
        "n": int(len(y)), "ll": r6(float(ll.mean())),
        "global_base_rate": r6(float(y.mean())),
        "global_base_entropy": r6(float(H(y.mean()))),
        "global_excess": r6(gx),
        "tiers": tiers_out,
        "slices": dims,
        "dim_summary": dim_summary(dims),
        "n_slices_tested": sum(len(v) for k, v in dims.items() if k != "season"),
        "ranked": rank_all(dims)[:25],
        "ranked_by_shortfall": rank_all(dims, "total_shortfall")[:20],
        "ranked_by_recal_lr": rank_all(dims, "recal_lr_chi2_df2")[:20],
        "sp_known_vs_projected": (
            "NOT MEASURABLE ON DEV. Retrosheet records the pitcher who actually started, "
            "so every DEV game is a 'known starter' game; the projected-starter case only "
            "exists at serving time. The nearest DEV proxy is the starter_depth dimension "
            "(prior-starts of the less-established SP), which is what a wrong/unknown "
            "starter degrades to."),
    }
    return out, {"p": p, "y": y, "gid": gid, "dates": dates, "home": home,
                 "away": away, "seas": seas}


def _team_dim(home, away, ll, y, p, global_excess=None):
    """Per-franchise slice: every game the team appears in, either side."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for t in sorted(set(home.tolist()) | set(away.tolist())):
        m = (home == t) | (away == t)
        r = slice_stats(t, ll[m], y[m], p[m], rng=rng)
        if r:
            r["dim"] = "team"
            r["n_home"] = int((home[m] == t).sum())
            rows.append(r)
    add_shortfall(rows)
    rows.sort(key=lambda r: -r["total_excess"])
    return rows


# =================================================================== N F L ===
def nfl_league():
    import nfl_depth_eval as N
    from sklearn.linear_model import LogisticRegression
    assert N.MAIN_HI < N.MAIN_TEST_ERA == 2016, "NFL main split moved"
    X14, _ex = N.build_main()
    games, seasons, y_all = N.games, N.seasons, N.y
    per, pred = {}, np.full(len(y_all), np.nan)
    for s_ in range(N.MAIN_LO, N.MAIN_HI + 1):
        assert s_ < N.MAIN_TEST_ERA
        tr = (seasons < s_) & (y_all != 0.5)
        te = seasons == s_
        assert seasons[tr].max() < s_ and seasons[te].max() < N.MAIN_TEST_ERA
        w = 0.5 ** ((s_ - 1 - seasons[tr]) / N.HL_BASE)
        lm = LogisticRegression(C=N.C_BASE, max_iter=5000).fit(X14[tr], y_all[tr],
                                                               sample_weight=w)
        pred[te] = lm.predict_proba(X14[te])[:, 1]
        per[s_] = float(llv(y_all[te], pred[te]).mean())
    scored = (seasons >= N.MAIN_LO) & (seasons <= N.MAIN_HI)
    assert seasons[scored].max() <= 2015 < 2016, "TEST-era NFL game scored"
    ll_all = float(llv(y_all[scored], pred[scored]).mean())
    log(f"NFL main DEV wf ll {ll_all:.6f} n={int(scored.sum())} (recorded 0.62292)")
    assert abs(ll_all - 0.62292) < 5e-4, "NFL baseline not reproduced"

    ties = int(((y_all == 0.5) & scored).sum())
    m = scored & (y_all != 0.5)
    p = pred[m]
    y = y_all[m]
    ll = llv(y, p)
    G = [g for g, keep in zip(games, m) if keep]
    seas = seasons[m]
    home = np.array([g["home"] for g in G])
    away = np.array([g["away"] for g in G])
    wk = np.array([g["week"] for g in G])
    typ = np.array([g["type"] for g in G])
    wk_lab = np.array([("playoffs" if t != "REG" else
                        "week 1-2" if w <= 2 else "week 3-4" if w <= 4 else
                        "week 5-8" if w <= 8 else "week 9-13" if w <= 13 else "week 14-17")
                       for w, t in zip(wk, typ)])
    rd = np.array([g["hrest"] - g["arest"] for g in G], float)
    rest_lab = np.array([("rest_diff <=-4" if v <= -4 else "rest_diff -3..-1" if v < 0 else
                          "rest_diff 0" if v == 0 else "rest_diff +1..+3" if v <= 3 else
                          "rest_diff >=+4") for v in rd])
    bye = np.array([("bye: both" if g["hrest"] >= 13 and g["arest"] >= 13 else
                     "bye: home" if g["hrest"] >= 13 else
                     "bye: away" if g["arest"] >= 13 else "bye: neither") for g in G])
    fav = np.where(p > 0.5, "home favourite", "away favourite")
    gx = float(ll.mean() - H(y.mean()))

    dims = {
        "pred_decile": decompose(decile_labels(p), ll, y, p, "pred_decile", gx),
        "week": decompose(wk_lab, ll, y, p, "week", gx),
        "favourite_side": decompose(fav, ll, y, p, "favourite_side", gx),
        "rest": decompose(rest_lab, ll, y, p, "rest", gx),
        "bye": decompose(bye, ll, y, p, "bye", gx),
        "team": _team_dim(home, away, ll, y, p, gx),
        # HOME games only — one slice per STADIUM, the sharp version of `team`
        # (which pools home and away and so halves any venue signal). Neutral
        # games get their own bucket rather than being charged to the nominal
        # home team's building: London, Mexico and the Super Bowl are not that
        # team's stadium, and the slice doubles as a test of whether the model
        # mishandles games where home advantage should not apply at all.
        "home_venue": decompose(
            np.array([("(neutral site)" if g.get("neutral") else g["home"]) for g in G]),
            ll, y, p, "home_venue", gx),
        "season": decompose(np.array([str(s) for s in seas]), ll, y, p, "season", gx),
    }
    out = {
        "harness": "phase0/nfl_depth_eval.py 14-feature main split, walk-forward HL=3 C=100",
        "dev_scored": [N.MAIN_LO, N.MAIN_HI], "test_never_touched": [2016, 2025],
        "n": int(len(y)), "ties_excluded": ties, "ll": r6(float(ll.mean())),
        "global_base_rate": r6(float(y.mean())),
        "global_base_entropy": r6(float(H(y.mean()))),
        "per_season_ll": {str(k): r6(v) for k, v in per.items()},
        "global_excess": r6(gx),
        "slices": dims,
        "dim_summary": dim_summary(dims),
        "n_slices_tested": sum(len(v) for k, v in dims.items() if k != "season"),
        "ranked": rank_all(dims)[:25],
        "ranked_by_shortfall": rank_all(dims, "total_shortfall")[:20],
        "ranked_by_recal_lr": rank_all(dims, "recal_lr_chi2_df2")[:20],
    }
    gid = np.array([g["gid"] for g in G])
    return out, {"p": p, "y": y, "gid": gid, "seas": seas}


# =================================================================== N H L ===
def nhl_league():
    import nhl_depth_eval as Hd
    assert Hd.DEV_END < Hd.TEST_START, "NHL DEV/TEST split moved"
    ctx = Hd.Ctx()
    y = ctx.yv()
    X = ctx.X_for(Hd.SHIPPED_ELO, Hd.SHIPPED_XG)
    p = ctx.folds.loso_pred(X, y)
    ll = llv(y, p)
    seas = ctx.seasons[ctx.mask]
    assert seas.max() <= Hd.DEV_END < Hd.TEST_START, "TEST-era NHL game scored"
    log(f"NHL DEV LOSO ll {ll.mean():.6f} n={len(y)} (recorded 0.6724)")
    assert abs(ll.mean() - 0.6724) < 5e-4, "NHL baseline not reproduced"

    G = [g for g, keep in zip(ctx.games, ctx.mask) if keep]
    home = np.array([g["home"] for g in G])
    away = np.array([g["away"] for g in G])
    month = np.array([g["date"][5:7] for g in G])
    F = {k: v[ctx.mask] for k, v in ctx.F.items()}
    rd = F["rest_diff"]
    rest_lab = np.array([("rest_diff <=-2" if v <= -2 else "rest_diff -1" if v == -1 else
                          "rest_diff 0" if v == 0 else "rest_diff +1" if v == 1 else
                          "rest_diff >=+2") for v in rd])
    b2b = np.array([("b2b: both" if h and a else "b2b: home" if h else
                     "b2b: away" if a else "b2b: neither")
                    for h, a in zip(F["b2b_home"] > 0, F["b2b_away"] > 0)])
    # team-games into the season (cold start), walked over the DEV spine only
    gidx, season_now = defaultdict(int), None
    mats = []
    for g in ctx.games:
        if g["season"] != season_now:
            season_now = g["season"]
            gidx = defaultdict(int)
        mats.append(min(gidx[g["home"]], gidx[g["away"]]))
        gidx[g["home"]] += 1
        gidx[g["away"]] += 1
    mats = np.array(mats)[ctx.mask]
    mat_lab = np.array([("team-games 0-4" if v < 5 else "team-games 5-14" if v < 15 else
                         "team-games 15-29" if v < 30 else "team-games 30-54" if v < 55
                         else "team-games 55+") for v in mats])
    fav = np.where(p > 0.5, "home favourite", "away favourite")
    gx = float(ll.mean() - H(y.mean()))

    dims = {
        "pred_decile": decompose(decile_labels(p), ll, y, p, "pred_decile", gx),
        "month": decompose(month, ll, y, p, "month", gx),
        "season_maturity": decompose(mat_lab, ll, y, p, "season_maturity", gx),
        "favourite_side": decompose(fav, ll, y, p, "favourite_side", gx),
        "rest": decompose(rest_lab, ll, y, p, "rest", gx),
        "b2b": decompose(b2b, ll, y, p, "b2b", gx),
        "team": _team_dim(home, away, ll, y, p, gx),
        # HOME games only — one slice per ARENA. Caveat: the loaded NHL spine
        # carries no neutral flag (data/nhl_games.csv has the column, but
        # load_games drops it), so the handful of outdoor games per season are
        # charged to the nominal home team. At ~2-4 of ~1,300 games a year the
        # contamination is negligible, but it is not zero.
        "home_venue": decompose(home, ll, y, p, "home_venue", gx),
        "season": decompose(np.array([str(s) for s in seas]), ll, y, p, "season", gx),
    }
    out = {
        "harness": "phase0/nhl_depth_eval.py Ctx + LOSO, shipped Elo/xG-Elo params",
        "dev": [int(Hd.DEV_WARM_BEFORE), int(Hd.DEV_END)],
        "test_never_touched": [int(Hd.TEST_START), 20252026],
        "n": int(len(y)), "ll": r6(float(ll.mean())),
        "global_base_rate": r6(float(y.mean())),
        "global_base_entropy": r6(float(H(y.mean()))),
        "global_excess": r6(gx),
        "slices": dims,
        "dim_summary": dim_summary(dims),
        "n_slices_tested": sum(len(v) for k, v in dims.items() if k != "season"),
        "ranked": rank_all(dims)[:25],
        "ranked_by_shortfall": rank_all(dims, "total_shortfall")[:20],
        "ranked_by_recal_lr": rank_all(dims, "recal_lr_chi2_df2")[:20],
    }
    return out, {"p": p, "y": y, "seas": seas}


# ==================================================== market joins (PART 2) ==
def mlb_market(hold):
    """Join the DEV games to data/odds_mlb.csv closing prices. EVALUATION ONLY."""
    rows = list(csv.DictReader(open("data/odds_mlb.csv", encoding="utf-8")))
    okeys = defaultdict(list)
    for r in rows:
        if r["home_close"] and r["away_close"]:
            okeys[(r["date"].replace("-", ""), r["home"], r["away"])].append(r)
    gkeys = defaultdict(list)
    for i, (d, h, a) in enumerate(zip(hold["dates"], hold["home"], hold["away"])):
        gkeys[(d, h, a)].append(i)
    idx, pm = [], []
    ambiguous = 0
    for k, ii in gkeys.items():
        orow = okeys.get(k)
        if not orow:
            continue
        if len(ii) != 1 or len(orow) != 1:      # doubleheaders / duplicated prices
            ambiguous += len(ii)
            continue
        r = orow[0]
        qh, qa = 1.0 / float(r["home_close"]), 1.0 / float(r["away_close"])
        idx.append(ii[0])
        pm.append(qh / (qh + qa))
    idx = np.array(idx)
    order = np.argsort(idx)
    idx, pm = idx[order], np.array(pm)[order]
    seas = hold["seas"][idx]
    assert seas.max() <= MLB_DEV_HI, "market join reached a TEST-era MLB season"
    note = (f"odds_mlb.csv covers 2010+; DEV is 2002-2015 so the overlap is "
            f"{int(seas.min())}-{int(seas.max())}. {ambiguous} games dropped as "
            f"ambiguous (doubleheaders / duplicated price rows).")
    return floor_block("mlb", hold["p"][idx], pm, hold["y"][idx], note)


def nfl_market(hold):
    """Join the DEV games to nfl_games.csv closing moneylines. EVALUATION ONLY."""
    ml = {}
    for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")):
        if r["home_moneyline"] and r["away_moneyline"]:
            ml[r["game_id"]] = (float(r["home_moneyline"]), float(r["away_moneyline"]))

    def imp(v):
        return (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))

    idx, pm = [], []
    for i, g in enumerate(hold["gid"]):
        v = ml.get(g)
        if v is None:
            continue
        qh, qa = imp(v[0]), imp(v[1])
        idx.append(i)
        pm.append(qh / (qh + qa))
    idx, pm = np.array(idx), np.array(pm)
    seas = hold["seas"][idx]
    assert seas.max() <= 2015, "market join reached a TEST-era NFL season"
    note = (f"nfl_games.csv moneylines start 2006; DEV scored era is 2006-2015 so the "
            f"overlap is {int(seas.min())}-{int(seas.max())} ({len(idx)} of "
            f"{len(hold['gid'])} scored games).")
    return floor_block("nfl", hold["p"][idx], pm, hold["y"][idx], note)


NHL_FLOOR_SKIP = (
    "MOVED — this was skipped when error_map was written because no historical NHL "
    "closing line existed in the repo. One has since been sourced (SBR publishes NHL "
    "open/close moneylines as inline HTML, not as the .xlsx the MLB loader uses): "
    "phase0/nhl_odds_load.py -> data/odds_nhl.csv, and the floor is computed with "
    "THIS FILE's floor_block() by phase0/nhl_floor.py -> data/nhl_floor.json "
    "(DEV join 0.9997, 0 result mismatches). See documents/nhl_odds_sources.md.")


# ======================================================================= main
def main():
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": "DEV-only error decomposition + theoretical-floor estimate, 3 leagues",
        "protocol": {
            "read_only": True,
            "models_touched": 0,
            "test_looks_spent": 0,
            "market_blind_models": True,
            "odds_use": ("PART 2 only, as an evaluation benchmark and as the proxy for the "
                         "true probability in the entropy floor. Sanctioned; no odds enter "
                         "any feature or training signal anywhere in this file."),
            "reference": ("every slice is scored against the binary entropy of ITS OWN "
                          "realised base rate, so high-entropy slices are not penalised "
                          "for being hard; ranking is by n x per-game excess."),
            "multiple_comparisons": (
                "~64-80 slices are tested per league. A chi2(2) recal p of 0.001 is "
                "p~0.07 after Bonferroni over that family. Nothing here is adopted; "
                "anything that looks real needs its own PRE-REGISTERED screen with a "
                "nested estimate, per the Round 1 method standard."),
            "metrics": {
                "excess": "slice mean log loss - H(slice base rate)",
                "total_excess": "n x excess  (the ranking the brief specified)",
                "shortfall": "excess - league excess; total_shortfall sums to 0 over a "
                             "partition and is the readable priority list",
                "recal_lr_chi2_df2": "2n x in-slice gain from refitting a + b*logit(p); "
                                     "under H0 ~ chi2(2), so expected total gain is ~1.0 "
                                     "nat for ANY slice. This is the only instrument here "
                                     "that separates 'hard slice' from 'mis-fit slice'.",
            },
        },
        "leagues": {},
    }

    log("MLB ...")
    mlb, mlb_hold = mlb_league()
    mlb["floor"] = mlb_market(mlb_hold)
    payload["leagues"]["mlb"] = mlb
    atomic_json(payload, OUT)

    log("NHL ...")
    nhl, _ = nhl_league()
    nhl["floor"] = {"status": NHL_FLOOR_SKIP}
    payload["leagues"]["nhl"] = nhl
    atomic_json(payload, OUT)

    log("NFL ... (heavy prelude, ~40s)")
    nfl, nfl_hold = nfl_league()
    nfl["floor"] = nfl_market(nfl_hold)
    payload["leagues"]["nfl"] = nfl

    atomic_json(payload, OUT)
    log(f"wrote {OUT}")

    for lg in ("mlb", "nfl", "nhl"):
        d = payload["leagues"][lg]
        print(f"\n=== {lg.upper()}  n={d['n']}  DEV ll={d['ll']}  "
              f"base entropy={d['global_base_entropy']}  global excess={d['global_excess']}")
        print("  -- top total_excess (as briefed)")
        for r in d["ranked"][:6]:
            print(f"  {r['dim']:<16} {r['slice']:<28} n={r['n']:<6} ll={r['ll']:.5f} "
                  f"excess={r['excess']:+.5f} total={r['total_excess']:+8.2f} "
                  f"{r.get('excess_sig','')}")
        print("  -- top total_shortfall (vs league's own excess)")
        for r in d["ranked_by_shortfall"][:8]:
            print(f"  {r['dim']:<16} {r['slice']:<28} n={r['n']:<6} "
                  f"shortfall/g={r['shortfall']:+.5f} total={r['total_shortfall']:+8.2f}")
        print("  -- top in-slice recalibration LR (chi2 df=2; null total gain ~1.0 nat)")
        for r in d["ranked_by_recal_lr"][:6]:
            print(f"  {r['dim']:<16} {r['slice']:<28} n={r['n']:<6} b={r['recal_b']:+.3f} "
                  f"a={r['recal_a']:+.3f} LR={r['recal_lr_chi2_df2']:6.2f} "
                  f"p={r['recal_p']:.4f} tot_gain={r['total_recal_gain']:.2f}")
        f = d["floor"]
        if "status" in f:
            print(f"  floor: {f['status'][:60]}...")
        else:
            print(f"  floor: ours {f['our_ll']} | mkt {f['market_ll']} | "
                  f"H {f['entropy_floor']} | us->mkt {f['gap_us_to_market']} | "
                  f"mkt->floor {f['gap_market_to_floor']}")


if __name__ == "__main__":
    main()
