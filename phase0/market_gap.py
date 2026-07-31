"""MARKET-GAP DECOMPOSITION — WHERE does the closing line beat us, and by how much?

The follow-up the floor analysis (phase0/error_map.py, documents/depth_program.md)
demands. That leg established the SIZE of the gap to the theoretical limit
(MLB +0.00288 on n=13,930; NFL +0.01172 on n=2,531) and concluded the residual is
missing INFORMATION rather than mis-specification. This file asks the only
remaining question: *which games* carry that gap, because the answer names the
information we lack.

READ-ONLY. No model is fitted, refitted or altered. No feature is created. No
artifact that is ever served is touched.

MARKET-BLINDNESS
  Odds are used here as an EVALUATION BENCHMARK ONLY — exactly the sanctioned use
  declared in error_map.py. The closing line never enters a feature, never enters a
  training signal, and no model in this repo sees it. Every number below is a
  diagnostic comparison of two already-existing probability streams.

ONE-LOOK PROTOCOL (hard-asserted)
  MLB   DEV 2002-2015   TEST >= 2016      never loaded, never scored
  NFL   DEV 2006-2015   TEST >= 2016      never scored
  NHL   no historical closing line exists in the repo -> NOT ANALYSED (same reason
        error_map.py skipped its floor; inventing a price would be worse than
        reporting nothing).
No TEST-era number is computed, printed or written by this file.

LOADERS
  The model predictions come from error_map.mlb_league() / error_map.nfl_league()
  unchanged — the same shipped DEV harnesses, the same OOF/walk-forward folds, the
  same baseline asserts. Their slicing labels are captured in flight by spying on
  error_map.decompose / error_map._team_dim, so every dimension here is *literally*
  the dimension the error map used. Nothing is re-derived. The two market joins are
  lifted verbatim from error_map.mlb_market / error_map.nfl_market with the join
  INDEX retained (the originals discard it), and both are hard-checked against
  data/error_map.json: n, our_ll and market_ll must reproduce to 1e-6 or this file
  refuses to run.

THE THREE INSTRUMENTS
  gap            mean per-game (our log loss - market log loss) inside a slice.
                 total_gap = n x gap is the nats of aggregate deficit the slice
                 explains, which is the ranking that answers "where does it live".
  KL bracket     if the CLOSE were the truth, the expected gap is exactly
                 E[KL(p_mkt || p_ours)] >= 0. If WE were the truth it is exactly
                 -E[KL(p_ours || p_mkt)] <= 0. Those two numbers bracket the
                 realised gap and are computed from the PROBABILITIES ALONE — no
                 outcomes. They are the null the raw gap must be read against,
                 because a slice where we disagree with the market more will show a
                 bigger gap MECHANICALLY even if nobody is better informed.
  lambda         (gap + KL_ours) / (KL_mkt + KL_ours) in [0,1]: the share of the
                 disagreement the market wins. 1.0 = the close is truth and we are
                 pure noise around it; 0.0 = we are truth; ~0.5 = the truth sits
                 between us and neither side dominates. lambda is the DISAGREEMENT-
                 NORMALISED instrument, and it is what makes slices comparable.

Usage:  python -X utf8 phase0/market_gap.py     ->  data/market_gap.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "phase0")

import error_map as EM  # noqa: E402  the loaders, reused as-is

OUT = "data/market_gap.json"
REF = "data/error_map.json"
B_BOOT = 2000
SEED = 11
T0 = time.time()

MLB_TEST_LO = 2016
NFL_TEST_LO = 2016


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


r6 = EM.r6
llv = EM.llv
logit = EM.logit
sig = EM.sig


def chi2_sf(x, df):
    """Upper tail of chi2. scipy if available, Wilson-Hilferty otherwise."""
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:
        if df <= 0:
            return float("nan")
        z = ((x / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
        return float(0.5 * math.erfc(z / math.sqrt(2.0)))


# ======================================================= label capture (spy) ==
CAP: dict[str, dict] = {}
_LEAGUE = None
_real_decompose = EM.decompose
_real_team = EM._team_dim


def _stub(dim):
    return [{"slice": "captured", "dim": dim, "n": 1, "excess": 0.0,
             "total_excess": 0.0, "shortfall": 0.0, "total_shortfall": 0.0,
             "ll": 0.0}]


def _spy_decompose(labels, ll, y, p, dim, global_excess=None):
    CAP[_LEAGUE][dim] = np.asarray(labels)
    return _stub(dim)


def _spy_team(home, away, ll, y, p, global_excess=None):
    CAP[_LEAGUE]["_home"] = np.asarray(home)
    CAP[_LEAGUE]["_away"] = np.asarray(away)
    return _stub("team")


def load_league(name, fn):
    """Run a shipped error_map DEV harness, keeping predictions AND its labels.

    The slicing dimensions are intercepted rather than recomputed, so this file
    cannot drift from the error map even by accident. The stub returns keep the
    harness's own bookkeeping alive without paying for its bootstraps.
    """
    global _LEAGUE
    _LEAGUE = name
    CAP[name] = {}
    EM.decompose = _spy_decompose
    EM._team_dim = _spy_team
    try:
        _out, hold = fn()
    finally:
        EM.decompose = _real_decompose
        EM._team_dim = _real_team
        _LEAGUE = None
    return hold, CAP[name]


# ============================================================ market joins ====
def mlb_join(hold):
    """(index, devigged close) for DEV games in data/odds_mlb.csv.

    Lifted verbatim from error_map.mlb_market — same keys, same doubleheader
    rejection, same multiplicative devig — with the join index retained, which the
    original discards. Verified against data/error_map.json in check_join().
    EVALUATION ONLY.
    """
    rows = list(csv.DictReader(open("data/odds_mlb.csv", encoding="utf-8")))
    okeys = defaultdict(list)
    for r in rows:
        if r["home_close"] and r["away_close"]:
            okeys[(r["date"].replace("-", ""), r["home"], r["away"])].append(r)
    gkeys = defaultdict(list)
    for i, (d, h, a) in enumerate(zip(hold["dates"], hold["home"], hold["away"])):
        gkeys[(d, h, a)].append(i)
    idx, pm = [], []
    for k, ii in gkeys.items():
        orow = okeys.get(k)
        if not orow or len(ii) != 1 or len(orow) != 1:
            continue
        r = orow[0]
        qh, qa = 1.0 / float(r["home_close"]), 1.0 / float(r["away_close"])
        idx.append(ii[0])
        pm.append(qh / (qh + qa))
    idx = np.array(idx)
    order = np.argsort(idx)
    return idx[order], np.array(pm)[order]


def nfl_join(hold):
    """(index, devigged close) for DEV games with a closing moneyline in
    data/nfl_games.csv. Lifted verbatim from error_map.nfl_market with the index
    retained. EVALUATION ONLY."""
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
    return np.array(idx), np.array(pm)


def check_join(lg, p_o, p_m, y):
    """The join must reproduce the recorded floor block, or we stop.

    n and market_ll are pure data and must match to 1e-6 — the join itself is
    exact. our_ll is allowed 3e-4: the NFL walk-forward refits sklearn's lbfgs
    logistic on every call and its convergence jitter moves the 5th decimal
    (0.622043 / 0.622097 / 0.622109 across three runs here; MLB is bit-identical).
    That jitter is ~150x smaller than the NFL gap being measured. The realised
    deltas are recorded in the payload rather than hidden by the tolerance.
    """
    ref = json.load(open(REF, encoding="utf-8"))["leagues"][lg]["floor"]
    assert int(ref["n"]) == len(y), f"{lg} join size {len(y)} != error_map {ref['n']}"
    ours, mkt = float(llv(y, p_o).mean()), float(llv(y, p_m).mean())
    assert abs(ref["market_ll"] - mkt) < 1e-6, f"{lg} market_ll {mkt:.8f} != {ref['market_ll']}"
    assert abs(ref["our_ll"] - ours) < 3e-4, f"{lg} our_ll {ours:.8f} != {ref['our_ll']}"
    return {"n": len(y), "our_ll": r6(ours), "market_ll": r6(mkt),
            "error_map_our_ll": ref["our_ll"], "error_map_market_ll": ref["market_ll"],
            "delta_our_ll_vs_error_map": r6(ours - ref["our_ll"]),
            "delta_market_ll_vs_error_map": r6(mkt - ref["market_ll"]),
            "reproduces_error_map_floor": True}


# ================================================================ statistics ==
def kl(pa, pb):
    """KL(pa || pb) per game, both binary."""
    pa = np.clip(pa, 1e-12, 1 - 1e-12)
    pb = np.clip(pb, 1e-12, 1 - 1e-12)
    return pa * np.log(pa / pb) + (1 - pa) * np.log((1 - pa) / (1 - pb))


class Frame:
    """One league's joined DEV games, everything precomputed per game."""

    def __init__(self, lg, p_o, p_m, y, labels, total_ref=None):
        self.lg = lg
        self.p_o, self.p_m, self.y = p_o, p_m, y
        self.ll_o = llv(y, p_o)
        self.ll_m = llv(y, p_m)
        self.d = self.ll_o - self.ll_m               # per-game gap, nats
        self.kl_m = kl(p_m, p_o)                     # gap if the CLOSE is truth
        self.kl_o = kl(p_o, p_m)                     # -gap if WE are truth
        self.labels = labels
        self.N = len(y)
        self.total = float(self.d.sum())
        self.rng = np.random.default_rng(SEED)

    def slice_row(self, name, m, dim):
        n = int(m.sum())
        if n == 0:
            return None
        d = self.d[m]
        gap = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        km, ko = float(self.kl_m[m].mean()), float(self.kl_o[m].mean())
        denom = km + ko
        lam = (gap + ko) / denom if denom > 1e-12 else None
        row = {
            "dim": dim, "slice": name, "n": n,
            "our_ll": r6(float(self.ll_o[m].mean())),
            "market_ll": r6(float(self.ll_m[m].mean())),
            "gap": r6(gap), "gap_se": r6(se),
            "total_gap": r6(gap * n),
            "share_of_total_gap": r6(gap * n / self.total) if self.total else None,
            "n_share": r6(n / self.N),
            "kl_if_market_truth": r6(km),
            "kl_if_we_truth": r6(ko),
            "lambda_market_credit": r6(lam),
            "mean_abs_disagree": r6(float(np.abs(self.p_o - self.p_m)[m].mean())),
            "base_rate": r6(float(self.y[m].mean())),
        }
        if n > 4:
            bi = self.rng.integers(0, n, size=(B_BOOT, n))
            bs = d[bi].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row["gap_ci"] = [r6(lo), r6(hi)]
            row["gap_sig"] = ("market SIG better" if lo > 0 else
                              "we SIG better" if hi < 0 else "n.s.")
            km_b = self.kl_m[m][bi].mean(axis=1)
            ko_b = self.kl_o[m][bi].mean(axis=1)
            lb = (bs + ko_b) / np.maximum(km_b + ko_b, 1e-12)
            row["lambda_ci"] = [r6(np.percentile(lb, 2.5)), r6(np.percentile(lb, 97.5))]
        return row

    def dim_table(self, dim, labels=None):
        lab = self.labels[dim] if labels is None else labels
        rows = []
        for v in sorted(set(lab.tolist()), key=lambda z: (isinstance(z, str), z)):
            r = self.slice_row(str(v), lab == v, dim)
            if r:
                rows.append(r)
        rows.sort(key=lambda r: -r["total_gap"])
        return rows

    def team_table(self):
        home, away = self.labels["_home"], self.labels["_away"]
        rows = []
        for t in sorted(set(home.tolist()) | set(away.tolist())):
            r = self.slice_row(t, (home == t) | (away == t), "team")
            if r:
                rows.append(r)
        rows.sort(key=lambda r: -r["total_gap"])
        return rows


def heterogeneity(rows, pooled_gap):
    """Is the gap the SAME everywhere in this dimension?

    Q = sum (gap_s - pooled)^2 / se_s^2 ~ chi2(k-1) under a uniform gap. This is
    the direct test of CONCENTRATED vs DIFFUSE, and it is the reason the answer
    does not depend on eyeballing a ranking.
    """
    use = [r for r in rows if r.get("gap_se") and r["gap_se"] > 0 and r["n"] >= 30]
    if len(use) < 2:
        return None
    Q = sum((r["gap"] - pooled_gap) ** 2 / r["gap_se"] ** 2 for r in use)
    df = len(use) - 1
    mult = sum(r["n"] for r in rows) / rows[0]["n"] if False else None  # noqa
    tot = sum(r["total_gap"] for r in rows)
    top = sorted(rows, key=lambda r: -r["total_gap"])
    ses = sorted(r["gap_se"] for r in use)
    med_se = ses[len(ses) // 2]
    return {
        "k_slices_tested": len(use), "df": df, "Q": r6(Q), "Q_over_df": r6(Q / df),
        "median_slice_gap_se": r6(med_se),
        "mde_slice_vs_pooled": r6(2.8 * med_se),
        "power_note": ("a slice gap must differ from the pooled gap by about "
                       f"{2.8 * med_se:.5f} nats to be detectable here; 'uniform' means "
                       "'no difference bigger than that', not 'identical'."),
        "p": r6(chi2_sf(Q, df)),
        "verdict": ("HETEROGENEOUS (gap differs by slice)" if chi2_sf(Q, df) < 0.05
                    else "uniform within noise"),
        "top_slice": top[0]["slice"],
        "top_slice_share": r6(top[0]["total_gap"] / tot) if tot else None,
        "top_slice_n_share": r6(top[0]["n"] / sum(r["n"] for r in rows)),
        "top3_share": r6(sum(r["total_gap"] for r in top[:3]) / tot) if tot else None,
        "top3_n_share": r6(sum(r["n"] for r in top[:3]) / sum(r["n"] for r in rows)),
        "lambda_range": [r6(min(r["lambda_market_credit"] for r in use)),
                         r6(max(r["lambda_market_credit"] for r in use))],
    }


def contrast(fr, m, name, why):
    """gap INSIDE a mask minus gap outside it, with a paired bootstrap CI.

    The brief's decisive cut. If the market's advantage is concentrated in the
    cold-start window, the information we lack is roster/offseason knowledge. If
    the contrast is zero, the missing information is per-game and arrives all
    season long.
    """
    a, b = fr.d[m], fr.d[~m]
    if len(a) < 20 or len(b) < 20:
        return None
    rng = np.random.default_rng(SEED)
    bs = (a[rng.integers(0, len(a), size=(B_BOOT, len(a)))].mean(axis=1)
          - b[rng.integers(0, len(b), size=(B_BOOT, len(b)))].mean(axis=1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {
        "contrast": name, "why": why,
        "n_in": int(m.sum()), "n_out": int((~m).sum()),
        "gap_in": r6(float(a.mean())), "gap_out": r6(float(b.mean())),
        "diff": r6(float(a.mean() - b.mean())), "diff_ci": [r6(lo), r6(hi)],
        "verdict": ("CONCENTRATED here" if lo > 0 else
                    "concentrated ELSEWHERE" if hi < 0 else "n.s. — no concentration"),
        "lambda_in": r6(float((a.mean() + fr.kl_o[m].mean())
                              / (fr.kl_m[m].mean() + fr.kl_o[m].mean()))),
        "lambda_out": r6(float((b.mean() + fr.kl_o[~m].mean())
                               / (fr.kl_m[~m].mean() + fr.kl_o[~m].mean()))),
        "share_of_total_gap": r6(float(a.sum()) / fr.total) if fr.total else None,
        "n_share": r6(float(m.mean())),
    }


def fit_glm(X, y, iters=80):
    """Newton logistic with an intercept column expected in X. Returns (w, se)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(X @ w)
        g = X.T @ (p - y)
        s = np.clip(p * (1 - p), 1e-10, None)
        Hm = (X * s[:, None]).T @ X + np.eye(X.shape[1]) * 1e-9
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    p = sig(X @ w)
    s = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv((X * s[:, None]).T @ X + np.eye(X.shape[1]) * 1e-9)
    return w, np.sqrt(np.diag(cov))


def encompassing(fr, m=None):
    """y ~ a + b_us*logit(p_ours) + b_mkt*logit(p_close).

    The classical forecast-encompassing test. b_us ~ 0 means the close already
    contains everything we know; b_us > 0 with a real t-stat means our model holds
    information the closing line does not, whatever the log-loss ranking says.
    """
    m = np.ones(fr.N, bool) if m is None else m
    zo, zm, y = logit(fr.p_o[m]), logit(fr.p_m[m]), fr.y[m]
    X = np.column_stack([np.ones(m.sum()), zo, zm])
    w, se = fit_glm(X, y)
    return {"n": int(m.sum()),
            "intercept": r6(w[0]),
            "b_ours": r6(w[1]), "b_ours_se": r6(se[1]), "b_ours_t": r6(w[1] / se[1]),
            "b_market": r6(w[2]), "b_market_se": r6(se[2]), "b_market_t": r6(w[2] / se[2]),
            "reading": ("b_ours is the weight the outcome data puts on OUR forecast once "
                        "the close is in the equation; t<2 means the close encompasses us.")}


def disagreement_table(fr):
    """The single most diagnostic cut: bucket |p_ours - p_close|.

    Reported with the KL bracket, because the raw gap is guaranteed to grow with
    disagreement even under a completely uninformative disagreement. Only lambda,
    and the realised gap read against kl_if_market_truth, say who was right.
    """
    a = np.abs(fr.p_o - fr.p_m)
    edges = [0.0, 0.01, 0.02, 0.04, 0.07, 0.12, 1.01]
    lab = np.array([f"|dp| {edges[min(np.searchsorted(edges, v, 'right') - 1, len(edges) - 2)]:.2f}"
                    f"-{edges[min(np.searchsorted(edges, v, 'right'), len(edges) - 1)]:.2f}"
                    for v in a])
    rows = fr.dim_table("disagreement", lab)
    for r in rows:
        # what the gap WOULD be if the close were truth, vs what it was
        r["gap_minus_kl_null"] = r6(r["gap"] - r["kl_if_market_truth"])
    rows.sort(key=lambda r: r["slice"])
    q = np.quantile(a, [0.0, 0.5, 0.8, 0.95, 1.0])
    qlab = np.array(["Q1 lowest 50%" if v <= q[1] else "Q2 50-80%" if v <= q[2]
                     else "Q3 80-95%" if v <= q[3] else "Q4 top 5%" for v in a])
    qrows = fr.dim_table("disagreement_quantile", qlab)
    qrows.sort(key=lambda r: r["slice"])
    # who wins the side-disagreements (the two forecasts pick different favourites)
    side = (fr.p_o > 0.5) != (fr.p_m > 0.5)
    sd = {"n": int(side.sum())}
    if side.sum() > 10:
        sd.update({
            "our_side_win_rate": r6(float(((fr.p_o[side] > 0.5) == (fr.y[side] > 0.5)).mean())),
            "market_side_win_rate": r6(float(((fr.p_m[side] > 0.5) == (fr.y[side] > 0.5)).mean())),
            "our_ll": r6(float(fr.ll_o[side].mean())),
            "market_ll": r6(float(fr.ll_m[side].mean())),
            "gap": r6(float(fr.d[side].mean())),
        })
    return {"buckets": rows, "quantiles": qrows, "opposite_favourite": sd,
            "encompassing_all": encompassing(fr),
            "encompassing_top20pct_disagreement": encompassing(fr, a > q[2])}


# ================================================= league-specific extra dims ==
def nfl_qb_change(gids):
    """Did a team start a different QB than in its own previous game?

    Built from data/nfl_games.csv home_qb_id / away_qb_id. This is metadata, not a
    feature: it is never fitted, only used to label games. The point of the cut is
    that a QB switch is the single largest publicly-known information shock in the
    NFL, and it is exactly the kind of thing a closing line prices immediately.
    """
    rows = [r for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8"))
            if r["home_score"] != ""]
    rows.sort(key=lambda r: (r["gameday"], r["game_id"]))
    lastqb, lab = {}, {}
    for r in rows:
        s = int(r["season"])
        ch = {}
        for side in ("home", "away"):
            t, q = r[f"{side}_team"], r[f"{side}_qb_id"]
            prev = lastqb.get((t, s))
            ch[side] = (prev is not None and q != "" and prev != q)
            if q:
                lastqb[(t, s)] = q
        first = all(lastqb.get((r[f"{s_}_team"], s)) is None for s_ in ("home", "away"))
        lab[r["game_id"]] = ("QB change: both" if ch["home"] and ch["away"] else
                             "QB change: home" if ch["home"] else
                             "QB change: away" if ch["away"] else
                             "QB change: none (opener)" if first else "QB change: none")
    return np.array([lab.get(g, "QB change: unknown") for g in gids])


# ======================================================================= main ==
def league_block(lg, fr, dims, extra_note, contrasts=()):
    tables, hetero = {}, {}
    for d in dims:
        rows = fr.team_table() if d == "team" else fr.dim_table(d)
        tables[d] = rows
        h = heterogeneity(rows, float(fr.d.mean()))
        if h:
            hetero[d] = h
    flat = [r for d, rows in tables.items() if d not in ("season", "team")
            for r in rows]
    flat.sort(key=lambda r: -r["total_gap"])
    return {
        "n": fr.N,
        "our_ll": r6(float(fr.ll_o.mean())),
        "market_ll": r6(float(fr.ll_m.mean())),
        "gap": r6(float(fr.d.mean())),
        "total_gap_nats": r6(fr.total),
        "gap_ci": _boot_ci(fr.d),
        "kl_if_market_truth": r6(float(fr.kl_m.mean())),
        "kl_if_we_truth": r6(float(fr.kl_o.mean())),
        "lambda_market_credit": r6(float((fr.d.mean() + fr.kl_o.mean())
                                         / (fr.kl_m.mean() + fr.kl_o.mean()))),
        "lambda_ci": _lambda_ci(fr),
        "gap_minus_kl_null": r6(float(fr.d.mean() - fr.kl_m.mean())),
        "lambda_reading": ("1.0 = the close is the truth and our disagreements are pure "
                           "noise around it; 0.5 = the truth sits midway and we are as "
                           "informative as the close; 0 = we are the truth. A CI that "
                           "excludes 1.0 is evidence we hold information the close lacks."),
        "coverage_note": extra_note,
        "slices": tables,
        "heterogeneity": hetero,
        "ranked_by_total_gap": flat[:20],
        "contrasts": [c for c in (contrast(fr, m, nm, why) for nm, m, why in contrasts)
                      if c],
        "disagreement": disagreement_table(fr),
    }


def _lambda_ci(fr):
    rng = np.random.default_rng(SEED)
    bi = rng.integers(0, fr.N, size=(B_BOOT, fr.N))
    lb = ((fr.d[bi].mean(axis=1) + fr.kl_o[bi].mean(axis=1))
          / (fr.kl_m[bi].mean(axis=1) + fr.kl_o[bi].mean(axis=1)))
    return [r6(np.percentile(lb, 2.5)), r6(np.percentile(lb, 97.5))]


def _boot_ci(d):
    rng = np.random.default_rng(SEED)
    bs = d[rng.integers(0, len(d), size=(B_BOOT, len(d)))].mean(axis=1)
    return [r6(np.percentile(bs, 2.5)), r6(np.percentile(bs, 97.5))]


def main():
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": ("DEV-only decomposition of the gap between our shipped model and the "
                 "closing line, by game type. Follow-up to phase0/error_map.py."),
        "protocol": {
            "read_only": True, "models_touched": 0, "features_created": 0,
            "test_looks_spent": 0, "market_blind_models": True,
            "odds_use": ("EVALUATION / DIAGNOSTIC ONLY. The closing line is the benchmark "
                         "being decomposed against; it is never a feature and never a "
                         "training signal anywhere in this repo."),
            "loaders": ("error_map.mlb_league / error_map.nfl_league unchanged; slicing "
                        "labels captured from error_map.decompose in flight; both market "
                        "joins lifted verbatim and re-verified against data/error_map.json."),
            "instruments": {
                "gap": "our per-game log loss - market per-game log loss (nats)",
                "total_gap": "n x gap: the share of the league's aggregate deficit a slice "
                             "explains. Slices within a dimension partition the games, so "
                             "total_gap sums to the league total (team dim double-counts).",
                "kl_if_market_truth": "E[gap] if the close were the true probability. A "
                                      "slice with more disagreement has a bigger gap "
                                      "MECHANICALLY; this is that null.",
                "kl_if_we_truth": "-E[gap] if OUR probability were the truth.",
                "lambda_market_credit": "(gap + kl_if_we_truth) / (kl_if_market_truth + "
                                        "kl_if_we_truth). 1 = the close is truth and we are "
                                        "noise; 0 = we are truth; ~0.5 = the truth is "
                                        "between us. The disagreement-normalised instrument.",
                "heterogeneity Q": "sum (gap_s - pooled)^2 / se_s^2 ~ chi2(k-1). The direct "
                                   "CONCENTRATED-vs-DIFFUSE test.",
                "encompassing": "y ~ a + b_us*logit(p_ours) + b_mkt*logit(p_close).",
            },
            "multiple_comparisons": ("~60 slices per league are inspected. Nothing here is "
                                     "adopted and nothing is fitted; any slice that looks "
                                     "real needs its own pre-registered screen with a nested "
                                     "estimate, per the Round 1 method standard."),
        },
        "leagues": {},
    }

    # ------------------------------------------------------------------ MLB --
    log("MLB: running the shipped DEV harness via error_map ...")
    hold, lab = load_league("mlb", EM.mlb_league)
    idx, pm = mlb_join(hold)
    seas = hold["seas"][idx]
    assert seas.max() < MLB_TEST_LO, "market join reached a TEST-era MLB season"
    p_o, y = hold["p"][idx], hold["y"][idx]
    chk = check_join("mlb", p_o, pm, y)
    log(f"MLB joined n={len(y)} seasons {int(seas.min())}-{int(seas.max())} {chk}")
    labs = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(hold["p"]) else v)
            for k, v in lab.items()}
    labs["season"] = np.array([str(s) for s in seas])
    fr = Frame("mlb", p_o, pm, y, labs)
    dims = ["pred_decile", "month", "season_maturity", "favourite_side", "rest",
            "starter_depth", "season", "team"]
    mat, sp = labs["season_maturity"], labs["starter_depth"]
    dis = np.abs(fr.p_o - fr.p_m)
    cons = [
        ("cold start: team-games 0-9", mat == "team-games 0-9",
         "if the close's edge lives here, the missing information is roster/offseason"),
        ("first month (April)", labs["month"] == "04", "calendar version of the same test"),
        ("first third: team-games 0-29", np.isin(mat, ["team-games 0-9", "team-games 10-29"]),
         "wider cold-start window, more power"),
        ("unknown starter: SP prior starts 0-5",
         np.isin(sp, ["SP prior starts 0", "SP prior starts 1-5"]),
         "does the close price rookie/unfamiliar starters better than our FIP prior?"),
        ("top 5% disagreement", dis >= np.quantile(dis, 0.95),
         "where we and the close differ most"),
    ]
    payload["leagues"]["mlb"] = league_block(
        "mlb", fr, dims,
        f"odds_mlb.csv covers 2010+, DEV is 2002-2015 -> overlap "
        f"{int(seas.min())}-{int(seas.max())}; doubleheaders/duplicate price rows dropped.",
        cons)
    payload["leagues"]["mlb"]["join_check"] = chk
    EM.atomic_json(payload, OUT)

    # ------------------------------------------------------------------ NFL --
    log("NFL: running the shipped DEV harness via error_map (heavy prelude) ...")
    hold, lab = load_league("nfl", EM.nfl_league)
    idx, pm = nfl_join(hold)
    seas = hold["seas"][idx]
    assert seas.max() < NFL_TEST_LO, "market join reached a TEST-era NFL season"
    p_o, y = hold["p"][idx], hold["y"][idx]
    chk = check_join("nfl", p_o, pm, y)
    log(f"NFL joined n={len(y)} seasons {int(seas.min())}-{int(seas.max())} {chk}")
    labs = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(hold["p"]) else v)
            for k, v in lab.items()}
    labs["season"] = np.array([str(s) for s in seas])
    labs["qb_change"] = nfl_qb_change(hold["gid"][idx])
    fr = Frame("nfl", p_o, pm, y, labs)
    dims = ["pred_decile", "week", "favourite_side", "rest", "bye", "qb_change",
            "season", "team"]
    wk, qc = labs["week"], labs["qb_change"]
    dis = np.abs(fr.p_o - fr.p_m)
    cons = [
        ("cold start: weeks 1-2", wk == "week 1-2",
         "if the close's edge lives here, the missing information is roster/offseason"),
        ("cold start: weeks 1-4", np.isin(wk, ["week 1-2", "week 3-4"]),
         "wider cold-start window, more power"),
        ("QB change either side", np.isin(qc, ["QB change: home", "QB change: away",
                                               "QB change: both"]),
         "the biggest publicly-known in-season information shock in the NFL"),
        ("bye week involved", labs["bye"] != "bye: neither",
         "extra prep time: schedule information both sides can see"),
        ("top 5% disagreement", dis >= np.quantile(dis, 0.95),
         "where we and the close differ most"),
    ]
    payload["leagues"]["nfl"] = league_block(
        "nfl", fr, dims,
        f"nfl_games.csv moneylines start 2006; DEV scored era 2006-2015 -> overlap "
        f"{int(seas.min())}-{int(seas.max())}.",
        cons)
    payload["leagues"]["nfl"]["join_check"] = chk

    payload["leagues"]["nhl"] = {"status": EM.NHL_FLOOR_SKIP}
    EM.atomic_json(payload, OUT)
    log(f"wrote {OUT}")

    # ------------------------------------------------------------------ print
    for lg in ("mlb", "nfl"):
        d = payload["leagues"][lg]
        print(f"\n=== {lg.upper()}  n={d['n']}  ours {d['our_ll']}  close {d['market_ll']}  "
              f"gap {d['gap']:+.5f} {d['gap_ci']}  total {d['total_gap_nats']:+.1f} nats  "
              f"lambda {d['lambda_market_credit']} {d['lambda_ci']}  "
              f"gap-KLnull {d['gap_minus_kl_null']:+.5f}")
        print("  -- slices ranked by total gap contributed")
        print(f"  {'dim':<16} {'slice':<26} {'n':>6} {'ourLL':>8} {'mktLL':>8} "
              f"{'gap':>9} {'total':>8} {'share':>6} {'nshare':>6} {'lam':>6}")
        for r in d["ranked_by_total_gap"][:14]:
            print(f"  {r['dim']:<16} {r['slice']:<26} {r['n']:>6} {r['our_ll']:>8.5f} "
                  f"{r['market_ll']:>8.5f} {r['gap']:>+9.5f} {r['total_gap']:>+8.2f} "
                  f"{r['share_of_total_gap']:>6.2f} {r['n_share']:>6.2f} "
                  f"{r['lambda_market_credit'] if r['lambda_market_credit'] is not None else float('nan'):>6.2f}"
                  f"  {r.get('gap_sig','')}")
        print("  -- heterogeneity (is the gap uniform within a dimension?)")
        for dd, h in d["heterogeneity"].items():
            print(f"  {dd:<18} Q={h['Q']:8.2f} df={h['df']:<3} p={h['p']:.4f}  "
                  f"top={h['top_slice'][:24]:<24} share {h['top_slice_share']} vs n-share "
                  f"{h['top_slice_n_share']}  {h['verdict']}")
        print("  -- targeted contrasts (gap inside vs outside, paired bootstrap)")
        for c in d["contrasts"]:
            print(f"  {c['contrast']:<34} n={c['n_in']:>5} gap_in {c['gap_in']:+.5f} "
                  f"gap_out {c['gap_out']:+.5f} diff {c['diff']:+.5f} "
                  f"{str(c['diff_ci']):<24} lam {c['lambda_in']:.2f}/{c['lambda_out']:.2f}"
                  f"  {c['verdict']}")
        print("  -- DISAGREEMENT buckets  (gap vs its KL null; lambda = market's share)")
        print(f"  {'bucket':<16} {'n':>6} {'ourLL':>8} {'mktLL':>8} {'gap':>9} "
              f"{'KLnull':>9} {'gap-KL':>9} {'lam':>6} {'lamCI':>16}")
        for r in d["disagreement"]["quantiles"]:
            print(f"  {r['slice']:<16} {r['n']:>6} {r['our_ll']:>8.5f} {r['market_ll']:>8.5f} "
                  f"{r['gap']:>+9.5f} {r['kl_if_market_truth']:>9.5f} "
                  f"{r['gap'] - r['kl_if_market_truth']:>+9.5f} "
                  f"{r['lambda_market_credit']:>6.2f} {str(r.get('lambda_ci')):>16}")
        e = d["disagreement"]["encompassing_all"]
        e2 = d["disagreement"]["encompassing_top20pct_disagreement"]
        print(f"  encompassing ALL   b_ours {e['b_ours']:+.3f} (t {e['b_ours_t']:+.2f})  "
              f"b_mkt {e['b_market']:+.3f} (t {e['b_market_t']:+.2f})")
        print(f"  encompassing HIDIS b_ours {e2['b_ours']:+.3f} (t {e2['b_ours_t']:+.2f})  "
              f"b_mkt {e2['b_market']:+.3f} (t {e2['b_market_t']:+.2f})  n={e2['n']}")
        o = d["disagreement"]["opposite_favourite"]
        if "gap" in o:
            print(f"  opposite favourites n={o['n']}: we {o['our_side_win_rate']:.3f} vs "
                  f"close {o['market_side_win_rate']:.3f} hit rate, gap {o['gap']:+.5f}")


if __name__ == "__main__":
    main()
