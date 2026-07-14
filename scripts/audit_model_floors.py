"""Audit every league's best-OOS hybrid model against its estimated irreducible error floor.

For each league with a shipped hybrid model this script loads the walk-forward OOS
prediction file that matches the best config in ``focused_final.json`` and reports:

- model log-loss / Brier / ECE (equal-count bins) + reliability table
- de-vigged closing market log-loss on the odds subset
- two irreducible-floor estimates on the odds subset:
    * ``floor_entropy``      : E[H(p_mkt)]  — floor if the de-vigged close were the true prob
    * ``floor_entropy_cal``  : E[H(iso(p_mkt))] — same after isotonic recalibration of the
      close against outcomes (debiases favourite/longshot skew; slightly optimistic since
      the isotonic fit is in-sample)
- paired bootstrap CI for (model_ll - market_ll) so "distance to market" has error bars

Distance-to-floor = model_ll - floor_entropy_cal is the headroom metric used to rank
which league to attack next.  Results are printed and written to
``data/models/floor_audit.json``.

Usage:
    python scripts/audit_model_floors.py [--leagues nba,mlb,...] [--bins 10] [--boot 2000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "data", "models")

LEAGUES = ["nba", "wnba", "nfl", "cfb", "cbb", "nhl", "mlb", "soccer"]

EPS = 1e-12


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10):
    """Equal-count ECE plus the reliability table used to compute it."""
    order = np.argsort(p)
    y_s, p_s = y[order], p[order]
    splits = np.array_split(np.arange(len(p_s)), bins)
    total = len(p_s)
    ece = 0.0
    table = []
    for idx in splits:
        if len(idx) == 0:
            continue
        conf = float(np.mean(p_s[idx]))
        acc = float(np.mean(y_s[idx]))
        ece += (len(idx) / total) * abs(conf - acc)
        table.append({"n": int(len(idx)), "mean_pred": round(conf, 4), "obs_rate": round(acc, 4)})
    return float(ece), table


def _isotonic_calibrate(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """In-sample isotonic regression of outcome on predicted prob (PAV)."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return p
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
    return iso.fit_transform(p, y)


def _paired_bootstrap_delta(y, p_model, p_ref, n_boot=2000, seed=42):
    """Bootstrap CI for mean per-game logloss(model) - logloss(ref)."""
    p_model = np.clip(p_model, EPS, 1 - EPS)
    p_ref = np.clip(p_ref, EPS, 1 - EPS)
    ll_m = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
    ll_r = -(y * np.log(p_ref) + (1 - y) * np.log(1 - p_ref))
    d = ll_m - ll_r
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "mean": float(d.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "p_worse_than_ref": float(np.mean(boots > 0)),
    }


def _best_oos_csv(league: str):
    """Locate the OOS csv matching the best config recorded in focused_final.json."""
    hybrid_dir = os.path.join(MODELS_DIR, f"{league}_hybrid")
    ff = os.path.join(hybrid_dir, "focused_final.json")
    best_name = None
    if os.path.exists(ff):
        with open(ff, "r", encoding="utf-8") as f:
            d = json.load(f)
        best = d.get("best", d)
        best_name = (best.get("config") or {}).get("name")
    candidates = sorted(glob.glob(os.path.join(hybrid_dir, "oos_*.csv")))
    if best_name:
        for c in candidates:
            if best_name in os.path.basename(c):
                return c, best_name
    return (candidates[0], best_name) if candidates else (None, best_name)


def _load_csv(path: str) -> dict:
    import csv as _csv

    with open(path, "r", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # duplicate column names exist (home/away twice); keep first occurrence
    cols: dict[str, list] = {}
    seen = {}
    for j, name in enumerate(header):
        if name in seen:
            continue
        seen[name] = j
        cols[name] = [r[j] if j < len(r) else "" for r in rows]
    return cols


def _to_float(vals):
    out = np.full(len(vals), np.nan)
    for i, v in enumerate(vals):
        if v not in ("", None):
            try:
                out[i] = float(v)
            except ValueError:
                pass
    return out


def _american_to_prob(ml: np.ndarray) -> np.ndarray:
    """Implied (vigged) win prob from American odds."""
    p = np.full_like(ml, np.nan)
    neg = ml < 0
    p[neg] = -ml[neg] / (-ml[neg] + 100.0)
    pos = ml > 0
    p[pos] = 100.0 / (ml[pos] + 100.0)
    return p


def _devig_close(cols: dict, n: int) -> np.ndarray:
    """De-vigged home prob from close moneyline columns when market_home_prob is absent."""
    home_key = "home_close_ml" if "home_close_ml" in cols else "home_ml"
    away_key = "away_close_ml" if "away_close_ml" in cols else "away_ml"
    if home_key not in cols or away_key not in cols:
        return np.full(n, np.nan)
    ph = _american_to_prob(_to_float(cols[home_key]))
    pa = _american_to_prob(_to_float(cols[away_key]))
    tot = ph + pa
    out = np.full(n, np.nan)
    ok = ~np.isnan(tot) & (tot > 0.9)
    out[ok] = ph[ok] / tot[ok]
    return out


def audit_league(league: str, bins: int, n_boot: int):
    path, cfg_name = _best_oos_csv(league)
    if path is None:
        return {"league": league, "error": "no OOS csv found"}

    cols = _load_csv(path)
    y = _to_float(cols["home_win"])
    p_model = _to_float(cols["model_prob"])
    p_mkt = _to_float(cols.get("market_home_prob", [""] * len(y)))
    if np.isnan(p_mkt).all():
        p_mkt = _devig_close(cols, len(y))
        res_market_source = "devig_close_ml"
    else:
        res_market_source = "market_home_prob"

    ok = ~np.isnan(y) & ~np.isnan(p_model)
    y, p_model, p_mkt = y[ok], p_model[ok], p_mkt[ok]

    res = {
        "league": league,
        "oos_file": os.path.basename(path),
        "config": cfg_name,
        "n": int(len(y)),
        "home_win_rate": round(float(np.mean(y)), 4),
        "model_logloss": round(_logloss(y, p_model), 5),
        "model_brier": round(_brier(y, p_model), 5),
    }
    ece, table = _ece(y, p_model, bins)
    res["model_ece"] = round(ece, 5)
    res["reliability"] = table

    # multiclass leagues: also report 3-way model logloss for reference
    if "model_prob_d" in cols:
        ph = _to_float(cols["model_prob_h"])[ok]
        pd_ = _to_float(cols["model_prob_d"])[ok]
        pa = _to_float(cols["model_prob_a"])[ok]
        lab = _to_float(cols["label"])[ok]
        P = np.clip(np.stack([ph, pd_, pa], axis=1), EPS, 1)
        P = P / P.sum(axis=1, keepdims=True)
        idx = lab.astype(int)
        valid = (idx >= 0) & (idx <= 2)
        res["model_logloss_3way"] = round(float(-np.mean(np.log(P[np.arange(len(idx))[valid], idx[valid]]))), 5)

    has_mkt = ~np.isnan(p_mkt) & (p_mkt > 0) & (p_mkt < 1)
    res["market_source"] = res_market_source
    res["n_with_odds"] = int(has_mkt.sum())
    if has_mkt.sum() >= 200:
        ym, pm, pk = y[has_mkt], p_model[has_mkt], p_mkt[has_mkt]
        res["model_logloss_odds_subset"] = round(_logloss(ym, pm), 5)
        res["market_logloss"] = round(_logloss(ym, pk), 5)
        res["market_brier"] = round(_brier(ym, pk), 5)
        mkt_ece, _ = _ece(ym, pk, bins)
        res["market_ece"] = round(mkt_ece, 5)
        res["floor_entropy"] = round(float(np.mean(_binary_entropy(pk))), 5)
        pk_cal = _isotonic_calibrate(pk, ym)
        res["floor_entropy_cal"] = round(float(np.mean(_binary_entropy(pk_cal))), 5)
        res["gap_model_vs_market"] = round(res["model_logloss_odds_subset"] - res["market_logloss"], 5)
        res["gap_model_vs_floor"] = round(res["model_logloss_odds_subset"] - res["floor_entropy_cal"], 5)
        res["gap_market_vs_floor"] = round(res["market_logloss"] - res["floor_entropy_cal"], 5)
        res["bootstrap_model_minus_market"] = _paired_bootstrap_delta(ym, pm, pk, n_boot=n_boot)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leagues", default=",".join(LEAGUES))
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "floor_audit.json"))
    args = ap.parse_args()

    leagues = [s.strip() for s in args.leagues.split(",") if s.strip()]
    results = []
    for lg in leagues:
        try:
            results.append(audit_league(lg, args.bins, args.boot))
        except Exception as exc:  # keep going: one bad league shouldn't kill the audit
            results.append({"league": lg, "error": f"{type(exc).__name__}: {exc}"})

    hdr = f"{'league':8} {'n':>7} {'model_ll':>9} {'mkt_ll':>8} {'floor':>8} {'to_mkt':>8} {'to_floor':>9} {'ece':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['league']:8} ERROR: {r['error']}")
            continue
        print(
            f"{r['league']:8} {r['n']:>7} {r['model_logloss']:>9} "
            f"{r.get('market_logloss', float('nan')):>8} {r.get('floor_entropy_cal', float('nan')):>8} "
            f"{r.get('gap_model_vs_market', float('nan')):>8} {r.get('gap_model_vs_floor', float('nan')):>9} "
            f"{r['model_ece']:>7}"
        )

    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
