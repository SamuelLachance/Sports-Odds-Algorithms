"""Live scoring helpers for shipped hybrid CatBoost artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def fill_curve_proxies(features: dict[str, Any]) -> dict[str, Any]:
    """Identity CurveFM / MC proxies from tabular strength features (live path)."""
    out = dict(features)
    elo = float(out.get("elo_diff") or 0.0)
    net = float(
        out.get("net_rtg_slow_diff")
        or out.get("net_rtg_diff")
        or out.get("adj_margin_ewma_diff")
        or out.get("xg_diff")
        or out.get("epa_off_diff")
        or 0.0
    )
    out.setdefault("curve_elo_diff", elo)
    out.setdefault("curve_net_diff", net)
    out.setdefault("curve_elo_raw_diff", elo)
    out.setdefault("curve_unc_sum", 0.0)
    out.setdefault("curve_unc_diff", 0.0)
    out.setdefault("curve_backend", 0.0)
    # Match-layer MC proxies (NFL revolution / hybrid)
    points_per_elo = float(out.get("points_per_elo") or 25.0)
    mc_margin = float(out.get("mc_margin") if out.get("mc_margin") is not None else elo / max(points_per_elo, 1e-6))
    out.setdefault("mc_margin", mc_margin)
    out.setdefault("mc_sigma", float(out.get("mc_sigma") or 13.0))
    # Normal CDF approx for home win prior
    if "mc_home_prob" not in out or out.get("mc_home_prob") is None:
        import math

        z = mc_margin / max(float(out["mc_sigma"]), 1e-6)
        out["mc_home_prob"] = float(min(max(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 0.02), 0.98))
    out.setdefault("mc_x_unc", float(out["mc_margin"]) * float(out.get("curve_unc_sum") or 0.0))
    # Categorical defaults for CatBoost hybrid
    out.setdefault("roof_cat", "dome" if float(out.get("roof_dome") or 0.0) > 0.5 else "outdoor")
    out.setdefault("weekday_cat", "thu" if float(out.get("weekday_thu") or 0.0) > 0.5 else "other")
    week = float(out.get("week") or 1.0)
    if "week_bucket" not in out:
        if week <= 4:
            out["week_bucket"] = "early"
        elif week <= 9:
            out["week_bucket"] = "mid"
        elif week <= 13:
            out["week_bucket"] = "late"
        elif week <= 18:
            out["week_bucket"] = "dec"
        else:
            out["week_bucket"] = "post"
    # Market feature defaults when odds missing
    out.setdefault("has_market", float(out.get("has_market") or 0.0))
    out.setdefault("mkt_home_prob", float(out.get("mkt_home_prob") or 0.5))
    out.setdefault("has_spread", float(out.get("has_spread") or 0.0))
    out.setdefault("mkt_home_spread", float(out.get("mkt_home_spread") or out.get("home_spread") or 0.0))
    out.setdefault("ml_steam_pp", float(out.get("ml_steam_pp") or 0.0))
    out.setdefault("has_steam", float(out.get("has_steam") or 0.0))
    out.setdefault("mkt_open_home", float(out.get("mkt_open_home") or 1 / 3))
    out.setdefault("mkt_open_draw", float(out.get("mkt_open_draw") or 1 / 3))
    out.setdefault("mkt_open_away", float(out.get("mkt_open_away") or 1 / 3))
    return out


def try_hybrid_binary(
    league: str,
    features: dict[str, Any],
    *,
    home_id: str,
    away_id: str,
    market_home_prob: float | None = None,
) -> dict[str, Any] | None:
    row = fill_curve_proxies(features)
    row["home_id"] = str(home_id)
    row["away_id"] = str(away_id)
    result = score_hybrid_binary(league, row, market_home_prob=market_home_prob)
    # Hubáček β (decorrelated bettor) telemetry rides along when shipped.
    try:
        from web.hubacek_v2.live import score_bettor_binary

        beta = score_bettor_binary(league, row)
        if beta is not None:
            gap = (
                round(100.0 * (beta["home_win_prob"] - float(market_home_prob)), 2)
                if market_home_prob is not None and np.isfinite(market_home_prob)
                else None
            )
            payload = {
                "home_win_prob_pct": round(100.0 * beta["home_win_prob"], 2),
                "gap_vs_market_pp": gap,
                "phi": beta.get("phi"),
                "decorrelation_c": beta.get("decorrelation_c"),
            }
            # Only piggyback on a successful π result: league modules read
            # ["home_win_prob"] first and their guard except would otherwise
            # discard a β-only payload anyway (π ships alongside β).
            if result is not None:
                result["hubacek_beta"] = payload
    except Exception:  # noqa: BLE001 — β telemetry must never break scoring
        pass
    return result


@lru_cache(maxsize=16)
def load_hybrid_bundle(league: str) -> dict[str, Any] | None:
    v2 = PROJECT_ROOT / "data" / "models" / f"{league}_v2"
    meta_path = v2 / "hybrid_meta.json"
    if not meta_path.is_file():
        return None
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    clf_name = meta.get("clf_path", "model_clf_hybrid.cbm")
    clf_path = v2 / clf_name
    if not clf_path.is_file():
        return None
    clf = CatBoostClassifier()
    clf.load_model(str(clf_path))
    margin = None
    if meta.get("margin_path"):
        m_path = v2 / meta["margin_path"]
        if m_path.is_file():
            margin = CatBoostRegressor()
            margin.load_model(str(m_path))
    calibrator = None
    cal_path = v2 / "calibrator_hybrid.json"
    if cal_path.is_file():
        try:
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            if (
                cal.get("kind") == "isotonic_oos"
                and len(cal.get("x") or []) >= 2
                and len(cal.get("x")) == len(cal.get("y") or [])
            ):
                calibrator = {
                    "x": [float(v) for v in cal["x"]],
                    "y": [float(v) for v in cal["y"]],
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            calibrator = None
    return {
        "meta": meta,
        "clf": clf,
        "margin": margin,
        "feature_cols": list(meta.get("feature_cols") or []),
        "cat_cols": list(meta.get("cat_cols") or []),
        "margin_feature_cols": list(meta.get("margin_feature_cols") or []),
        "market_blend_w": float(meta.get("market_blend_w") or 0.0),
        "target_mode": str(meta.get("target_mode") or "prob"),
        "multiclass": bool(meta.get("multiclass")),
        "calibrator": calibrator,
    }


def hybrid_available(league: str) -> bool:
    return load_hybrid_bundle(league) is not None


def score_hybrid_binary(
    league: str,
    feature_row: dict[str, Any],
    *,
    market_home_prob: float | None = None,
) -> dict[str, Any] | None:
    """Score a single matchup; feature_row must include hybrid feature_cols + cat ids."""
    bundle = load_hybrid_bundle(league)
    if bundle is None or bundle["multiclass"]:
        return None
    from catboost import Pool

    cols = bundle["feature_cols"]
    cat_cols = bundle["cat_cols"]
    row = {c: feature_row.get(c, 0.0) for c in cols}
    for c in cat_cols:
        row[c] = str(feature_row.get(c, feature_row.get(c.replace("_id", ""), "UNK")))
    import pandas as pd

    frame = pd.DataFrame([row])[cols]
    baseline = None
    if bundle.get("target_mode") == "offset":
        # Market-as-init_score models predict a residual on logit(close);
        # no market at score time -> baseline 0 (= market-free estimate).
        if market_home_prob is not None and np.isfinite(market_home_prob):
            p = float(np.clip(market_home_prob, 1e-4, 1 - 1e-4))
            baseline = [float(np.log(p / (1 - p)))]
        else:
            baseline = [0.0]
    pool = Pool(frame, cat_features=cat_cols or None, baseline=baseline)
    raw = float(bundle["clf"].predict_proba(pool)[0][1])
    w = bundle["market_blend_w"]
    has_mkt = market_home_prob is not None and np.isfinite(market_home_prob)
    cal = bundle.get("calibrator")
    if bundle.get("target_mode") == "logit_stack":
        # Same order as training: calibrate the model prob, then logit-stack
        # with the market. No post-stack recalibration.
        if cal is not None:
            raw = float(np.interp(raw, cal["x"], cal["y"]))
        if w > 0 and has_mkt:
            pm = float(np.clip(raw, 1e-9, 1 - 1e-9))
            pk = float(np.clip(market_home_prob, 1e-9, 1 - 1e-9))
            z = (1.0 - w) * np.log(pm / (1 - pm)) + w * np.log(pk / (1 - pk))
            raw = float(1.0 / (1.0 + np.exp(-z)))
    else:
        if w > 0 and has_mkt:
            raw = (1.0 - w) * raw + w * float(market_home_prob)
        # Post-blend isotonic (fit on walk-forward OOS at ship time) keeps live
        # probs on the same scale the OOS metrics were computed on.
        if cal is not None:
            raw = float(np.interp(raw, cal["x"], cal["y"]))
    raw = float(np.clip(raw, 0.005, 0.995))
    out: dict[str, Any] = {
        "home_win_prob": raw,
        "model_variant": "hybrid",
        "market_blend_w": w,
    }
    if bundle["margin"] is not None:
        m_cols = bundle["margin_feature_cols"] or cols
        m_cats = [c for c in cat_cols if c in m_cols]
        m_row = {c: feature_row.get(c, 0.0) for c in m_cols}
        for c in m_cats:
            m_row[c] = str(feature_row.get(c, feature_row.get(c.replace("_id", ""), "UNK")))
        m_frame = pd.DataFrame([m_row])[m_cols]
        out["predicted_margin"] = float(
            bundle["margin"].predict(Pool(m_frame, cat_features=m_cats or None))[0]
        )
    return out


def score_hybrid_soccer(
    league: str,
    feature_row: dict[str, Any],
    *,
    market_probs: tuple[float, float, float] | None = None,
) -> dict[str, Any] | None:
    bundle = load_hybrid_bundle(league)
    if bundle is None or not bundle["multiclass"]:
        return None
    from catboost import Pool
    import pandas as pd

    cols = bundle["feature_cols"]
    cat_cols = bundle["cat_cols"]
    row = {c: feature_row.get(c, 0.0) for c in cols}
    for c in cat_cols:
        row[c] = str(feature_row.get(c, "UNK"))
    frame = pd.DataFrame([row])[cols]
    pool = Pool(frame, cat_features=cat_cols or None)
    proba = np.asarray(bundle["clf"].predict_proba(pool)[0], dtype=float)
    w = bundle["market_blend_w"]
    if w > 0 and market_probs is not None:
        mkt = np.asarray(market_probs, dtype=float)
        if mkt.shape == (3,) and np.all(np.isfinite(mkt)):
            proba = (1.0 - w) * proba + w * mkt
            proba = proba / max(proba.sum(), 1e-9)
    proba = np.clip(proba, 1e-4, 1.0)
    proba = proba / proba.sum()
    return {
        "home": float(proba[0]),
        "draw": float(proba[1]),
        "away": float(proba[2]),
        "model_variant": "hybrid",
        "market_blend_w": w,
    }


def clear_hybrid_cache() -> None:
    load_hybrid_bundle.cache_clear()
