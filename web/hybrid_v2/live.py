"""Live scoring helpers for shipped hybrid CatBoost artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def fill_curve_proxies(features: dict[str, Any]) -> dict[str, Any]:
    """Identity CurveFM proxies from tabular strength features (live path)."""
    out = dict(features)
    elo = float(out.get("elo_diff") or 0.0)
    net = float(
        out.get("net_rtg_slow_diff")
        or out.get("net_rtg_diff")
        or out.get("adj_margin_ewma_diff")
        or out.get("xg_diff")
        or 0.0
    )
    out.setdefault("curve_elo_diff", elo)
    out.setdefault("curve_net_diff", net)
    out.setdefault("curve_elo_raw_diff", elo)
    out.setdefault("curve_unc_sum", 0.0)
    out.setdefault("curve_unc_diff", 0.0)
    out.setdefault("curve_backend", 0.0)
    # Market feature defaults when odds missing
    out.setdefault("has_market", float(out.get("has_market") or 0.0))
    out.setdefault("mkt_home_prob", float(out.get("mkt_home_prob") or 0.5))
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
    return score_hybrid_binary(league, row, market_home_prob=market_home_prob)


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
    return {
        "meta": meta,
        "clf": clf,
        "margin": margin,
        "feature_cols": list(meta.get("feature_cols") or []),
        "cat_cols": list(meta.get("cat_cols") or []),
        "margin_feature_cols": list(meta.get("margin_feature_cols") or []),
        "market_blend_w": float(meta.get("market_blend_w") or 0.0),
        "multiclass": bool(meta.get("multiclass")),
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
    pool = Pool(frame, cat_features=cat_cols or None)
    raw = float(bundle["clf"].predict_proba(pool)[0][1])
    w = bundle["market_blend_w"]
    if w > 0 and market_home_prob is not None and np.isfinite(market_home_prob):
        raw = (1.0 - w) * raw + w * float(market_home_prob)
    raw = float(np.clip(raw, 0.02, 0.98))
    out: dict[str, Any] = {
        "home_win_prob": raw,
        "model_variant": "hybrid",
        "market_blend_w": w,
    }
    if bundle["margin"] is not None:
        m_cols = bundle["margin_feature_cols"] or [c for c in cols if c not in cat_cols]
        m_row = {c: feature_row.get(c, 0.0) for c in m_cols}
        m_frame = pd.DataFrame([m_row])[m_cols]
        out["predicted_margin"] = float(bundle["margin"].predict(Pool(m_frame))[0])
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
