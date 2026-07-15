"""Live scoring for the shipped Hubáček β (bettor) model.

Artifacts under data/models/{league}_v2/: ``model_clf_bettor.cbm`` +
``bettor_meta.json`` (feature_cols/cat_cols, decorrelation_c, phi, isotonic
calibrator points fit on walk-forward OOS). The β probability is DISPLAY /
EV telemetry — official picks stay on their validated gates until the β
meta-selection clears the enable bar.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=16)
def load_bettor_bundle(league: str) -> dict[str, Any] | None:
    v2 = PROJECT_ROOT / "data" / "models" / f"{league}_v2"
    meta_path = v2 / "bettor_meta.json"
    model_path = v2 / "model_clf_bettor.cbm"
    if not meta_path.is_file() or not model_path.is_file():
        return None
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return {
        "meta": meta,
        "model": model,
        "feature_cols": list(meta.get("feature_cols") or []),
        "cat_cols": list(meta.get("cat_cols") or []),
        "phi": float(meta.get("phi") or 0.0),
        "decorrelation_c": float(meta.get("decorrelation_c") or 0.0),
        "calibrator": meta.get("calibrator"),
    }


def score_bettor_binary(league: str, feature_row: dict[str, Any]) -> dict[str, Any] | None:
    """β home-win probability for one matchup (None when no artifacts)."""
    bundle = load_bettor_bundle(league)
    if bundle is None:
        return None
    from catboost import Pool

    from web.hubacek_v2.objective import sigmoid

    import pandas as pd

    cols = bundle["feature_cols"]
    cat_cols = bundle["cat_cols"]
    row = {c: feature_row.get(c, 0.0) for c in cols}
    for c in cat_cols:
        row[c] = str(feature_row.get(c, feature_row.get(c.replace("_id", ""), "UNK")))
    frame = pd.DataFrame([row])[cols]
    raw = float(bundle["model"].predict(Pool(frame, cat_features=cat_cols or None))[0])
    prob = float(sigmoid(np.array([raw]))[0])
    cal = bundle.get("calibrator")
    if cal and len(cal.get("x") or []) >= 2:
        prob = float(np.interp(prob, cal["x"], cal["y"]))
    prob = float(np.clip(prob, 0.005, 0.995))
    return {
        "home_win_prob": prob,
        "phi": bundle["phi"],
        "decorrelation_c": bundle["decorrelation_c"],
        "model_variant": "hubacek_beta",
    }


def clear_bettor_cache() -> None:
    load_bettor_bundle.cache_clear()
