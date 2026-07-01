"""Live inference for per-league ensemble mega-models."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from web.ensemble_ml.config import (
    SOCCER_STACKING_FEATURES,
    STACKING_FEATURES,
    ensemble_model_available,
    metadata_path,
)
from web.ensemble_ml.features import extract_binary_features, extract_soccer_features
from web.ensemble_ml.model import (
    BinaryEnsembleModel,
    SoccerEnsembleModel,
    load_binary_ensemble,
    load_soccer_ensemble,
    predict_binary,
    predict_soccer,
)
from web.league_profiles import is_soccer_league


@lru_cache(maxsize=32)
def _cached_binary(league: str) -> BinaryEnsembleModel | None:
    return load_binary_ensemble(league)


@lru_cache(maxsize=16)
def _cached_soccer(league: str) -> SoccerEnsembleModel | None:
    return load_soccer_ensemble(league)


def clear_ensemble_caches() -> None:
    _cached_binary.cache_clear()
    _cached_soccer.cache_clear()


def predict_ensemble_for_blend(
    blended: dict[str, Any],
    league: str,
    *,
    consensus_spread: float | None = None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
    draw_moneyline: int | None = None,
) -> dict[str, Any] | None:
    """Score a live blend payload; None when no trained artifact exists."""
    league = league.lower()
    if not ensemble_model_available(league):
        return None

    if is_soccer_league(league):
        model = _cached_soccer(league)
        if not model:
            return None
        features = extract_soccer_features(
            blended,
            league,
            home_moneyline=home_moneyline,
            draw_moneyline=draw_moneyline,
            away_moneyline=away_moneyline,
        )
        preds = predict_soccer(model, features)
        return {
            "algorithm": "EnsembleML (XGBoost 3-way)",
            "source": "ensemble_ml",
            **preds,
        }

    model = _cached_binary(league)
    if not model:
        return None
    features = extract_binary_features(
        blended,
        league,
        consensus_spread=consensus_spread,
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
    )
    preds = predict_binary(model, features)
    return {
        "algorithm": "EnsembleML (XGBoost)",
        "source": "ensemble_ml",
        **preds,
    }


def ensemble_metadata(league: str) -> dict[str, Any]:
    path = metadata_path(league.lower())
    if not path.is_file():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
