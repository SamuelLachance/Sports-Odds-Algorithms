"""Feature extraction from blend layers and market lines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from web.bet_advisor import devig_two_way_probs, model_home_margin
from web.blend_service import (
    _layer_home_margin,
    _third_layer_key,
    layer_home_win_probability,
    total_score_to_home_win_prob,
)
from web.ensemble_ml.config import SOCCER_STACKING_FEATURES, get_stacking_features
from web.sports_meta_model import stack_binary_blend_layers
from web.soccer_meta_model import stack_soccer_blend_layers


def _sport_layer(blended: dict[str, Any]) -> dict[str, Any] | None:
    key = _third_layer_key(blended)
    if not key:
        return None
    payload = blended.get(key)
    return payload if isinstance(payload, dict) else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def market_devig_home(
    home_ml: int | float | None,
    away_ml: int | float | None,
) -> float | None:
    _away, home = devig_two_way_probs(
        int(away_ml) if away_ml is not None else None,
        int(home_ml) if home_ml is not None else None,
    )
    return home


def market_devig_threeway(
    home_ml: int | float | None,
    draw_ml: int | float | None,
    away_ml: int | float | None,
) -> tuple[float | None, float | None, float | None]:
    if home_ml is None or draw_ml is None or away_ml is None:
        return None, None, None
    from web.bet_advisor import american_implied_prob

    raw = [
        american_implied_prob(int(away_ml)),
        american_implied_prob(int(draw_ml)),
        american_implied_prob(int(home_ml)),
    ]
    total = sum(raw)
    if total <= 0:
        return None, None, None
    away_p, draw_p, home_p = (value / total * 100.0 for value in raw)
    return home_p, draw_p, away_p


def extract_binary_features(
    blended: dict[str, Any],
    league: str,
    *,
    consensus_spread: float | None = None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
) -> dict[str, float]:
    legacy = blended.get("legacy") or {}
    power = blended.get("power") or {}
    sport = _sport_layer(blended)

    legacy_home = layer_home_win_probability(legacy)
    power_home = layer_home_win_probability(power)
    sport_home = layer_home_win_probability(sport) if sport else None

    meta_stacked = stack_binary_blend_layers(
        legacy_home=legacy_home,
        power_home=power_home,
        sport_home=sport_home,
        league=league,
    )
    if meta_stacked is None and power_home is not None:
        if sport_home is not None:
            meta_stacked = 0.35 * power_home + 0.65 * sport_home
        else:
            meta_stacked = power_home
    if league.lower() == "nhl" and meta_stacked is not None:
        from web.sports_meta_model import apply_binary_calibration

        meta_stacked = apply_binary_calibration(float(meta_stacked), league)
    if league.lower() == "mlb" and meta_stacked is not None:
        from web.sports_meta_model import apply_binary_calibration

        meta_stacked = apply_binary_calibration(float(meta_stacked), league)

    legacy_margin = _layer_home_margin(legacy, league) if legacy else None
    power_margin = _layer_home_margin(power, league) if power else None
    sport_margin = _layer_home_margin(sport, league) if sport else None

    market_home = market_devig_home(home_moneyline, away_moneyline)
    sport_minus_market = None
    if sport_home is not None and market_home is not None:
        sport_minus_market = sport_home - market_home

    expected_home_goals = _safe_float(sport.get("expected_home_goals")) if sport else None
    expected_away_goals = _safe_float(sport.get("expected_away_goals")) if sport else None
    overtime_probability = _safe_float(sport.get("overtime_probability")) if sport else None
    sport_xg_diff = None
    if expected_home_goals is not None and expected_away_goals is not None:
        sport_xg_diff = expected_home_goals - expected_away_goals

    return {
        "legacy_home_prob": legacy_home,
        "power_home_prob": power_home,
        "sport_home_prob": sport_home,
        "meta_stacked_home_prob": meta_stacked,
        "legacy_margin": legacy_margin,
        "power_margin": power_margin,
        "sport_margin": sport_margin,
        "sport_minus_market": sport_minus_market,
        "expected_home_goals": expected_home_goals,
        "expected_away_goals": expected_away_goals,
        "sport_xg_diff": sport_xg_diff,
        "overtime_probability": overtime_probability,
        "market_devig_home_prob": market_home,
        "market_spread": _safe_float(consensus_spread),
        "market_home_ml": _safe_float(home_moneyline),
        "market_away_ml": _safe_float(away_moneyline),
    }


def extract_soccer_features(
    blended: dict[str, Any],
    league: str,
    *,
    home_moneyline: int | None = None,
    draw_moneyline: int | None = None,
    away_moneyline: int | None = None,
) -> dict[str, float]:
    legacy = blended.get("legacy_threeway") or blended.get("legacy") or {}
    power = blended.get("power_threeway") or blended.get("power") or {}
    sport = blended.get("soccer_pred") or {}

    legacy_h = _safe_float(legacy.get("home_win_probability"))
    legacy_d = _safe_float(legacy.get("draw_probability"))
    legacy_a = _safe_float(legacy.get("away_win_probability"))
    power_h = _safe_float(power.get("home_win_probability"))
    power_d = _safe_float(power.get("draw_probability"))
    power_a = _safe_float(power.get("away_win_probability"))
    sport_h = _safe_float(sport.get("home_win_probability"))
    sport_d = _safe_float(sport.get("draw_probability"))
    sport_a = _safe_float(sport.get("away_win_probability"))

    meta = stack_soccer_blend_layers(
        legacy=(legacy_h, legacy_d, legacy_a) if legacy_h is not None else None,
        power=(power_h, power_d, power_a) if power_h is not None else None,
        soccer_pred=(sport_h, sport_d, sport_a) if sport_h is not None else None,
        league=league,
    )
    meta_h, meta_d, meta_a = (None, None, None)
    if meta:
        meta_h, meta_d, meta_a = meta

    mkt_h, mkt_d, mkt_a = market_devig_threeway(home_moneyline, draw_moneyline, away_moneyline)

    return {
        "legacy_home_prob": legacy_h,
        "legacy_draw_prob": legacy_d,
        "legacy_away_prob": legacy_a,
        "power_home_prob": power_h,
        "power_draw_prob": power_d,
        "power_away_prob": power_a,
        "sport_home_prob": sport_h,
        "sport_draw_prob": sport_d,
        "sport_away_prob": sport_a,
        "meta_home_prob": meta_h,
        "meta_draw_prob": meta_d,
        "meta_away_prob": meta_a,
        "market_devig_home_prob": mkt_h,
        "market_devig_draw_prob": mkt_d,
        "market_devig_away_prob": mkt_a,
    }


def features_to_frame(
    features: dict[str, float | None],
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    row = {column: features.get(column) for column in columns}
    return pd.DataFrame([row])[list(columns)].astype(float)


def margin_from_home_prob(home_prob: float, league: str) -> float:
    total, _win = _home_prob_to_total(home_prob)
    return model_home_margin(total, league)


def _home_prob_to_total(home_prob: float) -> tuple[float, float]:
    from web.blend_service import home_win_prob_to_total_score

    return home_win_prob_to_total_score(home_prob)


def legacy_home_from_total(total_score: float) -> float:
    return total_score_to_home_win_prob(total_score)
