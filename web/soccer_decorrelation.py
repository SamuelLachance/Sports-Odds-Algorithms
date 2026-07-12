"""Hubáček-style 3-way market decorrelation for Soccer Path A."""

from __future__ import annotations

from web.bet_advisor import american_implied_prob, normalize_american_odds
from web.market_decorrelation import (
    DEFAULT_THREEWAY_DECORRELATION_WEIGHT,
    decorrelate_three_way,
)
from web.soccer_meta_model import normalize_threeway

SOCCER_MARKET_BLEND_MAX = DEFAULT_THREEWAY_DECORRELATION_WEIGHT


def devig_threeway_from_odds(
    home_odds: int | None,
    draw_odds: int | None,
    away_odds: int | None,
) -> tuple[float, float, float] | None:
    """Remove book vig from 1X2 moneylines (0–100 scale).

    Invalid American odds fail closed (``None``), matching ``devig_two_way_probs``.
    """
    home_n = normalize_american_odds(home_odds)
    draw_n = normalize_american_odds(draw_odds)
    away_n = normalize_american_odds(away_odds)
    if home_n is None or draw_n is None or away_n is None:
        return None
    home_raw = american_implied_prob(home_n)
    draw_raw = american_implied_prob(draw_n)
    away_raw = american_implied_prob(away_n)
    total = home_raw + draw_raw + away_raw
    if total <= 0:
        return None
    return normalize_threeway(
        home_raw / total * 100.0,
        draw_raw / total * 100.0,
        away_raw / total * 100.0,
    )


def decorrelate_threeway_from_market(
    model_probs: tuple[float, float, float],
    market_probs: tuple[float, float, float],
    *,
    weight: float | None = None,
) -> tuple[float, float, float]:
    """Push model 3-way probs away from devigged closing line."""
    w = DEFAULT_THREEWAY_DECORRELATION_WEIGHT if weight is None else weight
    return decorrelate_three_way(model_probs, market_probs, weight=w)


def blend_with_market_cap(
    model_probs: tuple[float, float, float],
    market_probs: tuple[float, float, float] | None,
    *,
    max_blend: float = SOCCER_MARKET_BLEND_MAX,
) -> tuple[float, float, float]:
    """Decorrelate when market available; otherwise return model probs."""
    if market_probs is None:
        return model_probs
    return decorrelate_threeway_from_market(
        model_probs,
        market_probs,
        weight=max_blend,
    )
