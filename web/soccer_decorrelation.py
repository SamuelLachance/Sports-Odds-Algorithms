"""Hubáček-style 3-way market decorrelation for Soccer Path A."""

from __future__ import annotations

from web.bet_advisor import american_implied_prob
from web.soccer_meta_model import normalize_threeway

SOCCER_MARKET_BLEND_MAX = 0.35


def devig_threeway_from_odds(
    home_odds: int | None,
    draw_odds: int | None,
    away_odds: int | None,
) -> tuple[float, float, float] | None:
    """Remove book vig from 1X2 moneylines (0–100 scale)."""
    if home_odds is None or draw_odds is None or away_odds is None:
        return None
    home_raw = american_implied_prob(home_odds)
    draw_raw = american_implied_prob(draw_odds)
    away_raw = american_implied_prob(away_odds)
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
    blend = SOCCER_MARKET_BLEND_MAX if weight is None else min(max(weight, 0.0), 1.0)
    adjusted: list[float] = []
    for model_p, market_p in zip(model_probs, market_probs):
        mp = min(max(model_p / 100.0, 0.01), 0.99)
        mk = min(max(market_p / 100.0, 0.01), 0.99)
        adj = mp - blend * (mp - mk)
        adjusted.append(min(max(adj, 0.01), 0.99))
    return normalize_threeway(
        adjusted[0] * 100.0,
        adjusted[1] * 100.0,
        adjusted[2] * 100.0,
    )


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
