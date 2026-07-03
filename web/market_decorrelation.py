"""Shared Hubáček-style market decorrelation for binary and 3-way predictions."""

from __future__ import annotations

DEFAULT_BINARY_DECORRELATION_WEIGHT = 0.12
DEFAULT_THREEWAY_DECORRELATION_WEIGHT = 0.35


def _as_fraction(prob: float) -> tuple[float, bool]:
    """Return probability in 0–1 and whether input was on 0–100 scale."""
    if prob > 1.0:
        return min(max(prob / 100.0, 0.01), 0.99), True
    return min(max(prob, 0.01), 0.99), False


def decorrelate_binary(
    model_p: float,
    market_p: float,
    *,
    weight: float = DEFAULT_BINARY_DECORRELATION_WEIGHT,
) -> float:
    """Amplify model–market disagreement (push away from market)."""
    mp, mp_pct = _as_fraction(model_p)
    mk, _ = _as_fraction(market_p)
    adjusted = mp + weight * (mp - mk)
    adjusted = min(max(adjusted, 0.01), 0.99)
    return adjusted * 100.0 if mp_pct else adjusted


def decorrelate_three_way(
    model_probs: tuple[float, float, float],
    market_probs: tuple[float, float, float],
    *,
    weight: float = DEFAULT_THREEWAY_DECORRELATION_WEIGHT,
) -> tuple[float, float, float]:
    """Amplify 3-way disagreement; renormalize to simplex."""
    blend = min(max(weight, 0.0), 1.0)
    adjusted: list[float] = []
    for model_p, market_p in zip(model_probs, market_probs):
        mp, _ = _as_fraction(model_p)
        mk, _ = _as_fraction(market_p)
        adj = mp + blend * (mp - mk)
        adjusted.append(min(max(adj, 0.01), 0.99))
    total = sum(adjusted)
    if total <= 0:
        return model_probs
    return tuple(p / total * 100.0 for p in adjusted)


def decorrelate_from_spread(
    model_p: float,
    spread: float,
    *,
    weight: float = DEFAULT_BINARY_DECORRELATION_WEIGHT,
    sigma: float = 12.0,
) -> float:
    """Convert spread to market prob, then decorrelate binary prediction."""
    from web.cbb_calibrate import spread_to_home_prob

    market_prob = spread_to_home_prob(float(spread), sigma=sigma)
    return decorrelate_binary(model_p, market_prob, weight=weight)
