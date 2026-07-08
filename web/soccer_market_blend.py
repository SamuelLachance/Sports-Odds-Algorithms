"""Market-calibrated blending for Soccer Path A display probabilities."""

from __future__ import annotations

from web.soccer_meta_model import normalize_threeway

Threeway = tuple[float, float, float]


def blend_model_with_market(
    model_probs: Threeway,
    market_probs: Threeway | None,
    *,
    model_weight: float,
) -> Threeway:
    """Convex blend of model and devigged market (0–100 scale)."""
    if market_probs is None:
        return model_probs
    weight = min(max(float(model_weight), 0.0), 1.0)
    home = weight * model_probs[0] + (1.0 - weight) * market_probs[0]
    draw = weight * model_probs[1] + (1.0 - weight) * market_probs[1]
    away = weight * model_probs[2] + (1.0 - weight) * market_probs[2]
    return normalize_threeway(home, draw, away)
