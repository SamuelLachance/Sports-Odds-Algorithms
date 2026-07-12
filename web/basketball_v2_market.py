"""Shared market-feature helpers for NBA/WNBA v2 live inference.

Keeps pure vs market-aware head selection identical across basketball leagues
so odds wiring cannot drift between ``nba_v2.live`` and ``wnba_v2.live``.
"""

from __future__ import annotations

import math
from typing import Any


# NBA/WNBA live spreads beyond this are almost always juice/ML dumps.
_MAX_BASKETBALL_SPREAD = 40.0


def _coerce_home_spread(home_spread: Any) -> float | None:
    """Fail-soft spread parse aligned with ESPN ``_parse_spread_line``.

    ``PK`` / ``EVEN`` → 0.0; junk / bool / |x|≥100 juice dumps → None
    (do not raise or invent market-aware margin features).
    """
    if home_spread is None or home_spread == "":
        return None
    # bool is a subclass of int; False→0.0 must not invent a pick'em market.
    if isinstance(home_spread, bool):
        return None
    text = str(home_spread).strip()
    upper = text.upper()
    if upper in {"PK", "EVEN"}:
        return 0.0
    if upper in {"OFF", "N/A", "NA"}:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    # American juice/ML magnitudes (|x| ≥ 100) are never real handicaps.
    if abs(value) >= 100.0:
        return None
    if abs(value) > _MAX_BASKETBALL_SPREAD:
        return None
    return value


def devig_home_prob(home_ml: float | None, away_ml: float | None) -> float | None:
    """Two-way no-vig implied home win probability from American moneylines."""

    def implied(american: float) -> float | None:
        # bool is a subclass of int; False→0.0 must not invent EVEN (+100).
        if american is None or isinstance(american, bool):
            return None
        try:
            american = float(american)
        except (TypeError, ValueError):
            return None
        # ESPN EVEN sometimes arrives as numeric 0 — treat as +100.
        if american == 0:
            american = 100.0
        if american >= 100:
            return 100.0 / (american + 100.0)
        if american <= -100:
            return -american / (-american + 100.0)
        return None

    if home_ml is None or away_ml is None:
        return None
    if isinstance(home_ml, bool) or isinstance(away_ml, bool):
        return None
    ph = implied(home_ml)
    pa = implied(away_ml)
    if ph is None or pa is None or ph + pa <= 0:
        return None
    return ph / (ph + pa)


def apply_market_features(
    features: dict[str, float],
    *,
    home_moneyline: float | None = None,
    away_moneyline: float | None = None,
    home_spread: float | None = None,
) -> tuple[bool, bool]:
    """Inject ``mkt_*`` / ``has_*`` keys used by basketball v2 market heads.

    Returns ``(has_market, has_spread)``.
    """
    mkt_prob = devig_home_prob(home_moneyline, away_moneyline)
    has_market = mkt_prob is not None
    spread = _coerce_home_spread(home_spread)
    has_spread = spread is not None
    features["mkt_home_prob"] = float(mkt_prob) if has_market else 0.5
    features["has_market"] = 1.0 if has_market else 0.0
    features["mkt_home_spread"] = float(spread) if has_spread else 0.0
    features["has_spread"] = 1.0 if has_spread else 0.0
    return has_market, has_spread


def resolve_market_heads(
    art: dict[str, Any],
    *,
    has_market: bool,
    has_spread: bool,
) -> tuple[bool, bool, str, list[str], list[str]]:
    """Choose pure vs market heads and the feature columns each head needs.

    Returns
    -------
    use_market_clf, use_market_margin, model_variant, clf_cols, margin_cols
    """
    use_market_clf = bool(
        has_market
        and art.get("clf_market") is not None
        and art.get("lr_market") is not None
        and art.get("calibrator_market") is not None
    )
    use_market_margin = bool(has_spread and art.get("margin_market") is not None)
    model_variant = "market_aware" if (use_market_clf or use_market_margin) else "pure"

    pure_cols = list(art["feature_columns"])
    clf_extra = list(
        art.get("clf_market_features") or ("mkt_home_prob", "has_market")
    )
    margin_extra = list(
        art.get("margin_market_features") or ("mkt_home_spread", "has_spread")
    )
    clf_cols = pure_cols + clf_extra if use_market_clf else pure_cols
    margin_cols = pure_cols + margin_extra if use_market_margin else pure_cols
    return use_market_clf, use_market_margin, model_variant, clf_cols, margin_cols
