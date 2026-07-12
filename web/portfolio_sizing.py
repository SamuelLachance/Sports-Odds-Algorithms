"""Hubáček / modern-portfolio inspired stake sizing.

Slightly more conservative than raw quarter-Kelly when same-slate bets are
correlated (shared league, sides, or market regime). Stake is expressed in
units where ``bankroll_units=100`` means 1u ≈ 1% of bankroll.
"""

from __future__ import annotations

import math


def portfolio_stake_units(
    ev_pct: float,
    kelly_fraction: float,
    *,
    bankroll_units: float = 100.0,
    max_units: float = 3.0,
    min_units: float = 0.25,
    correlation_penalty: float = 0.15,
) -> float:
    """Quarter-Kelly stake with a correlation haircut and EV soft dampener.

    ``kelly_fraction`` is full Kelly as a fraction of bankroll (0.04 = 4%).
    ``correlation_penalty`` shrinks the stake (0.15 → 15% haircut) to reflect
    same-slate correlation — more conservative than raw quarter-Kelly.

    Returns units clamped to ``[min_units, max_units]``. Non-positive Kelly or
    explicitly negative EV yields ``min_units`` (callers that want a default 1u
    should gate first). Missing EV should be passed as ``0`` so sizing still
    follows Kelly without the negative-EV short-circuit. Non-finite Kelly/EV
    (±inf / NaN) fail closed to ``min_units`` — NaN comparisons are never true,
    so they must not slip through as a full-size stake.
    """
    try:
        kelly = float(kelly_fraction)
        ev = float(ev_pct)
        bankroll = max(float(bankroll_units), 1.0)
        penalty_raw = float(correlation_penalty)
        lo = float(min_units)
        hi = float(max_units)
    except (TypeError, ValueError):
        try:
            return float(min_units)
        except (TypeError, ValueError):
            return 0.25

    if not math.isfinite(kelly) or not math.isfinite(ev):
        return lo
    if not math.isfinite(bankroll) or not math.isfinite(lo) or not math.isfinite(hi):
        return 0.25

    # Do not Kelly-size a non-positive edge or a non-positive Kelly fraction.
    if kelly <= 0 or ev < 0:
        return lo

    # Quarter-Kelly in units (bankroll_units=100 → 1u = 1% bankroll).
    base = (kelly / 4.0) * bankroll

    penalty = min(max(penalty_raw, 0.0), 0.5)
    base *= 1.0 - penalty

    # Soft dampener: extreme printed EV is often sample noise; do not size up.
    if ev > 40.0:
        base *= 0.85
    elif ev > 25.0:
        base *= 0.92

    if base <= 0:
        return lo
    return round(min(hi, max(lo, base)), 2)
