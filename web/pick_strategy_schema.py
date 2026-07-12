"""Validate ``data/pick_strategy.json`` shapes on load.

Shared by ``web.pick_strategy`` and ``web.hubacek_picks`` so both loaders
reject corrupt entries without circular imports.
"""

from __future__ import annotations

from typing import Any

VALID_BET_TYPES = frozenset({"spread", "moneyline", "soccer_1x2", "none"})

_NUMERIC_KEYS = frozenset(
    {
        "min_edge",
        "min_ev_pct",
        "min_market_gap_pp",
        "min_win_confidence_pp",
        "min_spread_cover_gap_pp",
        "min_spread_confidence_pp",
        "min_spread_point_edge",
        "min_profit_score",
        "min_kelly_pct",
        "ml_lo",
        "ml_hi",
        "backtest_roi_pct",
        "backtest_bets",
        "backtest_units",
        "backtest_open_roi_pct",
        "backtest_open_bets",
    }
)

_INT_KEYS = frozenset({"backtest_bets", "backtest_open_bets"})
_META_KEYS = frozenset({"policy", "generated_at"})


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def validate_strategy_entry(entry: Any) -> dict[str, Any] | None:
    """Return a cleaned league/default entry, or None if unusable."""
    if not isinstance(entry, dict):
        return None

    cleaned: dict[str, Any] = {}
    bet_type = entry.get("bet_type")
    if bet_type is not None:
        if not isinstance(bet_type, str) or bet_type.lower() not in VALID_BET_TYPES:
            return None
        cleaned["bet_type"] = bet_type.lower()

    if "enabled" in entry:
        cleaned["enabled"] = bool(entry["enabled"])

    for key in _NUMERIC_KEYS:
        if key not in entry or entry[key] is None:
            continue
        num = _as_number(entry[key])
        if num is None:
            continue
        cleaned[key] = int(num) if key in _INT_KEYS else num

    sides = entry.get("allowed_sides")
    if isinstance(sides, list):
        cleaned_sides = [
            str(side).lower()
            for side in sides
            if isinstance(side, str) and side.strip()
        ]
        if cleaned_sides:
            cleaned["allowed_sides"] = cleaned_sides

    note = entry.get("note")
    if isinstance(note, str) and note.strip():
        cleaned["note"] = note

    return cleaned


def validate_pick_strategy_payload(payload: Any) -> dict[str, Any]:
    """Normalize a raw JSON payload into a safe strategy config dict."""
    if not isinstance(payload, dict):
        return {}

    out: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if key in _META_KEYS:
            if isinstance(value, str):
                out[key] = value
            continue
        cleaned = validate_strategy_entry(value)
        if cleaned is not None:
            out[key] = cleaned
    return out
