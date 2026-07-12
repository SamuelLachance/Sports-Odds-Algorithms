"""Shared schedule-window helpers for v2 feature engines.

ISO YYYY-MM-DD strings compare lexicographically in calendar order, so we can
count games in a lookback window without reparsing every recent date to
``datetime.date`` on every feature row (hot path during walk-forward replay).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence


def iso_day(value: date | str | None) -> str | None:
    """Return ``YYYY-MM-DD`` or ``None`` if the value is empty / unparseable."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if len(text) < 10:
        return None
    day = text[:10]
    # Cheap structural guard; full validation only when needed by callers.
    if day[4] != "-" or day[7] != "-":
        return None
    try:
        date.fromisoformat(day)
    except ValueError:
        return None
    return day


def count_games_in_last_n_days(
    recent_iso_dates: Sequence[str],
    game_date: date | str,
    *,
    days: int,
) -> int:
    """Count prior games with ``0 < (game_date - played).days <= days``.

    Equivalent to parsing each ISO date, but uses string bounds:
    ``(game_date - days) <= played < game_date``.
    """
    if days <= 0 or not recent_iso_dates:
        return 0
    gd = iso_day(game_date)
    if gd is None:
        return 0
    try:
        floor = (date.fromisoformat(gd) - timedelta(days=days)).isoformat()
    except ValueError:
        return 0
    count = 0
    for raw in recent_iso_dates:
        played = iso_day(raw)
        if played is None:
            continue
        if floor <= played < gd:
            count += 1
    return count
