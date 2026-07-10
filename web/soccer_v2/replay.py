"""Chronological country-pool replay for the soccer v2 feature engine."""

from __future__ import annotations

from typing import Any, Callable

from web.soccer_v2.data import DIVISION_TIER, DIVISION_TO_LEAGUE
from web.soccer_v2.feature_engine import SoccerFeatureEngine


def merge_country_rows(
    rows_by_division: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """All divisions of one country merged chronologically."""
    merged: list[dict[str, Any]] = []
    for rows in rows_by_division.values():
        merged.extend(rows)
    merged.sort(key=lambda row: (str(row["date"]), str(row["division"])))
    return merged


def replay_country(
    engine: SoccerFeatureEngine,
    rows: list[dict[str, Any]],
    *,
    on_features: Callable[[dict[str, Any], dict[str, float]], None] | None = None,
    stop_before_date: str | None = None,
) -> None:
    """Replay merged country rows; emit tier-1 features before each update.

    ``stop_before_date``: only fold in matches strictly before this ISO date
    (live morning-of-slate semantics).
    """
    for row in rows:
        if stop_before_date is not None and str(row["date"]) >= stop_before_date:
            break
        season = int(row.get("season") or 0)
        if season > engine.current_season:
            engine.start_season(season)
        division = str(row["division"])
        tier = DIVISION_TIER.get(division, 2)
        if on_features is not None and tier == 1:
            league = DIVISION_TO_LEAGUE.get(division, "")
            features = engine.features_for_match(row, league)
            on_features(row, features)
        engine.update_after_match(row, tier=tier)
