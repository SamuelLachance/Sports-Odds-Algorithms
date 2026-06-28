"""Player algo ratings on a 0–100 scale from internal predictive models.

Ratings inherit sport-specific team strength (matrix, Poisson xG, Elo, nfelo, etc.)
plus a roster position/experience spread. External game DBs and ESPN stat z-scores
are not used — see web.sports_db.model_player_ratings.
"""

from __future__ import annotations

from typing import Any

from web.sports_db.model_player_ratings import (
    RATING_LAYER_LABELS,
    enrich_team_roster_model_ratings,
    player_model_rating,
    rating_layer_label,
    team_average_player_rating,
)

# Tier thresholds (documented scale)
RATING_ELITE = 85
RATING_GOOD = 75
RATING_AVERAGE = 55
RATING_BASELINE = 50


def resolve_player_rating(
    league: str,
    *,
    player_name: str | None = None,
    team_abbr: str | None = None,
    espn_id: str | None = None,
    position: str | None = None,
    roster_meta: dict[str, Any] | None = None,
    cutoff_date: str | None = None,
) -> dict[str, Any]:
    """Return model-derived rating fields for one player."""
    meta = roster_meta or {}
    if not cutoff_date:
        cutoff_date = "12-31-2099"
    payload = player_model_rating(
        league,
        team_abbr or "",
        {
            **meta,
            "name": player_name or meta.get("name"),
            "position": position or meta.get("position"),
            "id": espn_id or meta.get("id"),
        },
        cutoff_date,
    )
    return payload


def player_algo_rating(
    league: str,
    season_stats: list[dict[str, Any]] | None = None,
    overview_stats: list[dict[str, Any]] | None = None,
    position: str | None = None,
    roster_meta: dict[str, Any] | None = None,
    *,
    player_name: str | None = None,
    team_abbr: str | None = None,
    espn_id: str | None = None,
    cutoff_date: str | None = None,
) -> float:
    """Return player overall (0–100) from predictive models. ESPN stats are ignored."""
    return resolve_player_rating(
        league,
        player_name=player_name,
        team_abbr=team_abbr,
        espn_id=espn_id,
        position=position,
        roster_meta=roster_meta,
        cutoff_date=cutoff_date,
    )["algo_rating"]


def rating_tier(rating: float | None) -> str:
    """FM-style tier label for UI color coding."""
    if rating is None:
        return "unknown"
    if rating >= RATING_ELITE:
        return "elite"
    if rating >= RATING_GOOD:
        return "good"
    if rating >= RATING_AVERAGE:
        return "average"
    return "low"


def rating_source_label(source: str | None, layer: str | None = None) -> str:
    if source == "model" and layer:
        return rating_layer_label(layer)
    return "Model"


def enrich_team_roster_ratings(
    league: str,
    roster: list[dict[str, Any]],
    ratings_by_id: dict[str, float | dict[str, Any]] | None = None,
    *,
    team_abbr: str | None = None,
    cutoff_date: str | None = None,
) -> tuple[list[dict[str, Any]], float | None]:
    """Attach model algo_rating to each roster row and compute team average."""
    if not cutoff_date:
        cutoff_date = "12-31-2099"

    flat_by_id: dict[str, float] = {}
    if ratings_by_id:
        for pid, cached in ratings_by_id.items():
            if isinstance(cached, dict):
                if cached.get("algo_rating") is not None:
                    flat_by_id[pid] = float(cached["algo_rating"])
            else:
                flat_by_id[pid] = float(cached)

    enriched, avg, _layer = enrich_team_roster_model_ratings(
        league,
        team_abbr or "",
        roster,
        cutoff_date,
        flat_by_id or None,
    )
    return enriched, avg


__all__ = [
    "RATING_AVERAGE",
    "RATING_BASELINE",
    "RATING_ELITE",
    "RATING_GOOD",
    "RATING_LAYER_LABELS",
    "enrich_team_roster_ratings",
    "player_algo_rating",
    "rating_source_label",
    "rating_tier",
    "resolve_player_rating",
    "team_average_player_rating",
]
