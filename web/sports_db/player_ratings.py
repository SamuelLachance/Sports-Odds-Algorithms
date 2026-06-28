"""Player algo ratings on a 0–99 scale.

Primary and only source: curated third-party game OVR (2K, Madden, NHL, MLB The Show, EA FC).
When no publisher match exists, algo_rating is omitted (no model spread or roster prior).
"""

from __future__ import annotations

from typing import Any

from web.sports_db.external_ratings import (
    RATING_PRIOR_SOURCE,
    clear_rating_cache,
    is_publisher_rating_source,
    match_external_rating,
    rating_source_label as external_rating_source_label,
)

# Tier thresholds (documented scale)
RATING_ELITE = 85
RATING_GOOD = 75
RATING_AVERAGE = 55
RATING_BASELINE = 50


def team_average_player_rating(ratings: list[float]) -> float | None:
    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 1)


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


def rating_source_label(
    source: str | None,
    layer: str | None = None,
    *,
    year: int | None = None,
) -> str:
    """Human label for UI tooltips (e.g. 2K '26, Madden '26)."""
    del layer
    key = (source or "").lower()
    if key and key != RATING_PRIOR_SOURCE:
        return external_rating_source_label(key, year)
    if key == RATING_PRIOR_SOURCE:
        return external_rating_source_label(RATING_PRIOR_SOURCE, year)
    return "Missing"


def _external_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rating_source": result.rating_source,
        "rating_year": result.rating_year,
        "matched": result.matched,
    }
    if (
        is_publisher_rating_source(result.rating_source)
        and result.rating is not None
    ):
        payload["algo_rating"] = result.rating
        payload["rating_tier"] = rating_tier(result.rating)
    else:
        payload["algo_rating"] = None
        payload["rating_tier"] = "unknown"
    return payload


def _missing_payload() -> dict[str, Any]:
    return {
        "algo_rating": None,
        "rating_source": RATING_PRIOR_SOURCE,
        "rating_year": None,
        "rating_tier": "unknown",
        "matched": False,
    }


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
    """Return rating fields from external publisher data only."""
    del cutoff_date
    meta = dict(roster_meta or {})
    if player_name:
        meta.setdefault("name", player_name)
    if position:
        meta.setdefault("position", position)
    if espn_id:
        meta.setdefault("id", espn_id)

    external = match_external_rating(
        league,
        meta.get("name"),
        team_abbr=team_abbr,
        espn_id=meta.get("id"),
        nationality=str(meta.get("nationality") or meta.get("country") or ""),
    )
    if external:
        return _external_payload(external)
    return _missing_payload()


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
    """Return player overall (0–99). ESPN season stats are ignored."""
    del season_stats, overview_stats
    return resolve_player_rating(
        league,
        player_name=player_name,
        team_abbr=team_abbr,
        espn_id=espn_id,
        position=position,
        roster_meta=roster_meta,
        cutoff_date=cutoff_date,
    ).get("algo_rating")


def enrich_team_roster_ratings(
    league: str,
    roster: list[dict[str, Any]],
    ratings_by_id: dict[str, float | dict[str, Any]] | None = None,
    *,
    team_abbr: str | None = None,
    cutoff_date: str | None = None,
) -> tuple[list[dict[str, Any]], float | None]:
    """Attach publisher OVR to each roster row and compute team average."""
    del ratings_by_id
    if not cutoff_date:
        cutoff_date = "12-31-2099"

    enriched: list[dict[str, Any]] = []
    values: list[float] = []
    for player in roster:
        pid = str(player.get("id") or "")
        row = {
            **player,
            **resolve_player_rating(
                league,
                player_name=player.get("name"),
                team_abbr=team_abbr,
                espn_id=pid or None,
                position=player.get("position"),
                roster_meta=player,
                cutoff_date=cutoff_date,
            ),
        }
        row.pop("rating_layer", None)
        enriched.append(row)
        if row.get("algo_rating") is not None:
            values.append(float(row["algo_rating"]))

    return enriched, team_average_player_rating(values)


__all__ = [
    "RATING_AVERAGE",
    "RATING_BASELINE",
    "RATING_ELITE",
    "RATING_GOOD",
    "clear_rating_cache",
    "enrich_team_roster_ratings",
    "player_algo_rating",
    "rating_source_label",
    "rating_tier",
    "resolve_player_rating",
    "team_average_player_rating",
]
