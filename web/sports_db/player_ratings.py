"""Player algo ratings on a 0–99 scale.

Primary source: curated third-party game OVR (2K, Madden, NHL, MLB The Show, EA FC).
Fallback chain when no CSV player match:
  1. External game DB lookup (data/ratings/*.csv)
  2. Predictive model team-strength spread (see model_player_ratings)
  3. Conservative roster prior (~50±, never ESPN stat z-scores)
"""

from __future__ import annotations

from typing import Any

from web.sports_db.external_ratings import (
    RATING_PRIOR_SOURCE,
    clear_rating_cache,
    conservative_prior_rating,
    match_external_rating,
    rating_source_label as external_rating_source_label,
)
from web.sports_db.model_player_ratings import (
    league_model_rating_context,
    player_model_rating,
    rating_layer_label,
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
    """Human label for UI tooltips (e.g. 2K '26, Madden '26, Matrix model)."""
    key = (source or "").lower()
    if key and key not in {"model", RATING_PRIOR_SOURCE}:
        return external_rating_source_label(key, year)
    if key == RATING_PRIOR_SOURCE:
        return external_rating_source_label(RATING_PRIOR_SOURCE, year)
    if key == "model" and layer:
        return rating_layer_label(layer)
    return "Model"


def _external_payload(result: Any) -> dict[str, Any]:
    return {
        "algo_rating": result.rating,
        "rating_source": result.rating_source,
        "rating_year": result.rating_year,
        "rating_tier": rating_tier(result.rating),
        "matched": result.matched,
    }


def _model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rating = payload["algo_rating"]
    return {
        "algo_rating": rating,
        "rating_source": payload["rating_source"],
        "rating_layer": payload.get("rating_layer"),
        "rating_tier": rating_tier(rating),
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
    """Return rating fields using external → model spread → prior fallback."""
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
    )
    if external:
        return _external_payload(external)

    if cutoff_date:
        team_ratings, layer = league_model_rating_context(league, cutoff_date)
        if team_ratings:
            return _model_payload(
                player_model_rating(
                    league,
                    team_abbr or "",
                    meta,
                    cutoff_date,
                    team_ratings=team_ratings,
                    layer=layer,
                )
            )

    return _external_payload(
        conservative_prior_rating(
            league,
            meta.get("position"),
            meta,
        )
    )


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
    )["algo_rating"]


def enrich_team_roster_ratings(
    league: str,
    roster: list[dict[str, Any]],
    ratings_by_id: dict[str, float | dict[str, Any]] | None = None,
    *,
    team_abbr: str | None = None,
    cutoff_date: str | None = None,
) -> tuple[list[dict[str, Any]], float | None]:
    """Attach algo_rating to each roster row and compute team average."""
    if not cutoff_date:
        cutoff_date = "12-31-2099"

    cached_by_id: dict[str, dict[str, Any]] = {}
    if ratings_by_id:
        for pid, cached in ratings_by_id.items():
            if isinstance(cached, dict) and cached.get("algo_rating") is not None:
                cached_by_id[pid] = cached
            elif not isinstance(cached, dict):
                cached_by_id[pid] = {
                    "algo_rating": float(cached),
                    "rating_source": "cached",
                }

    enriched: list[dict[str, Any]] = []
    values: list[float] = []
    for player in roster:
        pid = str(player.get("id") or "")
        if pid and pid in cached_by_id:
            fields = {k: v for k, v in cached_by_id[pid].items() if v is not None}
            row = {**player, **fields}
        else:
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
