"""Player algo ratings on a 0–99 scale from external game databases.

Primary source: curated OVR lists in data/ratings/ (2K, Madden, NHL, MLB The Show,
EA FC, FM). Unmatched roster players inherit team CSV averages or a conservative
league prior — never ESPN stat z-scores or predictive model team-strength spread.
"""

from __future__ import annotations

from typing import Any

from web.sports_db.external_ratings import (
    RATING_DERIVED_SOURCE,
    RATING_PRIOR_SOURCE,
    clear_rating_cache,
    lookup_player_rating,
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
    """Human label for UI tooltips, e.g. 2K '26 or Estimated."""
    del layer  # legacy model layer — no longer used for player cards
    return external_rating_source_label(source, year)


def _result_to_payload(result: Any) -> dict[str, Any]:
    return {
        "algo_rating": result.rating,
        "rating_source": result.rating_source,
        "rating_year": result.rating_year,
        "rating_tier": rating_tier(result.rating),
        "matched": result.matched,
    }


class PlayerRatingModel:
    """External-game-database player rating model with team-context fallback."""

    def rate_player(
        self,
        league: str,
        roster_row: dict[str, Any],
        *,
        team_abbr: str | None = None,
    ) -> dict[str, Any]:
        """Return algo_rating and metadata for one roster row."""
        meta = roster_row or {}
        result = lookup_player_rating(
            league,
            meta.get("name"),
            team_abbr=team_abbr,
            espn_id=meta.get("id"),
            position=meta.get("position"),
            roster_meta=meta,
        )
        return _result_to_payload(result)

    def enrich_roster(
        self,
        league: str,
        roster: list[dict[str, Any]],
        *,
        team_abbr: str | None = None,
        ratings_by_id: dict[str, float | dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], float | None]:
        """Attach external-based algo_rating to each roster row."""
        ratings_by_id = ratings_by_id or {}
        enriched: list[dict[str, Any]] = []
        values: list[float] = []

        for player in roster:
            pid = str(player.get("id") or "")
            cached = ratings_by_id.get(pid) if pid else None
            use_cached = (
                isinstance(cached, dict)
                and cached.get("algo_rating") is not None
                and cached.get("rating_source") not in (None, "model")
            )
            if use_cached:
                rating = float(cached["algo_rating"])
                source = cached.get("rating_source")
                row = {
                    **player,
                    "algo_rating": rating,
                    "rating_source": source,
                    "rating_year": cached.get("rating_year"),
                    "rating_tier": rating_tier(rating),
                    "matched": source not in (RATING_DERIVED_SOURCE, RATING_PRIOR_SOURCE),
                }
            else:
                row = {**player, **self.rate_player(league, player, team_abbr=team_abbr)}
            enriched.append(row)
            values.append(row["algo_rating"])

        avg = team_average_player_rating(values)
        return enriched, avg

    @staticmethod
    def clear_cache() -> None:
        clear_rating_cache()


_DEFAULT_MODEL = PlayerRatingModel()


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
    """Return external-based rating fields for one player."""
    del cutoff_date  # ratings are game-edition snapshots, not date-sensitive
    meta = dict(roster_meta or {})
    if player_name:
        meta.setdefault("name", player_name)
    if position:
        meta.setdefault("position", position)
    if espn_id:
        meta.setdefault("id", espn_id)
    return _DEFAULT_MODEL.rate_player(league, meta, team_abbr=team_abbr)


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
    """Return player overall (0–99) from external game databases. ESPN stats are ignored."""
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
    """Attach external-based algo_rating to each roster row and compute team average."""
    del cutoff_date
    return _DEFAULT_MODEL.enrich_roster(
        league,
        roster,
        team_abbr=team_abbr,
        ratings_by_id=ratings_by_id,
    )


__all__ = [
    "PlayerRatingModel",
    "RATING_AVERAGE",
    "RATING_BASELINE",
    "RATING_ELITE",
    "RATING_GOOD",
    "enrich_team_roster_ratings",
    "player_algo_rating",
    "rating_source_label",
    "rating_tier",
    "resolve_player_rating",
    "team_average_player_rating",
]
