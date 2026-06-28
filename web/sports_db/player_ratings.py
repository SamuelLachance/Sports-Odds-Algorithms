"""Player algo ratings on a 0–100 scale (50 = league-average starter).

Consistent across sports for roster cards, player profiles, and team averages.
Full profiles use ESPN season stats; roster-only profiles use position + experience priors.
"""

from __future__ import annotations

import re
from typing import Any

from web.league_profiles import LEAGUE_PROFILES

# Tier thresholds (documented scale)
RATING_ELITE = 85
RATING_GOOD = 75
RATING_AVERAGE = 55
RATING_BASELINE = 50


def _league_category(league: str) -> str:
    return LEAGUE_PROFILES.get(league.lower(), {}).get("category", "basketball")


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _flatten_player_stats(
    season_stats: list[dict[str, Any]] | None,
    overview_stats: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Merge stat categories into a lowercase lookup of numeric values."""
    out: dict[str, float] = {}
    for block in (season_stats or []) + (overview_stats or []):
        for stat in block.get("stats") or []:
            raw_name = stat.get("short_name") or stat.get("name") or ""
            key = _normalize_key(str(raw_name))
            if not key:
                continue
            value = stat.get("value")
            if value is None:
                display = stat.get("display")
                if display is not None:
                    try:
                        value = float(str(display).replace(",", ""))
                    except ValueError:
                        continue
                else:
                    continue
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _stat_match_score(alias_index: int, alias: str, stat_key: str) -> int:
    """Higher score = better match (prefer per-game keys over season totals)."""
    norm_alias = _normalize_key(alias)
    score = 0
    if stat_key == norm_alias:
        score += 200
    if "pergame" in stat_key:
        score += 120
    elif stat_key.endswith("pg"):
        score += 100
    if "percentage" in stat_key or stat_key.endswith("pct"):
        score += 30
    if stat_key in {
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "offensiverebounds",
        "defensiverebounds",
        "fouls",
    }:
        score -= 80
    score -= alias_index
    score += min(len(stat_key), 40)
    return score


def _find_stat(stats: dict[str, float], *aliases: str) -> float | None:
    best: tuple[int, float] | None = None
    for alias_index, alias in enumerate(aliases):
        key = _normalize_key(alias)
        if key in stats:
            score = _stat_match_score(alias_index, alias, key) + 250
            if best is None or score > best[0]:
                best = (score, stats[key])
        if len(key) < 2:
            continue
        for stat_key, value in stats.items():
            if key == stat_key or (len(key) >= 3 and (key in stat_key or stat_key in key)):
                score = _stat_match_score(alias_index, alias, stat_key)
                if best is None or score > best[0]:
                    best = (score, value)
    return best[1] if best else None


def _normalize_rate_stat(value: float) -> float:
    """ESPN often returns shooting % as 0–100 instead of 0–1."""
    if value > 1.0 and value <= 100.0:
        return value / 100.0
    return value


_PER_GAME_TOTAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("pointspergame", "points"),
    ("ppg", "points"),
    ("reboundspergame", "rebounds"),
    ("rpg", "rebounds"),
    ("assistspergame", "assists"),
    ("apg", "assists"),
    ("stealspergame", "steals"),
    ("spg", "steals"),
    ("blockspergame", "blocks"),
    ("bpg", "blocks"),
    ("turnoverspergame", "turnovers"),
    ("topg", "turnovers"),
    ("offensivereboundspergame", "offensiverebounds"),
    ("defensivereboundspergame", "defensiverebounds"),
)

_RATE_STAT_KEYS = frozenset(
    {
        "fg",
        "fgpct",
        "fieldgoalpct",
        "fieldgoalpercentage",
        "3pointfieldgoalpercentage",
        "freethrowpercentage",
    }
)


def _prefer_per_game_basketball_stats(stats: dict[str, float]) -> dict[str, float]:
    """Drop season totals when per-game equivalents are present."""
    out = dict(stats)
    for per_game_key, total_key in _PER_GAME_TOTAL_PAIRS:
        if per_game_key in out:
            out.pop(total_key, None)
    for key, value in list(out.items()):
        if "percentage" in key or key.endswith("pct") or key in _RATE_STAT_KEYS:
            out[key] = _normalize_rate_stat(value)
    return out


_BASKETBALL_BASELINES: dict[str, dict[str, tuple[float, float]]] = {
    "nba": {
        "pts": (12.0, 8.0),
        "reb": (4.5, 2.5),
        "ast": (3.0, 2.5),
        "stl": (0.8, 0.5),
        "blk": (0.5, 0.5),
        "fg": (0.45, 0.06),
        "tov": (1.5, 1.0),
    },
    "wnba": {
        "pts": (9.0, 5.5),
        "reb": (3.5, 2.0),
        "ast": (2.2, 1.8),
        "stl": (0.7, 0.4),
        "blk": (0.4, 0.4),
        "fg": (0.44, 0.06),
        "tov": (1.2, 0.8),
    },
    "cbb": {
        "pts": (10.0, 7.0),
        "reb": (4.0, 2.5),
        "ast": (2.5, 2.0),
        "stl": (0.7, 0.5),
        "blk": (0.4, 0.4),
        "fg": (0.44, 0.06),
        "tov": (1.4, 1.0),
    },
}


def _basketball_baselines(league: str) -> dict[str, tuple[float, float]]:
    return _BASKETBALL_BASELINES.get(league.lower(), _BASKETBALL_BASELINES["nba"])


def _z_score(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return (value - mean) / std


def _composite_to_rating(z: float) -> float:
    """Map composite z-score to 0–100 (50 = average)."""
    rating = 50.0 + 17.0 * z
    return round(max(25.0, min(99.0, rating)), 1)


def _weighted_z(
    stats: dict[str, float],
    metrics: list[tuple[str, tuple[str, ...], float, float, float]],
) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for weight, aliases, mean, std, invert in metrics:
        value = _find_stat(stats, *aliases)
        if value is None:
            continue
        z = _z_score(value, mean, std)
        if invert:
            z = -z
        total += weight * z
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return total / weight_sum


def _is_pitcher(position: str | None, stats: dict[str, float]) -> bool:
    pos = (position or "").upper()
    if pos in {"P", "SP", "RP", "CP", "CL", "LHP", "RHP"}:
        return True
    if _find_stat(stats, "era", "earnedrunaverage") is not None:
        return True
    if _find_stat(stats, "whip") is not None and _find_stat(stats, "avg", "battingaverage") is None:
        return True
    return False


def _is_goalie(position: str | None, stats: dict[str, float]) -> bool:
    pos = (position or "").upper()
    if pos in {"G", "GK", "GOALIE", "GOALTENDER"}:
        return True
    return _find_stat(stats, "gaa", "goalsagainstaverage", "sv", "savepercentage") is not None


def _roster_fallback(
    category: str,
    position: str | None,
    roster_meta: dict[str, Any] | None,
) -> float:
    """Estimate rating when only roster metadata is available (capped range)."""
    meta = roster_meta or {}
    base = RATING_BASELINE
    try:
        exp = float(meta.get("experience") or 0)
    except (TypeError, ValueError):
        exp = 0.0
    base += min(6.0, exp * 0.6)

    pos = (position or "").upper()
    if category == "football":
        if pos in {"QB", "WR", "RB", "TE", "EDGE", "DE", "CB", "LB"}:
            base += 1.5
    elif category == "baseball" and pos in {"P", "SP", "RP", "C"}:
        base += 1.0
    elif category == "hockey" and pos in {"C", "LW", "RW", "D"}:
        base += 0.5

    try:
        age = float(meta.get("age") or 0)
        if 24 <= age <= 30:
            base += 1.0
        elif age > 0 and age < 22:
            base -= 1.0
    except (TypeError, ValueError):
        pass

    return round(max(38.0, min(62.0, base)), 1)


def _basketball_rating(
    stats: dict[str, float],
    position: str | None,
    roster_meta: dict[str, Any] | None,
    *,
    league: str = "nba",
) -> float:
    stats = _prefer_per_game_basketball_stats(stats)
    base = _basketball_baselines(league)
    z = _weighted_z(
        stats,
        [
            (
                0.35,
                ("pointspergame", "ppg", "avgpoints", "pts", "points"),
                base["pts"][0],
                base["pts"][1],
                False,
            ),
            (
                0.20,
                ("reboundspergame", "rpg", "avgrebounds", "reb", "rebounds"),
                base["reb"][0],
                base["reb"][1],
                False,
            ),
            (
                0.20,
                ("assistspergame", "apg", "avgassists", "ast", "assists"),
                base["ast"][0],
                base["ast"][1],
                False,
            ),
            (
                0.10,
                ("stealspergame", "spg", "avgsteals", "stl", "steals"),
                base["stl"][0],
                base["stl"][1],
                False,
            ),
            (
                0.10,
                ("blockspergame", "bpg", "avgblocks", "blk", "blocks"),
                base["blk"][0],
                base["blk"][1],
                False,
            ),
            (
                0.05,
                ("fieldgoalpercentage", "fgpct", "fieldgoalpct", "fg"),
                base["fg"][0],
                base["fg"][1],
                False,
            ),
        ],
    )
    if z is None:
        return _roster_fallback("basketball", position, roster_meta)
    tov = _find_stat(
        stats,
        "turnoverspergame",
        "topg",
        "avgturnovers",
        "tov",
        "turnovers",
    )
    if tov is not None:
        tov_mean, tov_std = base["tov"]
        z -= 0.08 * _z_score(tov, tov_mean, tov_std)
    return _composite_to_rating(z)


def _baseball_rating(
    stats: dict[str, float],
    position: str | None,
    roster_meta: dict[str, Any] | None,
) -> float:
    if _is_pitcher(position, stats):
        strikeouts = _find_stat(stats, "so", "strikeouts", "k", "strikeoutsper9innings", "k9")
        so_mean, so_std = (8.5, 3.0)
        if strikeouts is not None and strikeouts > 40:
            so_mean, so_std = 120.0, 60.0
        z = _weighted_z(
            stats,
            [
                (0.35, ("era", "earnedrunaverage"), 4.20, 1.20, True),
                (0.30, ("whip",), 1.30, 0.20, True),
                (0.20, ("so", "strikeouts", "k", "strikeoutsper9innings", "k9"), so_mean, so_std, False),
                (0.15, ("wins", "w"), 8.0, 5.0, False),
            ],
        )
    else:
        z = _weighted_z(
            stats,
            [
                (0.30, ("avg", "battingaverage", "ba"), 0.250, 0.040, False),
                (0.25, ("ops", "onbaseplusslugging"), 0.720, 0.120, False),
                (0.20, ("hr", "homeruns", "home runs"), 15.0, 12.0, False),
                (0.15, ("rbi", "runsbattedin"), 55.0, 30.0, False),
                (0.10, ("sb", "stolenbases"), 8.0, 10.0, False),
            ],
        )
    if z is None:
        return _roster_fallback("baseball", position, roster_meta)
    return _composite_to_rating(z)


def _hockey_rating(
    stats: dict[str, float],
    position: str | None,
    roster_meta: dict[str, Any] | None,
) -> float:
    if _is_goalie(position, stats):
        z = _weighted_z(
            stats,
            [
                (0.45, ("svpct", "savepercentage", "sv", "svp"), 0.910, 0.015, False),
                (0.35, ("gaa", "goalsagainstaverage"), 2.80, 0.60, True),
                (0.20, ("w", "wins"), 20.0, 10.0, False),
            ],
        )
    else:
        points = _find_stat(stats, "pts", "points", "p")
        goals = _find_stat(stats, "g", "goals")
        assists = _find_stat(stats, "a", "assists")
        if points is None and goals is not None and assists is not None:
            points = goals + assists
        if points is not None:
            stats = {**stats, "pointspergame": points}
        z = _weighted_z(
            stats,
            [
                (0.40, ("pointspergame", "pts", "points", "p"), 0.55, 0.40, False),
                (0.25, ("g", "goals"), 0.25, 0.20, False),
                (0.20, ("a", "assists"), 0.35, 0.25, False),
                (0.15, ("plusminus", "plusminusrating"), 0.0, 8.0, False),
            ],
        )
    if z is None:
        return _roster_fallback("hockey", position, roster_meta)
    return _composite_to_rating(z)


def _football_rating(
    stats: dict[str, float],
    position: str | None,
    roster_meta: dict[str, Any] | None,
) -> float:
    pos = (position or "").upper()
    if pos in {"QB"}:
        metrics = [
            (0.35, ("passingyards", "yds", "passyds", "passyards"), 2500.0, 800.0, False),
            (0.30, ("passingtouchdowns", "td", "passtds", "passingtds"), 18.0, 10.0, False),
            (0.20, ("qbrating", "passerrating", "passer rating"), 90.0, 15.0, False),
            (0.15, ("completionpct", "completionspercentage", "cmppct"), 64.0, 5.0, False),
        ]
    elif pos in {"RB", "FB"}:
        metrics = [
            (0.45, ("rushingyards", "rushyds", "rushyards"), 600.0, 400.0, False),
            (0.30, ("rushingtouchdowns", "rushtds", "rushingtds"), 5.0, 4.0, False),
            (0.25, ("yardspercarry", "ypc", "avgrush"), 4.3, 0.8, False),
        ]
    elif pos in {"WR", "TE"}:
        metrics = [
            (0.40, ("receivingyards", "recyds", "recyards"), 500.0, 350.0, False),
            (0.35, ("receivingtouchdowns", "rectds", "receivingtds"), 4.0, 3.0, False),
            (0.25, ("receptions", "rec"), 40.0, 25.0, False),
        ]
    else:
        metrics = [
            (0.30, ("tackles", "totaltackles", "tkl"), 50.0, 30.0, False),
            (0.30, ("sacks", "sk"), 4.0, 4.0, False),
            (0.20, ("interceptions", "int"), 2.0, 2.0, False),
            (0.20, ("passesdefended", "pdef", "pd"), 5.0, 4.0, False),
        ]
    z = _weighted_z(stats, metrics)
    if z is None:
        return _roster_fallback("football", position, roster_meta)
    return _composite_to_rating(z)


def _soccer_rating(
    stats: dict[str, float],
    position: str | None,
    roster_meta: dict[str, Any] | None,
) -> float:
    z = _weighted_z(
        stats,
        [
            (0.35, ("goals", "g"), 5.0, 5.0, False),
            (0.30, ("assists", "a"), 4.0, 4.0, False),
            (0.20, ("shots", "sh"), 30.0, 20.0, False),
            (0.15, ("saves", "sv"), 40.0, 30.0, False),
        ],
    )
    if z is None:
        return _roster_fallback("soccer", position, roster_meta)
    return _composite_to_rating(z)


def player_algo_rating(
    league: str,
    season_stats: list[dict[str, Any]] | None,
    overview_stats: list[dict[str, Any]] | None,
    position: str | None,
    roster_meta: dict[str, Any] | None = None,
) -> float:
    """Return a 0–100 player rating for the given league and stat payload."""
    category = _league_category(league)
    stats = _flatten_player_stats(season_stats, overview_stats)

    if category == "basketball":
        return _basketball_rating(stats, position, roster_meta, league=league)
    if category == "baseball":
        return _baseball_rating(stats, position, roster_meta)
    if category == "hockey":
        return _hockey_rating(stats, position, roster_meta)
    if category == "football":
        return _football_rating(stats, position, roster_meta)
    if category == "soccer":
        return _soccer_rating(stats, position, roster_meta)
    return _roster_fallback(category, position, roster_meta)


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


def team_average_player_rating(ratings: list[float]) -> float | None:
    """Average roster player rating (active contributors only)."""
    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 1)


def enrich_team_roster_ratings(
    league: str,
    roster: list[dict[str, Any]],
    ratings_by_id: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], float | None]:
    """Attach algo_rating to each roster row and compute team average."""
    ratings_by_id = ratings_by_id or {}
    enriched: list[dict[str, Any]] = []
    values: list[float] = []
    for player in roster:
        pid = str(player.get("id") or "")
        rating = ratings_by_id.get(pid)
        if rating is None:
            rating = player_algo_rating(league, [], [], player.get("position"), player)
        row = {**player, "algo_rating": rating}
        enriched.append(row)
        values.append(rating)
    return enriched, team_average_player_rating(values)
