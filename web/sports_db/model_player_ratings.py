"""Player ratings derived from internal predictive models (not ESPN stat z-scores).

Team strength comes from the same sport-specific models used in blend_service
(basketball matrix, hockey Poisson xG, baseball Elo, nfelo, soccer Dixon–Coles,
power ratings fallback). Roster players inherit team strength plus a small
position/experience spread so stars rank above bench on the same squad.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from web.baseball_pred_model import (
    extract_team_elo_strengths,
    get_baseball_pred_context,
    is_baseball_league,
)
from web.basketball_pred_model import (
    extract_team_matrix_strengths,
    get_basketball_pred_context,
    is_basketball_league,
)
from web.football_pred_model import (
    extract_team_nfelo_strengths,
    get_football_pred_context,
    is_football_league,
)
from web.hockey_pred_model import (
    extract_team_xg_strengths,
    get_hockey_pred_context,
    is_hockey_league,
)
from web.league_profiles import LEAGUE_PROFILES
from web.power_model import extract_team_power_strengths
from web.season_games import get_league_power_context
from web.soccer_pred_model import extract_team_soccer_strengths, get_soccer_pred_context

RATING_LAYER_LABELS: dict[str, str] = {
    "basketball_matrix": "Matrix model",
    "hockey_poisson": "Poisson xG",
    "baseball_elo": "Elo",
    "nfelo": "nfelo",
    "soccer_elo": "Soccer Elo",
    "power_ratings": "Power ratings",
}

RATING_BASELINE = 50.0


def rating_layer_label(layer: str | None) -> str:
    if not layer:
        return "Model"
    return RATING_LAYER_LABELS.get(layer, layer.replace("_", " ").title())


def _normalize_name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _league_category(league: str) -> str:
    return LEAGUE_PROFILES.get(league.lower(), {}).get("category", "basketball")


def _normalize_strengths_to_100(raw: dict[str, float]) -> dict[str, float]:
    """Map native model strengths to a 0–100 scale (50 = league average)."""
    if not raw:
        return {}
    values = list(raw.values())
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return {key: RATING_BASELINE for key in raw}
    mid = (lo + hi) / 2.0
    half_span = (hi - lo) / 2.0
    out: dict[str, float] = {}
    for key, value in raw.items():
        z = (value - mid) / half_span
        rating = RATING_BASELINE + 22.0 * z
        out[key] = round(max(25.0, min(99.0, rating)), 1)
    return out


def _raw_team_strengths(league: str, cutoff_date: str) -> tuple[dict[str, float], str]:
    """Return per-team native strengths and the model layer id."""
    league = league.lower()

    if is_basketball_league(league):
        model = get_basketball_pred_context(league, cutoff_date)
        if model:
            return extract_team_matrix_strengths(model), "basketball_matrix"

    if is_hockey_league(league):
        model = get_hockey_pred_context(league, cutoff_date)
        if model:
            return extract_team_xg_strengths(model), "hockey_poisson"

    if is_baseball_league(league):
        model = get_baseball_pred_context(league, cutoff_date)
        if model:
            return extract_team_elo_strengths(model), "baseball_elo"

    if is_football_league(league):
        model = get_football_pred_context(league, cutoff_date)
        if model:
            return extract_team_nfelo_strengths(model), "nfelo"

    category = _league_category(league)
    if category == "soccer":
        model = get_soccer_pred_context(league, cutoff_date)
        if model:
            return extract_team_soccer_strengths(model), "soccer_elo"

    context = get_league_power_context(league, cutoff_date)
    if context:
        teams, _games, _param = context
        return extract_team_power_strengths(teams), "power_ratings"

    return {}, "power_ratings"


@lru_cache(maxsize=64)
def league_model_rating_context(league: str, cutoff_date: str) -> tuple[dict[str, float], str]:
    """Cached 0–100 team ratings and layer id for a league at cutoff."""
    raw, layer = _raw_team_strengths(league, cutoff_date)
    return _normalize_strengths_to_100(raw), layer


def _intra_team_delta(
    category: str,
    position: str | None,
    roster_meta: dict[str, Any] | None,
) -> float:
    """Relative skill spread within a team (roster metadata only, no ESPN stat z-scores)."""
    meta = roster_meta or {}
    delta = 0.0

    try:
        exp = float(meta.get("experience") or 0)
    except (TypeError, ValueError):
        exp = 0.0
    delta += min(7.0, exp * 0.75)

    pos = (position or "").upper()
    if category == "football":
        if pos in {"QB"}:
            delta += 4.0
        elif pos in {"WR", "RB", "TE", "EDGE", "DE", "CB", "LB"}:
            delta += 2.0
        elif pos in {"K", "P", "LS"}:
            delta -= 3.0
    elif category == "baseball":
        if pos in {"SP", "RP", "P"}:
            delta += 1.5
        elif pos in {"C"}:
            delta += 0.5
    elif category == "hockey":
        if pos in {"G", "GK"}:
            delta += 2.0
        elif pos in {"C", "LW", "RW", "D"}:
            delta += 0.5
    elif category == "basketball":
        if pos in {"G", "PG", "SG"}:
            delta += 1.0
        elif pos in {"F", "SF", "PF", "C"}:
            delta += 0.5
    elif category == "soccer":
        if pos in {"F", "FW", "ST", "CF"}:
            delta += 1.5
        elif pos in {"M", "MF", "AM", "CM", "DM"}:
            delta += 0.5

    try:
        age = float(meta.get("age") or 0)
        if 24 <= age <= 30:
            delta += 1.0
        elif age > 0 and age < 22:
            delta -= 1.0
    except (TypeError, ValueError):
        pass

    return delta


def _player_rating_payload(
    rating: float,
    layer: str,
) -> dict[str, Any]:
    return {
        "algo_rating": rating,
        "rating_source": "model",
        "rating_layer": layer,
    }


def player_model_rating(
    league: str,
    team_abbr: str,
    player: dict[str, Any],
    cutoff_date: str,
    *,
    team_ratings: dict[str, float] | None = None,
    layer: str | None = None,
    team_mean_delta: float = 0.0,
) -> dict[str, Any]:
    """Rating for one roster player from predictive model team strength."""
    if team_ratings is None or layer is None:
        team_ratings, layer = league_model_rating_context(league, cutoff_date)

    team_key = (team_abbr or "").lower()
    team_base = team_ratings.get(team_key, RATING_BASELINE)
    delta = _intra_team_delta(_league_category(league), player.get("position"), player)
    spread = (delta - team_mean_delta) * 0.65
    rating = round(max(25.0, min(99.0, team_base + spread)), 1)
    return _player_rating_payload(rating, layer)


def match_model_player(
    player_name: str,
    team_abbr: str,
    model_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Lookup a pre-built model player row by normalized name + team."""
    team_key = (team_abbr or "").lower()
    name_key = _normalize_name_key(player_name)
    team_bucket = model_index.get(team_key) or {}
    return team_bucket.get(name_key)


def build_model_player_index(
    league: str,
    cutoff_date: str,
    rosters_by_team: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index model ratings by team abbr and normalized player name."""
    team_ratings, layer = league_model_rating_context(league, cutoff_date)
    index: dict[str, dict[str, dict[str, Any]]] = {}

    for team_abbr, roster in rosters_by_team.items():
        deltas = [
            _intra_team_delta(_league_category(league), p.get("position"), p)
            for p in roster
        ]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        bucket: dict[str, dict[str, Any]] = {}
        for player, delta in zip(roster, deltas):
            payload = player_model_rating(
                league,
                team_abbr,
                player,
                cutoff_date,
                team_ratings=team_ratings,
                layer=layer,
                team_mean_delta=mean_delta,
            )
            name_key = _normalize_name_key(str(player.get("name") or ""))
            if name_key:
                bucket[name_key] = payload
        index[team_abbr.lower()] = bucket
    return index


def enrich_team_roster_model_ratings(
    league: str,
    team_abbr: str,
    roster: list[dict[str, Any]],
    cutoff_date: str,
    ratings_by_id: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], float | None, str]:
    """Attach model-derived algo_rating fields to roster rows."""
    ratings_by_id = ratings_by_id or {}
    team_ratings, layer = league_model_rating_context(league, cutoff_date)

    deltas = [
        _intra_team_delta(_league_category(league), p.get("position"), p)
        for p in roster
    ]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    enriched: list[dict[str, Any]] = []
    values: list[float] = []
    for player in roster:
        pid = str(player.get("id") or "")
        if pid and pid in ratings_by_id:
            rating = ratings_by_id[pid]
            row = {
                **player,
                **_player_rating_payload(rating, layer),
            }
        else:
            payload = player_model_rating(
                league,
                team_abbr,
                player,
                cutoff_date,
                team_ratings=team_ratings,
                layer=layer,
                team_mean_delta=mean_delta,
            )
            row = {**player, **payload}
        enriched.append(row)
        values.append(row["algo_rating"])

    avg = round(sum(values) / len(values), 1) if values else None
    return enriched, avg, layer


def team_average_player_rating(ratings: list[float]) -> float | None:
    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 1)
