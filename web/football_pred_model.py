"""American football win probability via nfelo-style Elo (ported from greerreNFL/nfelo).

Source: https://github.com/greerreNFL/nfelo
- Utilities/spread_translation.elo_to_prob
- Utilities/elo_shift.calc_shift (margin-based rating updates)
- config.json nfelo_v4.02 parameters for NFL; CFB uses scaled equivalents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from web.league_profiles import LEAGUE_PROFILES
from web.season_games import load_league_completed_games

FOOTBALL_LEAGUES: tuple[str, ...] = ("nfl", "cfb")

MIN_LEAGUE_GAMES = 20
MIN_TEAM_GAMES = 3

# nfelo v4.02 (NFL) from config.json; CFB uses same z/k with higher HFA and K.
LEAGUE_CONFIG: dict[str, dict[str, float]] = {
    "nfl": {
        "z": 401.62,
        "k": 9.114,
        "b": 9.999,
        "hfa": 48.0,
        "start_elo": 1505.0,
        "points_per_elo": 25.0,
    },
    "cfb": {
        "z": 400.0,
        "k": 11.0,
        "b": 10.0,
        "hfa": 55.0,
        "start_elo": 1500.0,
        "points_per_elo": 24.0,
    },
}


def is_football_league(league: str) -> bool:
    league = league.lower()
    profile = LEAGUE_PROFILES.get(league)
    return profile is not None and profile["category"] == "football"


def _league_config(league: str) -> dict[str, float]:
    league = league.lower()
    return LEAGUE_CONFIG.get(league, LEAGUE_CONFIG["nfl"])


def elo_to_prob(elo_dif: float, *, z: float) -> float:
    """nfelo spread_translation.elo_to_prob — home win probability from Elo diff."""
    if z <= 0:
        raise ValueError("z must be positive")
    return 1.0 / (math.pow(10, -elo_dif / z) + 1)


def _elo_dif_to_model_line(elo_dif: float, points_per_elo: float) -> float:
    """Approximate nfelo home spread line (negative = home favored)."""
    expected_margin = elo_dif / points_per_elo
    return -expected_margin


def calc_shift(
    margin_measure: float,
    model_line: float,
    market_line: float,
    k: float,
    b: float,
    market_resist_factor: float,
    is_home: bool,
) -> float:
    """nfelo Utilities/elo_shift.calc_shift — margin vs expectation rating change."""
    if is_home:
        model_line *= -1
        market_line *= -1

    model_error = abs(margin_measure - model_line)
    market_error = abs(margin_measure - market_line)

    if model_error < 1 or market_resist_factor == 0:
        adj_k = k
    elif model_error <= market_error:
        adj_k = k
    else:
        adj_k = k * (1 + abs(model_error - market_error) / market_resist_factor)

    pd_measure = model_error
    mult_measure = math.log(max(pd_measure, 1) + 1, b)
    shift = mult_measure * adj_k

    if model_error == 0:
        shift = 0
    elif model_line > margin_measure:
        shift *= -1
    return shift


@dataclass
class TeamElo:
    abbr: str
    rating: float
    games: int = 0


def _build_elo_ratings(
    games: list[tuple],
    league: str,
) -> dict[str, TeamElo]:
    cfg = _league_config(league)
    ratings: dict[str, TeamElo] = {}

    def get_team(abbr: str) -> TeamElo:
        key = abbr.lower()
        if key not in ratings:
            ratings[key] = TeamElo(abbr=key, rating=cfg["start_elo"])
        return ratings[key]

    for home_abbr, away_abbr, _hn, _an, home_score, away_score in games:
        home = get_team(home_abbr)
        away = get_team(away_abbr)

        pre_dif = home.rating - away.rating
        model_line = _elo_dif_to_model_line(pre_dif + cfg["hfa"], cfg["points_per_elo"])

        home_margin = home_score - away_score
        away_margin = away_score - home_score

        # No market line in ESPN-only pipeline — use model line as market proxy.
        home_shift = calc_shift(
            float(home_margin),
            model_line,
            model_line,
            cfg["k"],
            cfg["b"],
            0.0,
            True,
        )
        away_shift = calc_shift(
            float(away_margin),
            model_line,
            model_line,
            cfg["k"],
            cfg["b"],
            0.0,
            False,
        )

        home.rating += home_shift
        away.rating += away_shift
        home.games += 1
        away.games += 1

    return ratings


def build_football_model(games: list[tuple], league: str) -> dict[str, Any] | None:
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    ratings = _build_elo_ratings(games, league)
    if not ratings:
        return None

    team_game_counts = {abbr: team.games for abbr, team in ratings.items()}
    return {
        "league": league.lower(),
        "ratings": ratings,
        "team_game_counts": team_game_counts,
        "config": _league_config(league),
        "games_sampled": len(games),
    }


def extract_team_nfelo_strengths(model: dict[str, Any]) -> dict[str, float]:
    """nfelo Elo rating per team."""
    ratings: dict[str, TeamElo] = model["ratings"]
    return {abbr: team.rating for abbr, team in ratings.items()}


def predict_matchup_from_model(
    model: dict[str, Any],
    home_abbr: str,
    away_abbr: str,
) -> dict[str, float | str] | None:
    home = home_abbr.lower()
    away = away_abbr.lower()
    ratings: dict[str, TeamElo] = model["ratings"]
    if home not in ratings or away not in ratings:
        return None

    cfg = model["config"]
    elo_dif = ratings[home].rating - ratings[away].rating + cfg["hfa"]
    home_prob = elo_to_prob(elo_dif, z=cfg["z"])
    model_line = _elo_dif_to_model_line(elo_dif, cfg["points_per_elo"])

    return {
        "home_key": home,
        "away_key": away,
        "home_elo": round(ratings[home].rating, 1),
        "away_elo": round(ratings[away].rating, 1),
        "elo_differential": round(elo_dif, 1),
        "projected_spread": round(model_line, 1),
        "home_win_probability": round(home_prob * 100.0, 2),
        "away_win_probability": round((1.0 - home_prob) * 100.0, 2),
    }


def football_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    league = league.lower()
    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_LEAGUE_GAMES:
        return (
            f"Insufficient completed games ({len(games)} < {MIN_LEAGUE_GAMES}) "
            "— likely off-season or sparse schedule."
        )

    model = build_football_model(games, league)
    if not model:
        return "Could not build nfelo-style football model on available games."

    counts = model["team_game_counts"]
    home = home_abbr.lower()
    away = away_abbr.lower()
    if home not in counts or away not in counts:
        missing = [k for k in (home, away) if k not in counts]
        return f"Teams not found in football model: {', '.join(missing)}."
    if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
        return "Teams have insufficient games in the football model sample."
    return "nfelo model unavailable."


@lru_cache(maxsize=32)
def get_football_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_football_league(league):
        return None
    games = load_league_completed_games(league, cutoff_date)
    return build_football_model(games, league)


def run_football_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """Run nfelo-style Elo model for an NFL or CFB matchup."""
    context = get_football_pred_context(league, cutoff_date)
    if not context:
        return None

    prediction = predict_matchup_from_model(context, home_abbr, away_abbr)
    if not prediction:
        return None

    counts = context["team_game_counts"]
    home_games = counts.get(home_abbr.lower(), 0)
    away_games = counts.get(away_abbr.lower(), 0)
    if home_games < MIN_TEAM_GAMES or away_games < MIN_TEAM_GAMES:
        return None

    return {
        "algorithm": "NfeloElo",
        "source": "nfelo",
        "home_win_probability": prediction["home_win_probability"],
        "away_win_probability": prediction["away_win_probability"],
        "home_elo": prediction["home_elo"],
        "away_elo": prediction["away_elo"],
        "elo_differential": prediction["elo_differential"],
        "projected_spread": prediction["projected_spread"],
        "home_games": home_games,
        "away_games": away_games,
    }
