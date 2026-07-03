"""CBB win probability: Torvik efficiency + calibration stack (replaces BasketballMatrix)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from web.cbb_availability import availability_home_shift_pp
from web.cbb_calibrate import (
    CalibratorState,
    apply_cbb_calibration,
    decorrelate_from_market,
    fit_best_calibrator,
    spread_to_home_prob,
)
from web.cbb_efficiency import (
    DEFAULT_HCA,
    DEFAULT_SIGMA,
    build_fallback_efficiency,
    expected_margin_from_ratings,
    margin_to_home_win_prob,
    resolve_team_ratings,
)
from web.cbb_opening import fetch_opening_spread_proxy, opening_steam_adjustment
from web.cbb_pick_signals import build_cbb_pick_signals
from web.cbb_torvik import fetch_torvik_ratings
from web.espn_client import current_season_year
from web.live_data import _parse_cutoff
from web.season_games import (
    GameTuple,
    _event_date_iso,
    _load_league_game_map,
    load_league_completed_games,
)

MIN_LEAGUE_GAMES = 20
MIN_TEAM_GAMES = 3
MIN_CALIBRATION_GAMES = 25
WALK_FORWARD_MIN_TRAIN = 15


def is_cbb_league(league: str) -> bool:
    return league.lower() == "cbb"


def load_cbb_dated_games(cutoff_date: str) -> list[tuple[str, GameTuple]]:
    merged = _load_league_game_map("cbb", cutoff_date, for_backtest=False)
    return [
        (_event_date_iso(event_date), game_tuple)
        for event_date, game_tuple in sorted(merged.values(), key=lambda item: item[0])
    ]


def _cutoff_to_season_year(cutoff_date: str) -> int:
    cutoff = _parse_cutoff(cutoff_date)
    return current_season_year("cbb", cutoff.date() if hasattr(cutoff, "date") else cutoff)


def _raw_matchup_prob(
    home_abbr: str,
    away_abbr: str,
    torvik: dict[str, Any],
    fallback: dict[str, Any],
    *,
    hca: float = DEFAULT_HCA,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[float, float, float, float] | None:
    pair = resolve_team_ratings(home_abbr, away_abbr, torvik, fallback)
    if pair is None:
        return None
    home_ratings, away_ratings = pair
    home_score, away_score, margin = expected_margin_from_ratings(
        home_ratings, away_ratings, hca=hca
    )
    prob = margin_to_home_win_prob(margin, sigma=sigma)
    return home_score, away_score, margin, prob


def build_cbb_model(
    dated_games: list[tuple[str, GameTuple]],
    cutoff_date: str,
    season_year: int,
) -> dict[str, Any] | None:
    """
    Train walk-forward calibrator on dated games WITHOUT market features.

    Market is used only at inference via decorrelate_from_market.
    """
    games_only = [game for _date, game in dated_games]
    if len(games_only) < MIN_LEAGUE_GAMES:
        return None

    team_keys = sorted(
        {home for home, away, *_ in games_only} | {away for home, away, *_ in games_only}
    )
    if len(team_keys) < 4:
        return None

    torvik = fetch_torvik_ratings(season_year)
    fallback = build_fallback_efficiency(games_only)
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}
    for home, away, *_rest in games_only:
        if home in team_game_counts:
            team_game_counts[home] += 1
        if away in team_game_counts:
            team_game_counts[away] += 1

    raw_probs: list[float] = []
    outcomes: list[float] = []
    train_count = 0

    for idx, (_game_date, game) in enumerate(dated_games):
        home, away, _hn, _an, home_score, away_score = game
        if idx < WALK_FORWARD_MIN_TRAIN:
            continue
        result = _raw_matchup_prob(home, away, torvik, fallback)
        if result is None:
            continue
        _hs, _as, _margin, prob = result
        raw_probs.append(prob)
        if home_score > away_score:
            outcomes.append(1.0)
        elif home_score < away_score:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)
        train_count += 1

    calibrator = fit_best_calibrator(raw_probs, outcomes) if train_count >= MIN_CALIBRATION_GAMES else CalibratorState()

    return {
        "torvik": torvik,
        "fallback": fallback,
        "calibrator": calibrator,
        "team_game_counts": team_game_counts,
        "season_year": season_year,
        "cutoff_date": cutoff_date,
        "sigma": DEFAULT_SIGMA,
        "hca": DEFAULT_HCA,
    }


def predict_matchup_from_cbb_model(
    model: dict[str, Any],
    home_abbr: str,
    away_abbr: str,
) -> dict[str, float | str] | None:
    result = _raw_matchup_prob(
        home_abbr,
        away_abbr,
        model["torvik"],
        model["fallback"],
        hca=float(model.get("hca", DEFAULT_HCA)),
        sigma=float(model.get("sigma", DEFAULT_SIGMA)),
    )
    if result is None:
        return None
    home_score, away_score, margin, prob = result
    return {
        "home_key": home_abbr.lower(),
        "away_key": away_abbr.lower(),
        "predicted_home_score": home_score,
        "predicted_away_score": away_score,
        "predicted_margin": margin,
        "home_win_probability": prob,
        "param": float(model.get("sigma", DEFAULT_SIGMA)),
    }


def cbb_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    if not is_cbb_league(league):
        return "Not a CBB league."
    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_LEAGUE_GAMES:
        return (
            f"Insufficient completed games ({len(games)} < {MIN_LEAGUE_GAMES}) "
            "— likely off-season or sparse schedule."
        )
    season_year = _cutoff_to_season_year(cutoff_date)
    dated = load_cbb_dated_games(cutoff_date)
    model = build_cbb_model(dated, cutoff_date, season_year)
    if not model:
        return "Could not build CBB Torvik efficiency model."
    counts = model["team_game_counts"]
    home = home_abbr.lower()
    away = away_abbr.lower()
    if home not in counts or away not in counts:
        missing = [k for k in (home, away) if k not in counts]
        return f"Teams not found in CBB model: {', '.join(missing)}."
    if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
        return "Teams have insufficient games in the CBB sample."
    return "CBB Torvik model unavailable."


@lru_cache(maxsize=32)
def get_cbb_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_cbb_league(league):
        return None
    dated = load_cbb_dated_games(cutoff_date)
    season_year = _cutoff_to_season_year(cutoff_date)
    return build_cbb_model(dated, cutoff_date, season_year)


def run_cbb_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
    home_name: str | None = None,
    away_name: str | None = None,
    market_spread: float | None = None,
    home_ml: int | None = None,
    away_ml: int | None = None,
) -> dict[str, Any] | None:
    """Run Torvik efficiency + calibrated CBB model for a matchup."""
    _ = home_name, away_name
    context = get_cbb_pred_context(league, cutoff_date)
    if not context:
        return None

    prediction = predict_matchup_from_cbb_model(context, home_abbr, away_abbr)
    if not prediction:
        return None

    counts = context["team_game_counts"]
    home_games = counts.get(home_abbr.lower(), 0)
    away_games = counts.get(away_abbr.lower(), 0)
    if home_games < MIN_TEAM_GAMES or away_games < MIN_TEAM_GAMES:
        return None

    raw_prob = float(prediction["home_win_probability"])
    margin = float(prediction["predicted_margin"])

    avail_shift = availability_home_shift_pp(
        league,
        home_espn_id,
        away_espn_id,
        cutoff_date,
        home_games,
        away_games,
    )
    raw_prob = min(max(raw_prob + avail_shift, 1.0), 99.0)

    opening_spread = fetch_opening_spread_proxy(
        league,
        cutoff_date,
        home_abbr.lower(),
        away_abbr.lower(),
    )
    margin, steam_meta = opening_steam_adjustment(
        margin,
        market_spread,
        opening_spread=opening_spread,
    )
    raw_prob = margin_to_home_win_prob(margin, sigma=float(context.get("sigma", DEFAULT_SIGMA)))

    if market_spread is not None:
        market_prob = spread_to_home_prob(float(market_spread))
        raw_prob = decorrelate_from_market(raw_prob / 100.0, market_prob / 100.0) * 100.0

    calibrated_prob = apply_cbb_calibration(raw_prob, context.get("calibrator"))

    pick_signals = build_cbb_pick_signals(
        model_margin=margin,
        market_spread=market_spread,
        home_ml=home_ml,
        away_ml=away_ml,
        steam_meta=steam_meta,
    )

    return {
        "algorithm": "CBBTorvik",
        "source": "torvik-efficiency-calibrated",
        "home_win_probability": calibrated_prob,
        "predicted_home_score": prediction["predicted_home_score"],
        "predicted_away_score": prediction["predicted_away_score"],
        "predicted_margin": round(margin, 2),
        "param": prediction["param"],
        "home_games": home_games,
        "away_games": away_games,
        "raw_home_win_probability": round(raw_prob, 2),
        "availability_shift_pp": avail_shift,
        "cbb_pick_signals": pick_signals,
        "opening_steam": steam_meta,
    }
