"""MLB win probability: run-efficiency + Monte Carlo + XGBoost (Path A / mlb-bet-engine pattern)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from web.bet_advisor import devig_two_way_probs
from web.cbb_calibrate import (
    CalibratorState,
    apply_cbb_calibration,
    decorrelate_from_market,
    fit_best_calibrator,
    spread_to_home_prob,
)
from web.market_decorrelation import decorrelate_binary
from web.mlb_efficiency import (
    DEFAULT_HCA_RUNS,
    DEFAULT_SIGMA,
    RunsEfficiencyState,
    margin_to_home_win_prob,
    update_runs_efficiency_after_game,
)
from web.mlb_features import (
    FEATURE_NAMES,
    build_matchup_features,
    features_to_vector,
    raw_prob_from_runs_sim,
)
from web.mlb_opening import fetch_opening_spread_proxy, opening_steam_adjustment
from web.mlb_pick_signals import build_mlb_pick_signals
from web.mlb_pitcher_edge import pitcher_matchup_margin
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
XGB_MIN_ROWS = 50


def is_mlb_league(league: str) -> bool:
    return league.lower() == "mlb"


def load_mlb_dated_games(cutoff_date: str) -> list[tuple[str, GameTuple]]:
    merged = _load_league_game_map("mlb", cutoff_date, for_backtest=False)
    return [
        (_event_date_iso(event_date), game_tuple)
        for event_date, game_tuple in sorted(merged.values(), key=lambda item: item[0])
    ]


def _train_xgb_classifier(
    x_rows: list[np.ndarray],
    y_rows: list[float],
) -> Any | None:
    if len(x_rows) < XGB_MIN_ROWS:
        return None
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    x_mat = np.vstack(x_rows)
    y = np.array(y_rows, dtype=int)
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    try:
        model.fit(x_mat, y)
        return model
    except ValueError:
        return None


def build_mlb_model(
    dated_games: list[tuple[str, GameTuple]],
    cutoff_date: str,
) -> dict[str, Any] | None:
    games_only = [game for _date, game in dated_games]
    if len(games_only) < MIN_LEAGUE_GAMES:
        return None

    team_keys = sorted(
        {home for home, away, *_ in games_only} | {away for home, away, *_ in games_only}
    )
    if len(team_keys) < 4:
        return None

    efficiency = RunsEfficiencyState()
    last_game_date: dict[str, str] = {}
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    raw_probs: list[float] = []
    outcomes: list[float] = []

    for idx, (game_date, game) in enumerate(dated_games):
        home, away, _hn, _an, home_score, away_score = game

        if idx >= WALK_FORWARD_MIN_TRAIN:
            feats = build_matchup_features(
                home,
                away,
                efficiency=efficiency,
                game_date=game_date,
                last_game_date=last_game_date,
                team_game_counts=team_game_counts,
                pitcher_margin=0.0,
                hca=DEFAULT_HCA_RUNS,
            )
            _hm, _am, margin, raw_prob = raw_prob_from_runs_sim(
                home,
                away,
                efficiency=efficiency,
                pitcher_margin=0.0,
                hca=DEFAULT_HCA_RUNS,
                seed=idx,
            )
            raw_probs.append(raw_prob)
            if home_score > away_score:
                x_rows.append(features_to_vector(feats))
                y_rows.append(1.0)
                outcomes.append(1.0)
            elif home_score < away_score:
                x_rows.append(features_to_vector(feats))
                y_rows.append(0.0)
                outcomes.append(0.0)
            else:
                # Soft 0.5 is fine for the calibrator; XGB uses dtype=int so
                # 0.5→0 and would train ties as away wins.
                outcomes.append(0.5)

        update_runs_efficiency_after_game(efficiency, home, away, home_score, away_score)
        last_game_date[home] = game_date
        last_game_date[away] = game_date
        if home in team_game_counts:
            team_game_counts[home] += 1
        if away in team_game_counts:
            team_game_counts[away] += 1

    xgb_model = _train_xgb_classifier(x_rows, y_rows)
    calibrator = (
        fit_best_calibrator(raw_probs, outcomes)
        if len(raw_probs) >= MIN_CALIBRATION_GAMES
        else CalibratorState()
    )

    return {
        "efficiency": efficiency,
        "last_game_date": last_game_date,
        "team_game_counts": team_game_counts,
        "xgb_model": xgb_model,
        "calibrator": calibrator,
        "sigma": DEFAULT_SIGMA,
        "hca": DEFAULT_HCA_RUNS,
        "cutoff_date": cutoff_date,
    }


def predict_matchup_from_mlb_model(
    model: dict[str, Any],
    home_abbr: str,
    away_abbr: str,
    *,
    game_date: str | None = None,
    pitcher_margin: float = 0.0,
) -> dict[str, float | str] | None:
    home = home_abbr.lower()
    away = away_abbr.lower()
    efficiency: RunsEfficiencyState = model["efficiency"]
    last_game_date: dict[str, str] = model.get("last_game_date") or {}
    team_game_counts: dict[str, int] = model.get("team_game_counts") or {}
    cutoff = game_date or str(model.get("cutoff_date") or "")

    if home not in team_game_counts or away not in team_game_counts:
        return None

    hca = float(model.get("hca", DEFAULT_HCA_RUNS))
    feats = build_matchup_features(
        home,
        away,
        efficiency=efficiency,
        game_date=cutoff,
        last_game_date=last_game_date,
        team_game_counts=team_game_counts,
        pitcher_margin=pitcher_margin,
        hca=hca,
    )
    home_mu, away_mu, margin, raw_prob = raw_prob_from_runs_sim(
        home,
        away,
        efficiency=efficiency,
        pitcher_margin=pitcher_margin,
        hca=hca,
        seed=hash((home, away, cutoff)) % 10_000,
    )

    xgb_model = model.get("xgb_model")
    if xgb_model is not None:
        try:
            xgb_p = float(xgb_model.predict_proba(features_to_vector(feats).reshape(1, -1))[0, 1])
            raw_prob = round(0.45 * raw_prob + 0.55 * xgb_p * 100.0, 2)
            # Keep sim margin aligned with run totals (do not invent a logit
            # margin that disagrees with predicted_home_runs - predicted_away_runs).
        except (ValueError, IndexError, AttributeError):
            pass

    return {
        "home_key": home,
        "away_key": away,
        "predicted_home_runs": home_mu,
        "predicted_away_runs": away_mu,
        "predicted_margin": round(margin, 2),
        "home_win_probability": raw_prob,
        "param": float(model.get("sigma", DEFAULT_SIGMA)),
    }


def mlb_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    if not is_mlb_league(league):
        return "Not an MLB league."
    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_LEAGUE_GAMES:
        return (
            f"Insufficient completed games ({len(games)} < {MIN_LEAGUE_GAMES}) "
            "— likely off-season."
        )
    dated = load_mlb_dated_games(cutoff_date)
    model = build_mlb_model(dated, cutoff_date)
    if not model:
        return "Could not build MLB RunCast model."
    counts = model["team_game_counts"]
    home = home_abbr.lower()
    away = away_abbr.lower()
    if home not in counts or away not in counts:
        missing = [k for k in (home, away) if k not in counts]
        return f"Teams not found in MLB model: {', '.join(missing)}."
    if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
        return "Teams have insufficient games in the MLB sample."
    return "MLB model unavailable."


@lru_cache(maxsize=32)
def get_mlb_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_mlb_league(league):
        return None
    dated = load_mlb_dated_games(cutoff_date)
    return build_mlb_model(dated, cutoff_date)


def _cutoff_to_iso(cutoff_date: str) -> str | None:
    parts = cutoff_date.split("-")
    if len(parts) != 3:
        return None
    month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
    return f"{year:04d}-{month:02d}-{day:02d}"


def _run_mlb_v2(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    market_spread: float | None = None,
    home_ml: int | None = None,
    away_ml: int | None = None,
    kickoff_iso: str | None = None,
    game_number: int | None = None,
) -> dict[str, Any] | None:
    """Persisted MLB v2 (statsapi feature engine + gradient boost ensemble)."""
    if not is_mlb_league(league):
        return None
    game_date = _cutoff_to_iso(cutoff_date)
    if not game_date:
        return None
    try:
        from web.mlb_v2.live import predict_matchup_v2

        v2 = predict_matchup_v2(
            game_date,
            home_abbr,
            away_abbr,
            kickoff_iso=kickoff_iso,
            game_number=game_number,
        )
    except Exception:  # noqa: BLE001 - any v2 failure falls back to legacy RunCast
        return None
    if not v2:
        return None

    calibrated_prob = float(v2["home_win_probability"])  # 0-100, calibrated
    # Keep a true pick'em margin of 0.0; only invent from win% when missing.
    raw_margin = v2.get("predicted_margin")
    if raw_margin is None:
        margin = DEFAULT_SIGMA * np.log(
            max(calibrated_prob / 100.0, 1e-6)
            / max(1.0 - calibrated_prob / 100.0, 1e-6)
        )
    else:
        margin = float(raw_margin)

    home_prob = calibrated_prob
    market_decorrelated = False
    market_decorrelation_source: str | None = None
    market_home = None
    if home_ml is not None and away_ml is not None:
        _away_mkt, market_home = devig_two_way_probs(away_ml, home_ml)
    if market_home is not None:
        home_prob = decorrelate_binary(calibrated_prob, market_home)
        market_decorrelated = True
        market_decorrelation_source = "moneyline"
    elif market_spread is not None and not isinstance(market_spread, bool):
        # Invalid/garbage ML must not block spread-based decorrelation.
        # bool False→0.0 must not invent a pick'em market.
        market_prob = spread_to_home_prob(float(market_spread), sigma=DEFAULT_SIGMA)
        home_prob = decorrelate_from_market(
            calibrated_prob / 100.0, market_prob / 100.0
        ) * 100.0
        market_decorrelated = True
        market_decorrelation_source = "spread"

    home_games = int(v2.get("home_games") or 0)
    away_games = int(v2.get("away_games") or 0)
    pick_signals = build_mlb_pick_signals(
        model_margin=margin,
        market_spread=market_spread,
        home_ml=home_ml,
        away_ml=away_ml,
        home_games=home_games,
        away_games=away_games,
        steam_meta=None,
    )

    return {
        "algorithm": "MLBGradientBoost",
        "model_version": "v2",
        "source": "statsapi-xgb-lr-isotonic",
        "market_decorrelated": market_decorrelated,
        "market_decorrelation_source": market_decorrelation_source,
        "pre_decorrelation_home_win_probability": round(calibrated_prob, 2),
        "home_win_probability": round(home_prob, 2),
        "predicted_home_runs": v2.get("predicted_home_runs"),
        "predicted_away_runs": v2.get("predicted_away_runs"),
        "predicted_margin": round(margin, 2),
        "param": DEFAULT_SIGMA,
        "home_games": home_games,
        "away_games": away_games,
        "raw_home_win_probability": round(calibrated_prob, 2),
        "pitcher_margin": round(
            (float(v2.get("away_sp_fip") or 4.2) - float(v2.get("home_sp_fip") or 4.2)) * 0.35,
            3,
        ),
        "mlb_pick_signals": pick_signals,
        "opening_steam": None,
        "home_probable_pitcher": v2.get("home_probable_pitcher"),
        "away_probable_pitcher": v2.get("away_probable_pitcher"),
        "home_sp_fip": v2.get("home_sp_fip"),
        "away_sp_fip": v2.get("away_sp_fip"),
        "home_elo": v2.get("home_elo"),
        "away_elo": v2.get("away_elo"),
        "park_factor": v2.get("park_factor"),
    }


def run_mlb_pred_model(
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
    kickoff_iso: str | None = None,
    game_number: int | None = None,
) -> dict[str, Any] | None:
    _ = home_espn_id, away_espn_id, home_name, away_name

    v2_payload = _run_mlb_v2(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        market_spread=market_spread,
        home_ml=home_ml,
        away_ml=away_ml,
        kickoff_iso=kickoff_iso,
        game_number=game_number,
    )
    if v2_payload is not None:
        return v2_payload

    context = get_mlb_pred_context(league, cutoff_date)
    if not context:
        return None

    game_date = _cutoff_to_iso(cutoff_date)
    pitcher_margin = 0.0
    if game_date:
        edge = pitcher_matchup_margin(
            home_abbr,
            away_abbr,
            game_date=game_date,
            game_number=game_number,
        )
        if edge is not None:
            pitcher_margin = float(edge)

    prediction = predict_matchup_from_mlb_model(
        context,
        home_abbr,
        away_abbr,
        game_date=game_date,
        pitcher_margin=pitcher_margin,
    )
    if not prediction:
        return None

    counts = context["team_game_counts"]
    home_games = counts.get(home_abbr.lower(), 0)
    away_games = counts.get(away_abbr.lower(), 0)
    if home_games < MIN_TEAM_GAMES or away_games < MIN_TEAM_GAMES:
        return None

    margin = float(prediction["predicted_margin"])

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
    model_prob = margin_to_home_win_prob(margin, sigma=float(context.get("sigma", DEFAULT_SIGMA)))
    calibrated_prob = apply_cbb_calibration(model_prob, context.get("calibrator"))

    home_prob = calibrated_prob
    market_decorrelated = False
    market_decorrelation_source: str | None = None
    market_home = None
    if home_ml is not None and away_ml is not None:
        _away_mkt, market_home = devig_two_way_probs(away_ml, home_ml)
    if market_home is not None:
        home_prob = decorrelate_binary(calibrated_prob, market_home)
        market_decorrelated = True
        market_decorrelation_source = "moneyline"
    elif market_spread is not None and not isinstance(market_spread, bool):
        # Invalid/garbage ML must not block spread-based decorrelation.
        # bool False→0.0 must not invent a pick'em market.
        market_prob = spread_to_home_prob(float(market_spread), sigma=float(context["sigma"]))
        home_prob = decorrelate_from_market(
            calibrated_prob / 100.0, market_prob / 100.0
        ) * 100.0
        market_decorrelated = True
        market_decorrelation_source = "spread"

    pick_signals = build_mlb_pick_signals(
        model_margin=margin,
        market_spread=market_spread,
        home_ml=home_ml,
        away_ml=away_ml,
        home_games=home_games,
        away_games=away_games,
        steam_meta=steam_meta,
    )

    return {
        "algorithm": "MLBRunCast",
        "source": "run-sim-xgb-calibrated",
        "market_decorrelated": market_decorrelated,
        "market_decorrelation_source": market_decorrelation_source,
        "pre_decorrelation_home_win_probability": round(calibrated_prob, 2),
        "home_win_probability": round(home_prob, 2),
        "predicted_home_runs": prediction["predicted_home_runs"],
        "predicted_away_runs": prediction["predicted_away_runs"],
        "predicted_margin": round(margin, 2),
        "param": prediction["param"],
        "home_games": home_games,
        "away_games": away_games,
        "raw_home_win_probability": round(model_prob, 2),
        "pitcher_margin": round(pitcher_margin, 3),
        "mlb_pick_signals": pick_signals,
        "opening_steam": steam_meta,
    }
