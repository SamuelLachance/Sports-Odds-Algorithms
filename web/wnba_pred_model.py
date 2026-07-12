"""WNBA win probability: 538 Elo + efficiency + XGBoost (ariusmak pattern)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from web.cbb_calibrate import (
    CalibratorState,
    apply_cbb_calibration,
    decorrelate_from_market,
    fit_best_calibrator,
    spread_to_home_prob,
)
from web.espn_client import current_season_year
from web.live_data import _parse_cutoff
from web.season_games import (
    GameTuple,
    _event_date_iso,
    _load_league_game_map,
    load_league_completed_games,
)
from web.wnba_availability import availability_home_shift_pp
from web.wnba_efficiency import (
    DEFAULT_HCA,
    DEFAULT_SIGMA,
    EfficiencyState,
    margin_to_home_win_prob,
    update_efficiency_after_game,
)
from web.wnba_elo import EloState, update_elo_after_game
from web.wnba_features import (
    FEATURE_NAMES,
    build_matchup_features,
    features_to_vector,
    raw_prob_from_elo_and_efficiency,
)
from web.wnba_opening import fetch_opening_spread_proxy, opening_steam_adjustment
from web.wnba_pick_signals import build_wnba_pick_signals

MIN_LEAGUE_GAMES = 20
MIN_TEAM_GAMES = 3
MIN_CALIBRATION_GAMES = 20
WALK_FORWARD_MIN_TRAIN = 12
XGB_MIN_ROWS = 40


def is_wnba_league(league: str) -> bool:
    return league.lower() == "wnba"


def load_wnba_dated_games(cutoff_date: str) -> list[tuple[str, GameTuple]]:
    merged = _load_league_game_map("wnba", cutoff_date, for_backtest=False)
    return [
        (_event_date_iso(event_date), game_tuple)
        for event_date, game_tuple in sorted(merged.values(), key=lambda item: item[0])
    ]


def _season_from_date(iso_date: str) -> int:
    try:
        return int(iso_date[:4])
    except (TypeError, ValueError):
        return 2025


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
        learning_rate=0.08,
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


def build_wnba_model(
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

    elo = EloState()
    efficiency = EfficiencyState()
    last_game_date: dict[str, str] = {}
    team_game_counts: dict[str, int] = {key: 0 for key in team_keys}

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    raw_probs: list[float] = []
    outcomes: list[float] = []

    for idx, (game_date, game) in enumerate(dated_games):
        home, away, _hn, _an, home_score, away_score = game
        season = _season_from_date(game_date)
        elo.regress_season(season)

        if idx >= WALK_FORWARD_MIN_TRAIN:
            feats = build_matchup_features(
                home,
                away,
                elo=elo,
                efficiency=efficiency,
                game_date=game_date,
                last_game_date=last_game_date,
                team_game_counts=team_game_counts,
            )
            _hs, _as, margin, raw_prob = raw_prob_from_elo_and_efficiency(
                home,
                away,
                elo=elo,
                efficiency=efficiency,
                sigma=DEFAULT_SIGMA,
                hca=DEFAULT_HCA,
            )
            x_rows.append(features_to_vector(feats))
            raw_probs.append(raw_prob)
            if home_score > away_score:
                y_rows.append(1.0)
                outcomes.append(1.0)
            elif home_score < away_score:
                y_rows.append(0.0)
                outcomes.append(0.0)
            else:
                y_rows.append(0.5)
                outcomes.append(0.5)

        update_elo_after_game(elo, home, away, home_score, away_score)
        update_efficiency_after_game(efficiency, home, away, home_score, away_score)
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
        "elo": elo,
        "efficiency": efficiency,
        "last_game_date": last_game_date,
        "team_game_counts": team_game_counts,
        "xgb_model": xgb_model,
        "calibrator": calibrator,
        "sigma": DEFAULT_SIGMA,
        "hca": DEFAULT_HCA,
        "cutoff_date": cutoff_date,
    }


def predict_matchup_from_wnba_model(
    model: dict[str, Any],
    home_abbr: str,
    away_abbr: str,
    *,
    game_date: str | None = None,
) -> dict[str, float | str] | None:
    home = home_abbr.lower()
    away = away_abbr.lower()
    elo: EloState = model["elo"]
    efficiency: EfficiencyState = model["efficiency"]
    last_game_date: dict[str, str] = model.get("last_game_date") or {}
    team_game_counts: dict[str, int] = model.get("team_game_counts") or {}
    cutoff = game_date or str(model.get("cutoff_date") or "")

    if home not in team_game_counts or away not in team_game_counts:
        return None

    feats = build_matchup_features(
        home,
        away,
        elo=elo,
        efficiency=efficiency,
        game_date=cutoff,
        last_game_date=last_game_date,
        team_game_counts=team_game_counts,
    )
    home_score, away_score, margin, raw_prob = raw_prob_from_elo_and_efficiency(
        home,
        away,
        elo=elo,
        efficiency=efficiency,
        sigma=float(model.get("sigma", DEFAULT_SIGMA)),
        hca=float(model.get("hca", DEFAULT_HCA)),
    )

    xgb_model = model.get("xgb_model")
    if xgb_model is not None:
        try:
            xgb_p = float(xgb_model.predict_proba(features_to_vector(feats).reshape(1, -1))[0, 1])
            raw_prob = round(0.4 * raw_prob + 0.6 * xgb_p * 100.0, 2)
            margin = float(model.get("sigma", DEFAULT_SIGMA)) * np.log(
                max(raw_prob / 100.0, 1e-6) / max(1.0 - raw_prob / 100.0, 1e-6)
            )
        except (ValueError, IndexError, AttributeError):
            pass

    return {
        "home_key": home,
        "away_key": away,
        "predicted_home_score": home_score,
        "predicted_away_score": away_score,
        "predicted_margin": round(margin, 2),
        "home_win_probability": raw_prob,
        "param": float(model.get("sigma", DEFAULT_SIGMA)),
    }


def wnba_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    if not is_wnba_league(league):
        return "Not a WNBA league."
    games = load_league_completed_games(league, cutoff_date)
    if len(games) < MIN_LEAGUE_GAMES:
        return (
            f"Insufficient completed games ({len(games)} < {MIN_LEAGUE_GAMES}) "
            "— likely off-season."
        )
    dated = load_wnba_dated_games(cutoff_date)
    model = build_wnba_model(dated, cutoff_date)
    if not model:
        return "Could not build WNBA Elo+XGB model."
    counts = model["team_game_counts"]
    home = home_abbr.lower()
    away = away_abbr.lower()
    if home not in counts or away not in counts:
        missing = [k for k in (home, away) if k not in counts]
        return f"Teams not found in WNBA model: {', '.join(missing)}."
    if counts.get(home, 0) < MIN_TEAM_GAMES or counts.get(away, 0) < MIN_TEAM_GAMES:
        return "Teams have insufficient games in the WNBA sample."
    return "WNBA model unavailable."


@lru_cache(maxsize=32)
def get_wnba_pred_context(league: str, cutoff_date: str) -> dict[str, Any] | None:
    league = league.lower()
    if not is_wnba_league(league):
        return None
    dated = load_wnba_dated_games(cutoff_date)
    return build_wnba_model(dated, cutoff_date)


def run_wnba_pred_model(
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
    _ = home_name, away_name
    context = get_wnba_pred_context(league, cutoff_date)
    if not context:
        return None

    prediction = predict_matchup_from_wnba_model(
        context, home_abbr, away_abbr, game_date=cutoff_date
    )
    if not prediction:
        return None

    counts = context["team_game_counts"]
    home_games = counts.get(home_abbr.lower(), 0)
    away_games = counts.get(away_abbr.lower(), 0)
    if home_games < MIN_TEAM_GAMES or away_games < MIN_TEAM_GAMES:
        return None

    margin = float(prediction["predicted_margin"])

    avail_shift = availability_home_shift_pp(
        league,
        home_espn_id,
        away_espn_id,
        cutoff_date,
        home_games,
        away_games,
    )

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
    # Rebuild from steam-adjusted margin, then apply availability (do not wipe it).
    raw_prob = margin_to_home_win_prob(margin, sigma=float(context.get("sigma", DEFAULT_SIGMA)))
    raw_prob = min(max(raw_prob + avail_shift, 1.0), 99.0)

    if market_spread is not None:
        market_prob = spread_to_home_prob(float(market_spread), sigma=float(context["sigma"]))
        raw_prob = decorrelate_from_market(raw_prob / 100.0, market_prob / 100.0) * 100.0

    calibrated_prob = apply_cbb_calibration(raw_prob, context.get("calibrator"))

    pick_signals = build_wnba_pick_signals(
        model_margin=margin,
        market_spread=market_spread,
        home_ml=home_ml,
        away_ml=away_ml,
        home_games=home_games,
        away_games=away_games,
        steam_meta=steam_meta,
    )

    return {
        "algorithm": "WNBAEloXGB",
        "source": "elo-efficiency-xgb-calibrated",
        "market_decorrelated": market_spread is not None,
        "home_win_probability": calibrated_prob,
        "predicted_home_score": prediction["predicted_home_score"],
        "predicted_away_score": prediction["predicted_away_score"],
        "predicted_margin": round(margin, 2),
        "param": prediction["param"],
        "home_games": home_games,
        "away_games": away_games,
        "raw_home_win_probability": round(raw_prob, 2),
        "availability_shift_pp": avail_shift,
        "wnba_pick_signals": pick_signals,
        "opening_steam": steam_meta,
    }
