"""Blend legacy Algo_V2 with Sports-pred power ratings into a unified signal."""

from __future__ import annotations

from typing import Any

from web.baseball_pred_model import (
    baseball_unavailable_reason,
    is_baseball_league,
    run_baseball_pred_model,
)
from web.basketball_pred_model import (
    basketball_unavailable_reason,
    is_basketball_league,
    run_basketball_pred_model,
)
from web.cbb_pred_model import is_cbb_league, run_cbb_pred_model
from web.mlb_pred_model import is_mlb_league, mlb_unavailable_reason, run_mlb_pred_model
from web.football_pred_model import (
    football_unavailable_reason,
    is_football_league,
    run_football_pred_model,
)
from web.hockey_pred_model import is_hockey_league, run_hockey_pred_model
from web.bet_advisor import (
    _odds_edge,
    model_home_margin,
    model_moneylines,
    spread_odds_edge,
    spread_point_edge,
)
from web.league_profiles import (
    DEFAULT_SPREAD_JUICE,
    MIN_RECOMMENDED_EDGE,
    is_soccer_league,
    uses_spread_bets,
)
from web.live_data import resolve_team
from web.power_model import PowerTeam, predict_matchup
from web.season_games import get_league_power_context, power_unavailable_reason
from web.soccer_blend import finalize_soccer_path_a, threeway_probs_to_total_score
from web.db_rating_model import apply_db_rating_blend
from web.soccer_pred_model import run_soccer_pred_model, soccer_unavailable_reason
from web.sports_meta_model import (
    apply_binary_calibration_to_blend,
    get_sports_meta_config,
    stack_binary_blend_layers,
)

LEGACY_BLEND_WEIGHT = 0.5
POWER_BLEND_WEIGHT = 0.5
THREE_LAYER_WEIGHT = 1.0 / 3.0


def _apply_context_layer(
    result: dict[str, Any],
    *,
    league: str,
    home_name: str | None,
    away_name: str | None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
    headlines: list[str] | None = None,
) -> dict[str, Any]:
    """Fourth-layer context (FLB / steam / news) after DB rating + calibration."""
    from web.context_signals import apply_context_to_blend

    market: dict[str, Any] = {}
    if home_moneyline is not None:
        market["home_moneyline"] = home_moneyline
    if away_moneyline is not None:
        market["away_moneyline"] = away_moneyline
    home_names = [home_name] if home_name else []
    away_names = [away_name] if away_name else []
    return apply_context_to_blend(
        result,
        market=market,
        headlines=headlines,
        home_names=home_names,
        away_names=away_names,
        league=league,
    )


def _with_db_rating_layer(
    result: dict[str, Any],
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_slug: str | None,
    away_slug: str | None,
    home_name: str | None,
    away_name: str | None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
    headlines: list[str] | None = None,
) -> dict[str, Any]:
    if not is_soccer_league(league) and home_espn_id and away_espn_id:
        from web.availability_signals import (
            availability_home_prob_shift,
            fetch_availability_snapshot,
        )

        snapshot = fetch_availability_snapshot(
            league,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
        )
        shift = availability_home_prob_shift(snapshot)
        if abs(shift) >= 0.05:
            home_prob = layer_home_win_probability(result)
            if home_prob is not None:
                adjusted = min(max(home_prob + shift, 5.0), 95.0)
                total, win_prob = home_win_prob_to_total_score(adjusted)
                result = dict(result)
                result["availability"] = {
                    "shift_pp": shift,
                    "home_injuries": snapshot.home_injuries,
                    "away_injuries": snapshot.away_injuries,
                    "sources": snapshot.sources or [],
                }
                result["win_probability"] = round(win_prob, 2)
                result["total_score"] = round(total, 2)
                result["favorite_side"] = "home" if total <= 0 else "away"
                if result.get("blended_home_win_probability") is not None:
                    result["blended_home_win_probability"] = round(adjusted, 2)
    if home_slug and away_slug:
        result = apply_db_rating_blend(
            result,
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_slug=home_slug,
            away_slug=away_slug,
            home_name=home_name,
            away_name=away_name,
        )
    if is_soccer_league(league):
        return _apply_context_layer(
            result,
            league=league,
            home_name=home_name,
            away_name=away_name,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
            headlines=headlines,
        )
    result = apply_binary_calibration_to_blend(result, league)
    result = _attach_home_spread_margin(result, league)
    return _apply_context_layer(
        result,
        league=league,
        home_name=home_name,
        away_name=away_name,
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
        headlines=headlines,
    )


def total_score_to_home_win_prob(total_score: float) -> float:
    """Convert Algo_V2 total_score to home win probability (0–100)."""
    if abs(total_score) < 1e-9:
        return 50.0
    if total_score < 0:
        return abs(total_score)
    return 100.0 - abs(total_score)


def layer_home_win_probability(layer: dict[str, Any]) -> float | None:
    """Resolve a layer payload to home win probability (0–100)."""
    if layer.get("home_win_probability") is not None:
        return float(layer["home_win_probability"])
    if layer.get("total_score") is not None:
        return total_score_to_home_win_prob(float(layer["total_score"]))
    if layer.get("win_probability") is not None and layer.get("favorite_side"):
        win_prob = float(layer["win_probability"])
        if layer["favorite_side"] == "home":
            return win_prob
        return 100.0 - win_prob
    return None


def home_win_prob_to_total_score(home_win_prob: float) -> tuple[float, float]:
    """Convert home win probability to (total_score, win_probability)."""
    if abs(home_win_prob - 50.0) < 1e-9:
        return 0.0, 50.0
    if home_win_prob > 50.0:
        return -home_win_prob, home_win_prob
    away_prob = 100.0 - home_win_prob
    return away_prob, away_prob


def _find_team_key(
    teams: dict[str, PowerTeam],
    league: str,
    abbr: str,
    display_name: str | None = None,
) -> str | None:
    """Resolve registry/ESPN abbreviations to power-rating team keys."""
    candidates: list[str] = []
    resolved = resolve_team(league, abbr, display_name)
    if resolved:
        candidates.append(resolved[0].lower())
    candidates.append(abbr.lower())

    for key in candidates:
        if key in teams:
            return key

    if display_name:
        target = display_name.lower()
        for key, team in teams.items():
            if team.name.lower() == target:
                return key

    return None


def _sport_pred_unavailable_reason(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> str:
    if is_basketball_league(league):
        return basketball_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    if is_mlb_league(league):
        return mlb_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    if is_baseball_league(league):
        return baseball_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    if is_hockey_league(league):
        return "Algo V1 unavailable — insufficient season data for factor inputs."
    if is_football_league(league):
        return football_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    if is_soccer_league(league):
        return soccer_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    return "Sport-specific model unavailable."


def _run_nba_ml_layer(
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    market_spread: float | None = None,
    home_ml: int | None = None,
    away_ml: int | None = None,
) -> dict[str, Any] | None:
    """NBA XGBoost layer (opt-in via NBA_ML_ENABLED). Safe no-op otherwise."""
    from web.nba_ml.config import nba_ml_enabled

    if not nba_ml_enabled():
        return None
    try:
        from web.nba_ml.predict import predict_nba

        return predict_nba(
            home_abbr,
            away_abbr,
            cutoff_date=cutoff_date,
            market_spread=market_spread,
            home_ml=home_ml,
            away_ml=away_ml,
        )
    except Exception:  # noqa: BLE001 - never break the blend on ML failure
        return None


def _run_sport_pred_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_name: str | None = None,
    away_name: str | None = None,
    market_spread: float | None = None,
    home_ml: int | None = None,
    away_ml: int | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Return (payload_key, payload) for sport-specific third layer."""
    if is_basketball_league(league):
        payload = run_basketball_pred_model(league, cutoff_date, home_abbr, away_abbr)
        return ("basketball_pred", payload) if payload else (None, None)
    if is_mlb_league(league):
        payload = run_mlb_pred_model(
            league,
            cutoff_date,
            home_abbr,
            away_abbr,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
            home_name=home_name,
            away_name=away_name,
            market_spread=market_spread,
            home_ml=home_ml,
            away_ml=away_ml,
        )
        return ("baseball_pred", payload) if payload else (None, None)
    if is_baseball_league(league):
        payload = run_baseball_pred_model(league, cutoff_date, home_abbr, away_abbr)
        return ("baseball_pred", payload) if payload else (None, None)
    if is_hockey_league(league):
        return None, None
    if is_football_league(league):
        payload = run_football_pred_model(league, cutoff_date, home_abbr, away_abbr)
        return ("football_pred", payload) if payload else (None, None)
    if is_soccer_league(league):
        payload = run_soccer_pred_model(
            league,
            cutoff_date,
            home_abbr,
            away_abbr,
            home_name=home_name,
            away_name=away_name,
        )
        return ("soccer_pred", payload) if payload else (None, None)
    return None, None


def _uses_three_layer_blend(league: str) -> bool:
    if (
        is_hockey_league(league)
        or is_basketball_league(league)
        or is_mlb_league(league)
        or is_soccer_league(league)
    ):
        return False
    return is_baseball_league(league) or is_football_league(league)


def requires_three_layer_agreement(league: str) -> bool:
    """Leagues that require unanimous agreement across all three model layers."""
    return _uses_three_layer_blend(league)


def _layer_binary_total_score(layer: dict[str, Any]) -> float | None:
    if layer.get("total_score") is not None:
        return float(layer["total_score"])
    if layer.get("home_win_probability") is not None:
        total, _ = home_win_prob_to_total_score(float(layer["home_win_probability"]))
        return total
    return None


def _layer_side_win_probs(total_score: float) -> tuple[float, float]:
    if abs(total_score) < 1e-9:
        return 50.0, 50.0
    win_prob = abs(total_score)
    home_is_favorite = total_score <= 0
    home_prob = win_prob if home_is_favorite else 100.0 - win_prob
    away_prob = 100.0 - home_prob
    return away_prob, home_prob


def _layer_home_margin(layer: dict[str, Any], league: str) -> float | None:
    """Home margin in spread convention (negative = home favored)."""
    if layer.get("predicted_margin") is not None:
        # Sport models use points margin (positive = home favored).
        return -float(layer["predicted_margin"])
    if layer.get("projected_spread") is not None:
        # nfelo projected_spread already uses book sign (negative = home favored).
        return float(layer["projected_spread"])
    total = _layer_binary_total_score(layer)
    if total is not None:
        return model_home_margin(total, league)
    return None


def _normalize_spread_margin_sign(margin: float, favorite_side: str | None) -> float:
    """Spread convention: negative = home favored, positive = away favored."""
    if favorite_side == "home" and margin > 0:
        return -margin
    if favorite_side == "away" and margin < 0:
        return -margin
    return margin


def blended_home_spread_margin(blended: dict[str, Any], league: str) -> float:
    """Home margin for spread picks, aligned with unified total_score / win%."""
    favorite_side = blended.get("favorite_side")
    cached = blended.get("home_spread_margin")
    if cached is not None:
        if blended.get("blend_mode") in {"wnba_v2", "nba_v2", "cbb_v2", "cfb_v2", "nfl_v2"}:
            # Dedicated margin head was backtested as-is; never flip its sign
            # to match the win-probability favorite.
            return float(cached)
        return _normalize_spread_margin_sign(float(cached), favorite_side)
    total = blended.get("total_score")
    if total is not None:
        return model_home_margin(float(total), league)
    return 0.0


def _attach_home_spread_margin(result: dict[str, Any], league: str) -> dict[str, Any]:
    """Expose the canonical spread margin used by official picks and UI."""
    if uses_spread_bets(league):
        result["home_spread_margin"] = round(blended_home_spread_margin(result, league), 2)
    return result


def _best_value_side_binary(
    total_score: float,
    away_market: int | None,
    home_market: int | None,
) -> str | None:
    away_prob, home_prob = _layer_side_win_probs(total_score)
    away_proj, home_proj = model_moneylines(total_score)
    edges: list[tuple[str, float]] = []
    if away_market is not None:
        edge = _odds_edge(away_proj, away_market, away_prob)
        if edge >= MIN_RECOMMENDED_EDGE:
            edges.append(("away", edge))
    if home_market is not None:
        edge = _odds_edge(home_proj, home_market, home_prob)
        if edge >= MIN_RECOMMENDED_EDGE:
            edges.append(("home", edge))
    if not edges:
        return None
    return max(edges, key=lambda item: item[1])[0]


def _layer_has_value_on_side_binary(
    total_score: float,
    side: str,
    away_market: int | None,
    home_market: int | None,
) -> bool:
    away_prob, home_prob = _layer_side_win_probs(total_score)
    away_proj, home_proj = model_moneylines(total_score)
    if side == "away":
        return (
            away_market is not None
            and _odds_edge(away_proj, away_market, away_prob) >= MIN_RECOMMENDED_EDGE
        )
    return (
        home_market is not None
        and _odds_edge(home_proj, home_market, home_prob) >= MIN_RECOMMENDED_EDGE
    )


def _layer_has_spread_value_on_side(
    layer: dict[str, Any],
    league: str,
    side: str,
    consensus_spread: float,
    *,
    spread_odds: int | None = None,
) -> bool:
    margin = _layer_home_margin(layer, league)
    if margin is None:
        return False
    point_edge = spread_point_edge(margin, consensus_spread, side)
    juice = spread_odds if spread_odds is not None else DEFAULT_SPREAD_JUICE
    return spread_odds_edge(point_edge, juice, league) >= MIN_RECOMMENDED_EDGE


def _best_value_spread_side(
    layer: dict[str, Any],
    league: str,
    consensus_spread: float,
    *,
    away_spread_odds: int | None = None,
    home_spread_odds: int | None = None,
) -> str | None:
    margin = _layer_home_margin(layer, league)
    if margin is None:
        return None
    edges: list[tuple[str, float]] = []
    for side, juice in (
        ("away", away_spread_odds),
        ("home", home_spread_odds),
    ):
        point_edge = spread_point_edge(margin, consensus_spread, side)
        edge = spread_odds_edge(point_edge, juice, league)
        if edge >= MIN_RECOMMENDED_EDGE:
            edges.append((side, edge))
    if not edges:
        return None
    return max(edges, key=lambda item: item[1])[0]


def _third_layer_key(blended: dict[str, Any]) -> str | None:
    for key in (
        "basketball_pred",
        "baseball_pred",
        "hockey_pred",
        "football_pred",
        "soccer_pred",
    ):
        if blended.get(key):
            return key
    return None


def compute_model_agreement(
    blended: dict[str, Any],
    league: str,
    *,
    market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    For 3-layer leagues, all layers must independently find value on the same side.

    Spread leagues: each layer needs spread edge >= MIN_RECOMMENDED_EDGE on one shared side.
    Other 3-layer sports: each layer needs moneyline edge >= MIN_RECOMMENDED_EDGE on one shared side.
    """
    if not requires_three_layer_agreement(league):
        return {"required": 0, "agreed": True, "agreement_mode": "value"}

    if is_soccer_league(league):
        return {"required": 0, "agreed": True, "agreement_mode": "value"}

    legacy = blended.get("legacy")
    power = blended.get("power")
    third_key = _third_layer_key(blended)
    third_payload = blended.get(third_key) if third_key else None
    market = market or {}

    away_market = market.get("away_moneyline")
    home_market = market.get("home_moneyline")
    consensus_spread = market.get("spread")
    away_spread_odds = market.get("away_spread_odds")
    home_spread_odds = market.get("home_spread_odds")
    use_spread = uses_spread_bets(league) and consensus_spread is not None

    def _spread_kwargs() -> dict[str, Any]:
        return {
            "away_spread_odds": away_spread_odds,
            "home_spread_odds": home_spread_odds,
        }

    def _incomplete_payload() -> dict[str, Any]:
        if use_spread and consensus_spread is not None:
            spread_line = float(consensus_spread)
            sk = _spread_kwargs()
            legacy_side = (
                _best_value_spread_side(legacy, league, spread_line, **sk)
                if legacy
                else None
            )
            power_side = (
                _best_value_spread_side(power, league, spread_line, **sk)
                if power
                else None
            )
            third_side = (
                _best_value_spread_side(third_payload, league, spread_line, **sk)
                if third_payload
                else None
            )
        else:
            legacy_total = _layer_binary_total_score(legacy) if legacy else None
            power_total = _layer_binary_total_score(power) if power else None
            third_total = (
                _layer_binary_total_score(third_payload) if third_payload else None
            )
            legacy_side = (
                _best_value_side_binary(legacy_total, away_market, home_market)
                if legacy_total is not None
                else None
            )
            power_side = (
                _best_value_side_binary(power_total, away_market, home_market)
                if power_total is not None
                else None
            )
            third_side = (
                _best_value_side_binary(third_total, away_market, home_market)
                if third_total is not None
                else None
            )
        return {
            "required": 3,
            "agreed": False,
            "legacy_side": legacy_side,
            "power_side": power_side,
            "third_side": third_side,
            "third_source": third_key,
            "agreement_mode": "value",
            "value_sides": [],
            "value_outcomes": [],
        }

    if not legacy or not power or not third_payload or blended.get("blend_layers", 0) < 3:
        return _incomplete_payload()

    legacy_total = _layer_binary_total_score(legacy)
    power_total = _layer_binary_total_score(power)
    third_total = _layer_binary_total_score(third_payload)
    if legacy_total is None or power_total is None or third_total is None:
        return _incomplete_payload()

    if use_spread:
        spread_line = float(consensus_spread)
        sk = _spread_kwargs()
        legacy_side = _best_value_spread_side(legacy, league, spread_line, **sk)
        power_side = _best_value_spread_side(power, league, spread_line, **sk)
        third_side = _best_value_spread_side(third_payload, league, spread_line, **sk)
        value_sides = [
            side
            for side in ("away", "home")
            if _layer_has_spread_value_on_side(
                legacy,
                league,
                side,
                spread_line,
                spread_odds=away_spread_odds if side == "away" else home_spread_odds,
            )
            and _layer_has_spread_value_on_side(
                power,
                league,
                side,
                spread_line,
                spread_odds=away_spread_odds if side == "away" else home_spread_odds,
            )
            and _layer_has_spread_value_on_side(
                third_payload,
                league,
                side,
                spread_line,
                spread_odds=away_spread_odds if side == "away" else home_spread_odds,
            )
        ]
    else:
        legacy_side = _best_value_side_binary(legacy_total, away_market, home_market)
        power_side = _best_value_side_binary(power_total, away_market, home_market)
        third_side = _best_value_side_binary(third_total, away_market, home_market)
        value_sides = [
            side
            for side in ("away", "home")
            if _layer_has_value_on_side_binary(
                legacy_total, side, away_market, home_market
            )
            and _layer_has_value_on_side_binary(
                power_total, side, away_market, home_market
            )
            and _layer_has_value_on_side_binary(
                third_total, side, away_market, home_market
            )
        ]

    return {
        "required": 3,
        "agreed": bool(value_sides),
        "legacy_side": legacy_side,
        "power_side": power_side,
        "third_side": third_side,
        "third_source": third_key,
        "agreement_mode": "value",
        "value_sides": value_sides,
        "value_outcomes": value_sides,
    }


def run_power_model(
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_name: str | None = None,
    away_name: str | None = None,
) -> dict[str, Any] | None:
    """Run power ratings for a matchup; None if insufficient data."""
    context = get_league_power_context(league, cutoff_date)
    if not context:
        return None

    teams, _games, param = context
    home_key = _find_team_key(teams, league, home_abbr, home_name)
    away_key = _find_team_key(teams, league, away_abbr, away_name)
    if not home_key or not away_key:
        return None

    prediction = predict_matchup(teams, param, home_key, away_key)
    if not prediction:
        return None

    return {
        "algorithm": "PowerRatings",
        "home_power": prediction["home_power"],
        "away_power": prediction["away_power"],
        "power_diff": prediction["power_diff"],
        "predicted_margin": prediction["power_diff"],
        "home_win_probability": prediction["home_win_probability"],
        "param": prediction["param"],
        "home_games": prediction["home_games"],
        "away_games": prediction["away_games"],
    }


def algo_v1_home_win_probability(total: float, favorite_win_probability: float) -> float:
    """Map Algo V1 signed total + favorite win% to home win probability."""
    if total <= 0:
        return favorite_win_probability
    return 100.0 - favorite_win_probability


def _blend_algo_v1_hockey_only(
    *,
    legacy_total_score: float,
    legacy_win_probability: float,
) -> dict[str, Any]:
    """Hockey uses legacy Algo V1 only — weighted factor sum + parabolic win curve."""
    home_prob = algo_v1_home_win_probability(legacy_total_score, legacy_win_probability)
    total, win_prob = home_win_prob_to_total_score(home_prob)
    legacy_payload = {
        "algorithm": "Algo_V1",
        "total_score": round(legacy_total_score, 2),
        "win_probability": round(legacy_win_probability, 2),
        "home_win_probability": round(home_prob, 2),
        "favorite_side": "home" if legacy_total_score <= 0 else "away",
        "source": "james-quintero-algo-v1",
    }
    return {
        "algorithm": "Algo_V1",
        "blend_mode": "algo_v1",
        "blend_layers": 1,
        "blend_weights": {"legacy": 1.0},
        "legacy": legacy_payload,
        "blended_home_win_probability": round(home_prob, 2),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": "home" if total <= 0 else "away",
    }


def _blend_hockey_pred_or_algo_v1(
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    legacy_total_score: float,
    legacy_win_probability: float,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
) -> dict[str, Any]:
    """NHL uses NHLGradientBoost v2 when artifacts exist; Algo V1 otherwise.

    College hockey always stays on Algo V1 (no v2 artifacts for those leagues).
    """
    sport_payload: dict[str, Any] | None = None
    if league.lower() == "nhl":
        try:
            sport_payload = run_hockey_pred_model(
                league,
                cutoff_date,
                home_abbr,
                away_abbr,
                home_moneyline=home_moneyline,
                away_moneyline=away_moneyline,
            )
        except Exception:  # noqa: BLE001 - hockey layer must never break the slate
            sport_payload = None
    if not sport_payload or sport_payload.get("model_version") != "v2":
        return _blend_algo_v1_hockey_only(
            legacy_total_score=legacy_total_score,
            legacy_win_probability=legacy_win_probability,
        )

    home_prob = float(sport_payload["home_win_probability"])
    total, win_prob = home_win_prob_to_total_score(home_prob)
    algorithm = str(sport_payload.get("algorithm") or "NHLGradientBoost")
    result: dict[str, Any] = {
        "algorithm": algorithm,
        "blend_mode": "nhl_v2",
        "blend_layers": 1,
        "blend_weights": {"hockey_pred": 1.0},
        "hockey_pred": sport_payload,
        "blended_home_win_probability": round(home_prob, 2),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": "home" if total <= 0 else "away",
        "model_version": "v2",
    }
    if sport_payload.get("market_decorrelated"):
        result["market_decorrelated"] = True
    if sport_payload.get("predicted_margin") is not None:
        result["home_spread_margin"] = round(-float(sport_payload["predicted_margin"]), 2)
    return result


def _run_wnba_v2(cutoff_date: str, home_abbr: str, away_abbr: str) -> dict[str, Any] | None:
    """WNBAGradientBoost v2 payload when artifacts + season context exist."""
    try:
        from web.hockey_pred_model import _cutoff_to_iso
        from web.wnba_v2.live import artifacts_available, predict_matchup_v2

        if not artifacts_available():
            return None
        day_iso = _cutoff_to_iso(cutoff_date)
        if not day_iso:
            return None
        return predict_matchup_v2(day_iso, home_abbr, away_abbr)
    except Exception:  # noqa: BLE001 - v2 layer must never break the slate
        return None


def _run_nba_v2(cutoff_date: str, home_abbr: str, away_abbr: str) -> dict[str, Any] | None:
    """NBAGradientBoost v2 payload when artifacts + season context exist."""
    try:
        from web.hockey_pred_model import _cutoff_to_iso
        from web.nba_v2.live import artifacts_available, predict_matchup_v2

        if not artifacts_available():
            return None
        day_iso = _cutoff_to_iso(cutoff_date)
        if not day_iso:
            return None
        return predict_matchup_v2(day_iso, home_abbr, away_abbr)
    except Exception:  # noqa: BLE001 - v2 layer must never break the slate
        return None


def _run_cbb_v2(
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    *,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
) -> dict[str, Any] | None:
    """CBBGradientBoost v2 payload when PIT-safe artifacts + season context exist."""
    try:
        from web.cbb_v2.live import artifacts_available, predict_matchup_v2
        from web.hockey_pred_model import _cutoff_to_iso

        if not artifacts_available():
            return None
        day_iso = _cutoff_to_iso(cutoff_date)
        if not day_iso:
            return None
        return predict_matchup_v2(
            day_iso,
            home_abbr,
            away_abbr,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
        )
    except Exception:  # noqa: BLE001 - v2 layer must never break the slate
        return None


def _run_cfb_v2(cutoff_date: str, home_abbr: str, away_abbr: str) -> dict[str, Any] | None:
    """CFBGradientBoost v2 payload when artifacts + season context exist."""
    try:
        from web.hockey_pred_model import _cutoff_to_iso
        from web.cfb_v2.live import artifacts_available, predict_matchup_v2

        if not artifacts_available():
            return None
        day_iso = _cutoff_to_iso(cutoff_date)
        if not day_iso:
            return None
        return predict_matchup_v2(day_iso, home_abbr, away_abbr)
    except Exception:  # noqa: BLE001 - v2 layer must never break the slate
        return None


def _run_nfl_v2(cutoff_date: str, home_abbr: str, away_abbr: str) -> dict[str, Any] | None:
    """NFLGradientBoost v2 payload when shipped artifacts + season context exist."""
    try:
        from web.hockey_pred_model import _cutoff_to_iso
        from web.nfl_v2.live import artifacts_available, predict_matchup_v2

        if not artifacts_available():
            return None
        day_iso = _cutoff_to_iso(cutoff_date)
        if not day_iso:
            return None
        return predict_matchup_v2(day_iso, home_abbr, away_abbr)
    except Exception:  # noqa: BLE001 - v2 layer must never break the slate
        return None


def _blend_cfb_v2_only(
    *,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """Prefer CFB v2 when shipped artifacts are available."""
    v2_payload = _run_cfb_v2(cutoff_date, home_abbr, away_abbr)
    if not v2_payload or v2_payload.get("model_version") != "v2":
        return None
    home_prob = float(v2_payload["home_win_probability"])
    total, win_prob = home_win_prob_to_total_score(home_prob)
    result: dict[str, Any] = {
        "algorithm": str(v2_payload.get("algorithm") or "CFBGradientBoost v2"),
        "blend_mode": "cfb_v2",
        "blend_layers": 1,
        "blend_weights": {"football_pred": 1.0},
        "football_pred": v2_payload,
        "blended_home_win_probability": round(home_prob, 2),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": "home" if total <= 0 else "away",
        "model_version": "v2",
    }
    if v2_payload.get("predicted_margin") is not None:
        result["home_spread_margin"] = round(-float(v2_payload["predicted_margin"]), 2)
    return result


def _blend_nfl_v2_only(
    *,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, Any] | None:
    """Prefer NFL v2 when shipped artifacts are available."""
    v2_payload = _run_nfl_v2(cutoff_date, home_abbr, away_abbr)
    if not v2_payload or v2_payload.get("model_version") != "v2":
        return None
    home_prob = float(v2_payload["home_win_probability"])
    total, win_prob = home_win_prob_to_total_score(home_prob)
    result: dict[str, Any] = {
        "algorithm": str(v2_payload.get("algorithm") or "NFLGradientBoost v2"),
        "blend_mode": "nfl_v2",
        "blend_layers": 1,
        "blend_weights": {"football_pred": 1.0},
        "football_pred": v2_payload,
        "blended_home_win_probability": round(home_prob, 2),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": "home" if total <= 0 else "away",
        "model_version": "v2",
    }
    if v2_payload.get("predicted_margin") is not None:
        result["home_spread_margin"] = round(-float(v2_payload["predicted_margin"]), 2)
    return result


def _blend_basketball_matrix_only(
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    legacy_total_score: float = 0.0,
    legacy_win_probability: float = 50.0,
    home_name: str | None = None,
    away_name: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
    consensus_spread: float | None = None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
) -> dict[str, Any]:
    """Basketball display prefers sport-specific models; BasketballMatrix is fallback.

    NBA/WNBA/CBB route to GradientBoost v2 when artifacts are available.
    CBB falls back to Torvik efficiency + calibration, then BasketballMatrix.
    """
    league_key = league.lower()
    if league_key == "nba":
        v2_payload = _run_nba_v2(cutoff_date, home_abbr, away_abbr)
        if v2_payload and v2_payload.get("model_version") == "v2":
            home_prob = float(v2_payload["home_win_probability"])
            total, win_prob = home_win_prob_to_total_score(home_prob)
            result: dict[str, Any] = {
                "algorithm": str(v2_payload.get("algorithm") or "NBAGradientBoost v2"),
                "blend_mode": "nba_v2",
                "blend_layers": 1,
                "blend_weights": {"basketball_pred": 1.0},
                "basketball_pred": v2_payload,
                "blended_home_win_probability": round(home_prob, 2),
                "total_score": round(total, 2),
                "win_probability": round(win_prob, 2),
                "favorite_side": "home" if total <= 0 else "away",
                "model_version": "v2",
            }
            if v2_payload.get("predicted_margin") is not None:
                result["home_spread_margin"] = round(
                    -float(v2_payload["predicted_margin"]), 2
                )
            return result
    if league_key == "wnba":
        v2_payload = _run_wnba_v2(cutoff_date, home_abbr, away_abbr)
        if v2_payload and v2_payload.get("model_version") == "v2":
            home_prob = float(v2_payload["home_win_probability"])
            total, win_prob = home_win_prob_to_total_score(home_prob)
            result = {
                "algorithm": str(v2_payload.get("algorithm") or "WNBAGradientBoost v2"),
                "blend_mode": "wnba_v2",
                "blend_layers": 1,
                "blend_weights": {"basketball_pred": 1.0},
                "basketball_pred": v2_payload,
                "blended_home_win_probability": round(home_prob, 2),
                "total_score": round(total, 2),
                "win_probability": round(win_prob, 2),
                "favorite_side": "home" if total <= 0 else "away",
                "model_version": "v2",
            }
            if v2_payload.get("predicted_margin") is not None:
                result["home_spread_margin"] = round(
                    -float(v2_payload["predicted_margin"]), 2
                )
            return result
    if league_key == "cbb":
        v2_payload = _run_cbb_v2(
            cutoff_date,
            home_abbr,
            away_abbr,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
        )
        if v2_payload and v2_payload.get("model_version") == "v2":
            home_prob = float(v2_payload["home_win_probability"])
            total, win_prob = home_win_prob_to_total_score(home_prob)
            result = {
                "algorithm": str(v2_payload.get("algorithm") or "CBBGradientBoost v2"),
                "blend_mode": "cbb_v2",
                "blend_layers": 1,
                "blend_weights": {"basketball_pred": 1.0},
                "basketball_pred": v2_payload,
                "blended_home_win_probability": round(home_prob, 2),
                "total_score": round(total, 2),
                "win_probability": round(win_prob, 2),
                "favorite_side": "home" if total <= 0 else "away",
                "model_version": "v2",
            }
            if v2_payload.get("predicted_margin") is not None:
                result["home_spread_margin"] = round(
                    -float(v2_payload["predicted_margin"]), 2
                )
            return result

    legacy_home = total_score_to_home_win_prob(legacy_total_score)
    legacy_payload = {
        "algorithm": "Algo_V2",
        "total_score": round(legacy_total_score, 2),
        "win_probability": round(legacy_win_probability, 2),
        "home_win_probability": round(legacy_home, 2),
        "favorite_side": "home" if legacy_total_score <= 0 else "away",
    }
    power_payload = run_power_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_name=home_name,
        away_name=away_name,
    )

    sport_payload: dict[str, Any] | None = None
    if is_cbb_league(league):
        try:
            sport_payload = run_cbb_pred_model(
                league,
                cutoff_date,
                home_abbr,
                away_abbr,
                home_espn_id=home_espn_id,
                away_espn_id=away_espn_id,
                home_name=home_name,
                away_name=away_name,
                market_spread=consensus_spread,
                home_ml=home_moneyline,
                away_ml=away_moneyline,
            )
        except Exception:  # noqa: BLE001 - Torvik must never break the slate
            sport_payload = None

    if not sport_payload:
        sport_payload = run_basketball_pred_model(
            league,
            cutoff_date,
            home_abbr,
            away_abbr,
        )
    if not sport_payload:
        reason = basketball_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
        return {
            "algorithm": "BasketballMatrix",
            "blend_mode": "basketball_matrix_unavailable",
            "blend_layers": 0,
            "blend_note": reason,
            "legacy": legacy_payload,
            "power": power_payload,
            "basketball_pred": None,
            "total_score": 0.0,
            "win_probability": 50.0,
            "favorite_side": "neutral",
        }

    home_prob = float(sport_payload["home_win_probability"])
    total, win_prob = home_win_prob_to_total_score(home_prob)
    feature_layers = 1 + int(power_payload is not None)
    algorithm = str(sport_payload.get("algorithm") or "BasketballMatrix")
    is_torvik = algorithm == "CBBTorvik"
    result = {
        "algorithm": algorithm,
        "blend_mode": "cbb_torvik" if is_torvik else "basketball_matrix",
        "blend_layers": feature_layers,
        "blend_weights": {"basketball_pred": 1.0},
        "legacy": legacy_payload,
        "power": power_payload,
        "basketball_pred": sport_payload,
        "blended_home_win_probability": round(home_prob, 2),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": "home" if total <= 0 else "away",
    }
    if sport_payload.get("market_decorrelated"):
        result["market_decorrelated"] = True
    if sport_payload.get("pre_decorrelation_home_win_probability") is not None:
        result["pre_decorrelation_home_win_probability"] = sport_payload[
            "pre_decorrelation_home_win_probability"
        ]
    if sport_payload.get("cbb_pick_signals") is not None:
        result["cbb_pick_signals"] = sport_payload["cbb_pick_signals"]
    if sport_payload.get("predicted_margin") is not None:
        result["home_spread_margin"] = round(-float(sport_payload["predicted_margin"]), 2)
    return result


def _blend_mlb_runcast_only(
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
    home_name: str | None = None,
    away_name: str | None = None,
    consensus_spread: float | None = None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
) -> dict[str, Any]:
    """MLB uses MLBRunCast only — no Algo V2, power, meta, db_rating, or EnsembleML."""
    sport_payload = run_mlb_pred_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
        home_name=home_name,
        away_name=away_name,
        market_spread=consensus_spread,
        home_ml=home_moneyline,
        away_ml=away_moneyline,
    )
    if not sport_payload:
        reason = mlb_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
        return {
            "algorithm": "MLBRunCast",
            "blend_mode": "mlb_runcast_unavailable",
            "blend_layers": 0,
            "blend_note": reason,
            "baseball_pred": None,
            "total_score": 0.0,
            "win_probability": 50.0,
            "favorite_side": "neutral",
        }

    home_prob = float(sport_payload["home_win_probability"])
    total, win_prob = home_win_prob_to_total_score(home_prob)
    algorithm = str(sport_payload.get("algorithm") or "MLBRunCast")
    result = {
        "algorithm": algorithm,
        "blend_mode": "mlb_runcast",
        "blend_layers": 1,
        "blend_weights": {"baseball_pred": 1.0},
        "baseball_pred": sport_payload,
        "blended_home_win_probability": round(home_prob, 2),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": "home" if total <= 0 else "away",
        "mlb_pick_signals": sport_payload.get("mlb_pick_signals"),
    }
    if sport_payload.get("model_version"):
        result["model_version"] = sport_payload["model_version"]
    if sport_payload.get("predicted_margin") is not None:
        result["home_spread_margin"] = round(-float(sport_payload["predicted_margin"]), 2)
    return result


def _blend_soccer_path_a_only(
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_name: str | None = None,
    away_name: str | None = None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
    draw_moneyline: int | None = None,
    home_slug: str | None = None,
    away_slug: str | None = None,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
    match_date: str = "",
) -> dict[str, Any]:
    """Soccer Path A with db_rating + ESPN context enrichment."""
    sport_payload = run_soccer_pred_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_name=home_name,
        away_name=away_name,
        home_ml=home_moneyline,
        draw_ml=draw_moneyline,
        away_ml=away_moneyline,
        event_id=event_id,
        match_date=match_date,
    )
    if not sport_payload:
        reason = soccer_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
        return {
            "algorithm": "SoccerPathA",
            "blend_mode": "soccer_path_a_unavailable",
            "blend_layers": 0,
            "blend_note": reason,
            "soccer_pred": None,
            "total_score": 0.0,
            "win_probability": 50.0,
            "favorite_side": "neutral",
            "threeway": True,
        }

    is_v2 = sport_payload.get("algorithm") == "SoccerGradientBoost v2"
    algorithm = "SoccerGradientBoost v2" if is_v2 else "SoccerPathA"
    blend_mode = "soccer_v2" if is_v2 else "soccer_path_a"

    home_p = float(sport_payload["home_win_probability"])
    draw_p = float(sport_payload["draw_probability"])
    away_p = float(sport_payload["away_win_probability"])
    total, win_prob, favorite = threeway_probs_to_total_score(home_p, draw_p, away_p)
    result = {
        "algorithm": algorithm,
        "blend_mode": blend_mode,
        "blend_layers": 1,
        "blend_weights": {"soccer_pred": 1.0},
        "soccer_pred": sport_payload,
        "blended_home_win_probability": round(home_p, 2),
        "home_win_probability": round(home_p, 2),
        "draw_probability": round(draw_p, 2),
        "away_win_probability": round(away_p, 2),
        "blended_threeway": {
            "home_win_probability": round(home_p, 2),
            "draw_probability": round(draw_p, 2),
            "away_win_probability": round(away_p, 2),
        },
        "expected_home_goals": sport_payload.get("expected_home_goals"),
        "expected_away_goals": sport_payload.get("expected_away_goals"),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": favorite,
        "threeway": True,
        "soccer_pick_signals": sport_payload.get("soccer_pick_signals"),
    }
    if is_v2:
        # v2 display probabilities are already market-calibrated and the pick
        # probabilities are decorrelated; keep them untouched (no db_rating /
        # ESPN-context layers, which are Path-A-tuned).
        result["market_decorrelated"] = bool(sport_payload.get("market_decorrelated"))
        if sport_payload.get("model_version"):
            result["model_version"] = sport_payload["model_version"]
        return result
    return finalize_soccer_path_a(
        result,
        league=league,
        cutoff_date=cutoff_date,
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        home_slug=home_slug,
        away_slug=away_slug,
        home_name=home_name,
        away_name=away_name,
        event_id=event_id,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
    )


def blend_predictions(
    *,
    legacy_total_score: float,
    legacy_win_probability: float,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_name: str | None = None,
    away_name: str | None = None,
    legacy_weight: float = LEGACY_BLEND_WEIGHT,
    power_weight: float = POWER_BLEND_WEIGHT,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
    home_slug: str | None = None,
    away_slug: str | None = None,
    consensus_spread: float | None = None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
    draw_moneyline: int | None = None,
    headlines: list[str] | None = None,
) -> dict[str, Any]:
    """
    Blend Algo_V2 and power model into unified total_score / win_probability.

    Hockey leagues use Algo V1 only. Basketball prefers sport models
    (NBA/WNBA v2, CBB Torvik) with BasketballMatrix fallback.
    Baseball and football use a third sport layer blended via meta weights.

    ``headlines`` feed the news context signal for leagues whose context
    layer runs inside this blend (football/generic); specialized paths get
    context applied by the caller instead.
    """
    if is_basketball_league(league):
        return _blend_basketball_matrix_only(
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            legacy_total_score=legacy_total_score,
            legacy_win_probability=legacy_win_probability,
            home_name=home_name,
            away_name=away_name,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
            consensus_spread=consensus_spread,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
        )

    if is_hockey_league(league):
        return _blend_hockey_pred_or_algo_v1(
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            legacy_total_score=legacy_total_score,
            legacy_win_probability=legacy_win_probability,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
        )

    if is_mlb_league(league):
        return _blend_mlb_runcast_only(
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
            home_name=home_name,
            away_name=away_name,
            consensus_spread=consensus_spread,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
        )

    if is_soccer_league(league):
        return _blend_soccer_path_a_only(
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_name=home_name,
            away_name=away_name,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
            draw_moneyline=draw_moneyline,
            home_slug=home_slug,
            away_slug=away_slug,
            event_id=event_id,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
        )

    if league.lower() == "cfb":
        cfb_v2 = _blend_cfb_v2_only(
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
        )
        if cfb_v2 is not None:
            return cfb_v2

    if league.lower() == "nfl":
        nfl_v2 = _blend_nfl_v2_only(
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
        )
        if nfl_v2 is not None:
            return nfl_v2

    legacy_home = total_score_to_home_win_prob(legacy_total_score)
    legacy_payload = {
        "algorithm": "Algo_V2",
        "total_score": round(legacy_total_score, 2),
        "win_probability": round(legacy_win_probability, 2),
        "home_win_probability": round(legacy_home, 2),
        "favorite_side": "home" if legacy_total_score <= 0 else "away",
    }

    power_payload = run_power_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_name=home_name,
        away_name=away_name,
    )

    if not power_payload:
        total = legacy_total_score
        win_prob = legacy_win_probability
        reason = power_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
        return _with_db_rating_layer(
            {
            "algorithm": "Unified",
            "blend_mode": "legacy_only",
            "blend_note": f"Power model unavailable — {reason} Using Algo V2 only.",
            "legacy": legacy_payload,
            "power": None,
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": legacy_payload["favorite_side"],
            },
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_slug=home_slug,
            away_slug=away_slug,
            home_name=home_name,
            away_name=away_name,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
            headlines=headlines,
        )

    legacy_home = total_score_to_home_win_prob(legacy_total_score)
    power_home = float(power_payload["home_win_probability"])
    sport_key, sport_payload = _run_sport_pred_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        market_spread=consensus_spread,
        home_ml=home_moneyline,
        away_ml=away_moneyline,
    )
    if is_baseball_league(league):
        meta = get_sports_meta_config(league)
        if meta["blend_weights"].get("sport_pred", 0.0) <= 0:
            sport_key, sport_payload = None, None

    if _uses_three_layer_blend(league) and sport_payload and sport_key:
        third_home = float(sport_payload["home_win_probability"])
        blended_home = stack_binary_blend_layers(
            legacy_home=legacy_home,
            power_home=power_home,
            sport_home=third_home,
            league=league,
            sport_key=sport_key,
        )
        if blended_home is None:
            blended_home = (
                THREE_LAYER_WEIGHT * legacy_home
                + THREE_LAYER_WEIGHT * power_home
                + THREE_LAYER_WEIGHT * third_home
            )
        blended_home = min(max(blended_home, 0.0), 100.0)
        total, win_prob = home_win_prob_to_total_score(blended_home)
        meta = get_sports_meta_config(league)
        blend_weights = {
            "legacy": meta["blend_weights"].get("legacy", THREE_LAYER_WEIGHT),
            "power": meta["blend_weights"].get("power", THREE_LAYER_WEIGHT),
            sport_key: meta["blend_weights"].get("sport_pred", THREE_LAYER_WEIGHT),
        }
        result: dict[str, Any] = {
            "algorithm": "Unified",
            "blend_mode": "blended",
            "blend_layers": 3,
            "blend_weights": blend_weights,
            "legacy": legacy_payload,
            "power": power_payload,
            sport_key: sport_payload,
            "blended_home_win_probability": round(blended_home, 2),
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": "home" if total <= 0 else "away",
        }
        return _with_db_rating_layer(
            result,
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_slug=home_slug,
            away_slug=away_slug,
            home_name=home_name,
            away_name=away_name,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
            headlines=headlines,
        )

    weight_sum = legacy_weight + power_weight
    meta_weights = get_sports_meta_config(league)["blend_weights"]
    blended_home = stack_binary_blend_layers(
        legacy_home=legacy_home,
        power_home=power_home,
        sport_home=None,
        league=league,
    )
    if blended_home is None:
        blended_home = (
            legacy_weight * legacy_home + power_weight * power_home
        ) / weight_sum
    blended_home = min(max(blended_home, 0.0), 100.0)
    total, win_prob = home_win_prob_to_total_score(blended_home)

    result = {
        "algorithm": "Unified",
        "blend_mode": "blended",
        "blend_layers": 2,
        "blend_weights": {
            "legacy": meta_weights.get("legacy", legacy_weight),
            "power": meta_weights.get("power", power_weight),
        },
        "legacy": legacy_payload,
        "power": power_payload,
        "blended_home_win_probability": round(blended_home, 2),
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": "home" if total <= 0 else "away",
    }

    if _uses_three_layer_blend(league) and not sport_payload:
        if is_basketball_league(league):
            layer_name = "Basketball matrix"
        elif is_baseball_league(league):
            layer_name = "SharpBaseball"
        elif is_hockey_league(league):
            layer_name = "Hockey-predictions"
        elif is_football_league(league):
            layer_name = "nfelo"
        elif is_soccer_league(league):
            layer_name = "Soccer ratings"
        else:
            layer_name = "Sport model"
        reason = _sport_pred_unavailable_reason(
            league, cutoff_date, home_abbr, away_abbr
        )
        result["blend_note"] = (
            f"{layer_name} layer unavailable — {reason} Using 2-layer blend."
        )

    return _with_db_rating_layer(
        result,
        league=league,
        cutoff_date=cutoff_date,
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        home_slug=home_slug,
        away_slug=away_slug,
        home_name=home_name,
        away_name=away_name,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
        headlines=headlines,
    )
