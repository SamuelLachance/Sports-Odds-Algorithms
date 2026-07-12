"""Three-layer 1X2 blend, optional context layer, and value agreement for soccer."""

from __future__ import annotations

from typing import Any

from web.bet_advisor import (
    MIN_RECOMMENDED_EDGE,
    _odds_edge,
    soccer_model_moneylines,
    soccer_threeway_probs,
)
from web.league_profiles import is_soccer_league
from web.season_games import power_unavailable_reason
from web.soccer_context import build_soccer_context_payload
from web.db_rating_model import apply_db_rating_blend
from web.soccer_meta_model import (
    apply_threeway_calibration,
    get_league_meta_config,
    stack_soccer_blend_layers,
)
from web.soccer_pred_model import run_soccer_pred_model, soccer_unavailable_reason

LEGACY_BLEND_WEIGHT = 0.5
POWER_BLEND_WEIGHT = 0.5
THREE_LAYER_WEIGHT = 1.0 / 3.0


def home_win_prob_to_total_score(home_win_prob: float) -> tuple[float, float]:
    if abs(home_win_prob - 50.0) < 1e-9:
        return 0.0, 50.0
    if home_win_prob > 50.0:
        return -home_win_prob, home_win_prob
    away_prob = 100.0 - home_win_prob
    return away_prob, away_prob


def threeway_probs_to_total_score(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
) -> tuple[float, float, str]:
    if draw_prob > home_prob and draw_prob > away_prob:
        return 0.0, draw_prob, "draw"
    non_draw = home_prob + away_prob
    if non_draw <= 0:
        return 0.0, 50.0, "home"
    home_binary = home_prob / non_draw * 100.0
    total, win_prob = home_win_prob_to_total_score(home_binary)
    favorite = "home" if home_prob >= away_prob else "away"
    return total, win_prob, favorite


def _normalize_threeway_blend(
    home: float, draw: float, away: float
) -> tuple[float, float, float]:
    total = home + draw + away
    if total <= 0:
        return 33.33, 33.33, 33.34
    scale = 100.0 / total
    return round(home * scale, 2), round(draw * scale, 2), round(away * scale, 2)


def blend_threeway_layers(
    layers: list[tuple[float, float, float]],
    weights: list[float],
) -> tuple[float, float, float]:
    if not layers or not weights or abs(sum(weights)) < 1e-12:
        return 33.33, 33.33, 33.34
    weight_sum = sum(weights)
    home = sum(w * layer[0] for w, layer in zip(weights, layers)) / weight_sum
    draw = sum(w * layer[1] for w, layer in zip(weights, layers)) / weight_sum
    away = sum(w * layer[2] for w, layer in zip(weights, layers)) / weight_sum
    return _normalize_threeway_blend(home, draw, away)


def power_threeway_probs(power_home: float, league: str) -> tuple[float, float, float]:
    total, _ = home_win_prob_to_total_score(power_home)
    return soccer_threeway_probs(total, league)


def best_value_outcome_threeway(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    away_market: int | None,
    draw_market: int | None,
    home_market: int | None,
) -> str | None:
    away_proj, draw_proj, home_proj = soccer_model_moneylines(
        home_prob, draw_prob, away_prob
    )
    edges: list[tuple[str, float]] = []
    for side, prob, proj, market in (
        ("away", away_prob, away_proj, away_market),
        ("draw", draw_prob, draw_proj, draw_market),
        ("home", home_prob, home_proj, home_market),
    ):
        if market is None:
            continue
        edge = _odds_edge(proj, market, prob)
        if edge >= MIN_RECOMMENDED_EDGE:
            edges.append((side, edge))
    if not edges:
        return None
    return max(edges, key=lambda item: item[1])[0]


def layer_has_value_on_outcome_threeway(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    outcome: str,
    away_market: int | None,
    draw_market: int | None,
    home_market: int | None,
) -> bool:
    away_proj, draw_proj, home_proj = soccer_model_moneylines(
        home_prob, draw_prob, away_prob
    )
    if outcome == "away":
        return (
            away_market is not None
            and _odds_edge(away_proj, away_market, away_prob) >= MIN_RECOMMENDED_EDGE
        )
    if outcome == "draw":
        return (
            draw_market is not None
            and _odds_edge(draw_proj, draw_market, draw_prob) >= MIN_RECOMMENDED_EDGE
        )
    return (
        home_market is not None
        and _odds_edge(home_proj, home_market, home_prob) >= MIN_RECOMMENDED_EDGE
    )


def compute_soccer_model_agreement(
    blended: dict[str, Any],
    *,
    market: dict[str, Any],
    third_key: str | None,
    third_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    legacy = blended.get("legacy")
    power = blended.get("power")
    away_market = market.get("away_moneyline")
    home_market = market.get("home_moneyline")
    draw_market = market.get("draw_moneyline")

    legacy_tw = blended.get("legacy_threeway")
    power_tw = blended.get("power_threeway")

    def _incomplete() -> dict[str, Any]:
        return {
            "required": 3,
            "agreed": False,
            "legacy_side": None,
            "power_side": None,
            "third_side": None,
            "third_source": third_key,
            "agreement_mode": "value",
            "value_outcomes": [],
            "value_sides": [],
        }

    if (
        not legacy
        or not power
        or not third_payload
        or not legacy_tw
        or not power_tw
        or blended.get("blend_layers", 0) < 3
    ):
        return _incomplete()

    legacy_probs = (
        float(legacy_tw["home_win_probability"]),
        float(legacy_tw["draw_probability"]),
        float(legacy_tw["away_win_probability"]),
    )
    power_probs = (
        float(power_tw["home_win_probability"]),
        float(power_tw["draw_probability"]),
        float(power_tw["away_win_probability"]),
    )
    third_probs = (
        float(third_payload["home_win_probability"]),
        float(third_payload["draw_probability"]),
        float(third_payload["away_win_probability"]),
    )
    legacy_side = best_value_outcome_threeway(
        *legacy_probs, away_market, draw_market, home_market
    )
    power_side = best_value_outcome_threeway(
        *power_probs, away_market, draw_market, home_market
    )
    third_side = best_value_outcome_threeway(
        *third_probs, away_market, draw_market, home_market
    )
    value_outcomes = [
        outcome
        for outcome in ("home", "draw", "away")
        if layer_has_value_on_outcome_threeway(
            *legacy_probs, outcome, away_market, draw_market, home_market
        )
        and layer_has_value_on_outcome_threeway(
            *power_probs, outcome, away_market, draw_market, home_market
        )
        and layer_has_value_on_outcome_threeway(
            *third_probs, outcome, away_market, draw_market, home_market
        )
    ]
    return {
        "required": 3,
        "agreed": bool(value_outcomes),
        "legacy_side": legacy_side,
        "power_side": power_side,
        "third_side": third_side,
        "third_source": third_key,
        "agreement_mode": "value",
        "value_outcomes": value_outcomes,
        "value_sides": value_outcomes,
    }


def _apply_final_threeway_calibration(
    result: dict[str, Any],
    league: str,
) -> dict[str, Any]:
    if not result.get("threeway"):
        return result
    home_p, draw_p, away_p = apply_threeway_calibration(
        float(result["home_win_probability"]),
        float(result["draw_probability"]),
        float(result["away_win_probability"]),
        league,
    )
    total, win_prob, favorite = threeway_probs_to_total_score(home_p, draw_p, away_p)
    updated = dict(result)
    updated.update(
        {
            "home_win_probability": home_p,
            "draw_probability": draw_p,
            "away_win_probability": away_p,
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": favorite,
        }
    )
    if updated.get("blended_threeway"):
        updated["blended_threeway"] = {
            "home_win_probability": home_p,
            "draw_probability": draw_p,
            "away_win_probability": away_p,
        }
    return updated


def finalize_soccer_path_a(
    result: dict[str, Any],
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_slug: str | None = None,
    away_slug: str | None = None,
    home_name: str | None = None,
    away_name: str | None = None,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
) -> dict[str, Any]:
    """Apply soccer-rating.com + ESPN context layers on top of SoccerPathA."""
    if not result.get("soccer_pred"):
        return result
    enriched = dict(result)
    enriched["threeway"] = True
    enriched.setdefault("blend_weights", {"soccer_pred": 1.0})
    enriched.setdefault("blend_layers", 1)
    if home_slug and away_slug:
        enriched = apply_db_rating_blend(
            enriched,
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_slug=home_slug,
            away_slug=away_slug,
            home_name=home_name,
            away_name=away_name,
        )
        soccer_pred = dict(enriched.get("soccer_pred") or {})
        soccer_pred.update(
            {
                "home_win_probability": enriched["home_win_probability"],
                "draw_probability": enriched["draw_probability"],
                "away_win_probability": enriched["away_win_probability"],
            }
        )
        enriched["soccer_pred"] = soccer_pred
        enriched["blend_layers"] = int(enriched.get("blend_layers") or 1)
    return _attach_soccer_context_layer(
        enriched,
        league=league,
        cutoff_date=cutoff_date,
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        event_id=event_id,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
    )


def _finalize_soccer_blend(
    result: dict[str, Any],
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_slug: str | None = None,
    away_slug: str | None = None,
    home_name: str | None = None,
    away_name: str | None = None,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
) -> dict[str, Any]:
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
    return _apply_final_threeway_calibration(
        _attach_soccer_context_layer(
        result,
        league=league,
        cutoff_date=cutoff_date,
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        event_id=event_id,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
        ),
        league,
    )


def attach_soccer_context_display(
    result: dict[str, Any],
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
) -> dict[str, Any]:
    """Attach ESPN injury/form context for UI display without mutating win probs.

    Used by SoccerGradientBoost v2 where calibrated / market-decorrelated
    probabilities must stay intact for official picks.
    """
    context = build_soccer_context_payload(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        float(result["home_win_probability"]),
        float(result["draw_probability"]),
        float(result["away_win_probability"]),
        event_id=event_id,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
    )
    if not context:
        # Still lock the daily FLB/news hook — v2 probs must not be re-shifted.
        updated = dict(result)
        updated["context_adjustment_pp"] = 0.0
        return updated
    updated = dict(result)
    updated["soccer_context"] = context
    # Sentinel so daily_service does not re-apply probability-shifting context.
    updated["context_adjustment_pp"] = 0.0
    return updated


def _attach_soccer_context_layer(
    result: dict[str, Any],
    *,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
) -> dict[str, Any]:
    context = build_soccer_context_payload(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        float(result["home_win_probability"]),
        float(result["draw_probability"]),
        float(result["away_win_probability"]),
        event_id=event_id,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
    )
    if not context:
        # Lock daily FLB/news re-apply even when ESPN context is unavailable,
        # matching attach_soccer_context_display (v2) so Path A is symmetric.
        updated = dict(result)
        updated["context_adjustment_pp"] = 0.0
        return updated

    home_p = float(context["home_win_probability"])
    draw_p = float(context["draw_probability"])
    away_p = float(context["away_win_probability"])
    total, win_prob, favorite = threeway_probs_to_total_score(home_p, draw_p, away_p)
    updated = dict(result)
    updated["home_win_probability"] = home_p
    updated["draw_probability"] = draw_p
    updated["away_win_probability"] = away_p
    # Keep blended_* in sync — apply_context_to_blend prefers blended_home.
    updated["blended_home_win_probability"] = round(home_p, 2)
    updated["total_score"] = round(total, 2)
    updated["win_probability"] = round(win_prob, 2)
    updated["favorite_side"] = favorite
    updated["soccer_context"] = context
    updated["context_adjusted"] = True
    # Path A ESPN context is final for display/EV base; block daily re-apply.
    updated["context_adjustment_pp"] = 0.0
    if updated.get("soccer_pred"):
        soccer_pred = dict(updated["soccer_pred"])
        soccer_pred.update(
            {
                "home_win_probability": home_p,
                "draw_probability": draw_p,
                "away_win_probability": away_p,
            }
        )
        updated["soccer_pred"] = soccer_pred
    if updated.get("blended_threeway"):
        updated["blended_threeway"] = {
            "home_win_probability": home_p,
            "draw_probability": draw_p,
            "away_win_probability": away_p,
        }
    return updated


def blend_soccer_predictions(
    *,
    legacy_total_score: float,
    legacy_win_probability: float,
    league: str,
    cutoff_date: str,
    home_abbr: str,
    away_abbr: str,
    home_name: str | None,
    away_name: str | None,
    legacy_payload: dict[str, Any],
    power_payload: dict[str, Any] | None,
    event_id: str | None = None,
    home_espn_id: str | None = None,
    away_espn_id: str | None = None,
    home_slug: str | None = None,
    away_slug: str | None = None,
) -> dict[str, Any]:
    legacy_threeway = soccer_threeway_probs(legacy_total_score, league)
    legacy_threeway_payload = {
        "algorithm": "Algo_V2",
        "home_win_probability": round(legacy_threeway[0], 2),
        "draw_probability": round(legacy_threeway[1], 2),
        "away_win_probability": round(legacy_threeway[2], 2),
        "total_score": legacy_payload["total_score"],
        "favorite_side": legacy_payload["favorite_side"],
    }

    if not power_payload:
        total = legacy_total_score
        win_prob = legacy_win_probability
        reason = power_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
        home_p, draw_p, away_p = legacy_threeway
        return _finalize_soccer_blend(
            {
            "algorithm": "Unified",
            "blend_mode": "legacy_only",
            "blend_note": f"Power model unavailable — {reason} Using Algo V2 only.",
            "legacy": legacy_payload,
            "legacy_threeway": legacy_threeway_payload,
            "power": None,
            "threeway": True,
            "home_win_probability": round(home_p, 2),
            "draw_probability": round(draw_p, 2),
            "away_win_probability": round(away_p, 2),
            "expected_home_goals": None,
            "expected_away_goals": None,
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": legacy_payload["favorite_side"],
            },
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_name=home_name,
            away_name=away_name,
            home_slug=home_slug,
            away_slug=away_slug,
            event_id=event_id,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
        )

    power_threeway = power_threeway_probs(
        float(power_payload["home_win_probability"]), league
    )
    power_threeway_payload = {
        "algorithm": "PowerRatings",
        "home_win_probability": round(power_threeway[0], 2),
        "draw_probability": round(power_threeway[1], 2),
        "away_win_probability": round(power_threeway[2], 2),
        "home_power": power_payload["home_power"],
        "away_power": power_payload["away_power"],
    }

    sport_payload = run_soccer_pred_model(
        league,
        cutoff_date,
        home_abbr,
        away_abbr,
        home_name=home_name,
        away_name=away_name,
    )

    if sport_payload:
        soccer_threeway = (
            float(sport_payload["home_win_probability"]),
            float(sport_payload["draw_probability"]),
            float(sport_payload["away_win_probability"]),
        )
        stacked = stack_soccer_blend_layers(
            legacy=legacy_threeway,
            power=power_threeway,
            soccer_pred=soccer_threeway,
            league=league,
        )
        if stacked:
            home_p, draw_p, away_p = stacked
        else:
            home_p, draw_p, away_p = blend_threeway_layers(
                [legacy_threeway, power_threeway, soccer_threeway],
                [THREE_LAYER_WEIGHT, THREE_LAYER_WEIGHT, THREE_LAYER_WEIGHT],
            )
        blend_weights = get_league_meta_config(league)["blend_weights"]
        total, win_prob, favorite = threeway_probs_to_total_score(home_p, draw_p, away_p)
        return _finalize_soccer_blend(
            {
            "algorithm": "Unified",
            "blend_mode": "blended",
            "blend_layers": 3,
            "blend_weights": blend_weights,
            "legacy": legacy_payload,
            "legacy_threeway": legacy_threeway_payload,
            "power": power_payload,
            "power_threeway": power_threeway_payload,
            "soccer_pred": sport_payload,
            "threeway": True,
            "home_win_probability": home_p,
            "draw_probability": draw_p,
            "away_win_probability": away_p,
            "blended_threeway": {
                "home_win_probability": home_p,
                "draw_probability": draw_p,
                "away_win_probability": away_p,
            },
            "expected_home_goals": sport_payload.get("expected_home_goals"),
            "expected_away_goals": sport_payload.get("expected_away_goals"),
            "total_score": round(total, 2),
            "win_probability": round(win_prob, 2),
            "favorite_side": favorite,
            },
            league=league,
            cutoff_date=cutoff_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_name=home_name,
            away_name=away_name,
            home_slug=home_slug,
            away_slug=away_slug,
            event_id=event_id,
            home_espn_id=home_espn_id,
            away_espn_id=away_espn_id,
        )

    home_p, draw_p, away_p = blend_threeway_layers(
        [legacy_threeway, power_threeway],
        [LEGACY_BLEND_WEIGHT, POWER_BLEND_WEIGHT],
    )
    stacked = stack_soccer_blend_layers(
        legacy=legacy_threeway,
        power=power_threeway,
        soccer_pred=None,
        league=league,
    )
    if stacked:
        home_p, draw_p, away_p = stacked
    total, win_prob, favorite = threeway_probs_to_total_score(home_p, draw_p, away_p)
    reason = soccer_unavailable_reason(league, cutoff_date, home_abbr, away_abbr)
    return _finalize_soccer_blend(
        {
        "algorithm": "Unified",
        "blend_mode": "blended",
        "blend_layers": 2,
        "blend_weights": {
            "legacy": LEGACY_BLEND_WEIGHT,
            "power": POWER_BLEND_WEIGHT,
        },
        "legacy": legacy_payload,
        "legacy_threeway": legacy_threeway_payload,
        "power": power_payload,
        "power_threeway": power_threeway_payload,
        "threeway": True,
        "home_win_probability": home_p,
        "draw_probability": draw_p,
        "away_win_probability": away_p,
        "total_score": round(total, 2),
        "win_probability": round(win_prob, 2),
        "favorite_side": favorite,
        "blend_note": f"Soccer ratings layer unavailable — {reason} Using 2-layer blend.",
        },
        league=league,
        cutoff_date=cutoff_date,
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        home_name=home_name,
        away_name=away_name,
        home_slug=home_slug,
        away_slug=away_slug,
        event_id=event_id,
        home_espn_id=home_espn_id,
        away_espn_id=away_espn_id,
    )


def is_soccer_blend(league: str) -> bool:
    return is_soccer_league(league)
