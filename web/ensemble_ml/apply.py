"""Apply trained ensemble mega-model on top of layer blend outputs."""

from __future__ import annotations

from typing import Any

from web.bet_advisor import model_home_margin
from web.blend_service import home_win_prob_to_total_score
from web.ensemble_ml.config import ensemble_model_available, is_spread_league
from web.ensemble_ml.predict import predict_ensemble_for_blend
from web.hockey_pred_model import is_hockey_league
from web.cbb_pred_model import is_cbb_league
from web.mlb_pred_model import is_mlb_league
from web.wnba_pred_model import is_wnba_league
from web.league_profiles import is_soccer_league


def apply_ensemble_ml(
    result: dict[str, Any],
    league: str,
    *,
    consensus_spread: float | None = None,
    home_moneyline: int | None = None,
    away_moneyline: int | None = None,
    draw_moneyline: int | None = None,
) -> dict[str, Any]:
    """Replace unified probabilities with the per-sport ensemble ML head when available."""
    league = league.lower()
    if (
        is_hockey_league(league)
        or is_cbb_league(league)
        or is_wnba_league(league)
        or is_mlb_league(league)
        or is_soccer_league(league)
    ):
        return result
    if not ensemble_model_available(league):
        return result

    preds = predict_ensemble_for_blend(
        result,
        league,
        consensus_spread=consensus_spread,
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
        draw_moneyline=draw_moneyline,
    )
    if not preds:
        return result

    updated = dict(result)
    updated["ensemble_ml"] = preds
    updated["blend_mode"] = "ensemble_ml"
    updated["algorithm"] = "Unified (EnsembleML)"

    if is_soccer_league(league):
        home_p = float(preds["home_win_probability"])
        draw_p = float(preds["draw_probability"])
        away_p = float(preds["away_win_probability"])
        updated["home_win_probability"] = round(home_p, 2)
        updated["draw_probability"] = round(draw_p, 2)
        updated["away_win_probability"] = round(away_p, 2)
        updated["blended_home_win_probability"] = round(home_p, 2)
        total, win_prob = home_win_prob_to_total_score(home_p)
        updated["total_score"] = round(total, 2)
        updated["win_probability"] = round(win_prob, 2)
        updated["favorite_side"] = "home" if total <= 0 else "away"
        return updated

    home_prob = float(preds["home_win_probability"])
    total, win_prob = home_win_prob_to_total_score(home_prob)
    updated["blended_home_win_probability"] = round(home_prob, 2)
    updated["total_score"] = round(total, 2)
    updated["win_probability"] = round(win_prob, 2)
    updated["favorite_side"] = "home" if total <= 0 else "away"
    if home_moneyline is not None and away_moneyline is not None:
        updated["market_decorrelated"] = True
    elif consensus_spread is not None:
        updated["market_decorrelated"] = True

    if is_spread_league(league):
        margin = preds.get("predicted_home_margin")
        if margin is not None:
            # Ensemble margin is score diff (positive = home favored); spread picks use
            # book convention (negative = home favored), same as sport layer margins.
            updated["home_spread_margin"] = round(-float(margin), 2)
        else:
            updated["home_spread_margin"] = round(model_home_margin(total, league), 2)

    return updated
