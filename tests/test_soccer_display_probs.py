"""Soccer Path A display vs Hubáček pick probability separation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_pred_model import run_soccer_pred_model  # noqa: E402


def test_display_probs_use_raw_not_decorrelated_pick_probs() -> None:
    fake_prediction = type(
        "Pred",
        (),
        {
            "home_win_probability": 40.0,
            "draw_probability": 30.0,
            "away_win_probability": 30.0,
            "expected_home_goals": 1.4,
            "expected_away_goals": 1.2,
            "elo_home": 1500.0,
            "elo_away": 1480.0,
            "pi_expected_gd": 0.1,
            "source": "test",
        },
    )()
    context = {
        "team_keys": ["hom", "awy"],
        "team_game_counts": {"hom": 10, "awy": 10},
        "temperature": 1.0,
    }

    with patch(
        "web.soccer_pred_model.get_soccer_pred_context",
        return_value=context,
    ), patch(
        "web.soccer_pred_model.predict_matchup_from_model",
        return_value=fake_prediction,
    ), patch(
        "web.soccer_pred_model.devig_threeway_from_odds",
        return_value=(60.0, 22.0, 18.0),
    ), patch(
        "web.soccer_pred_model.blend_with_market_cap",
        return_value=(20.0, 25.0, 55.0),
    ), patch(
        "web.soccer_pred_model.build_soccer_pick_signals",
        return_value={},
    ):
        payload = run_soccer_pred_model(
            "epl",
            "7-7-2026",
            "hom",
            "awy",
            home_ml=-150,
            draw_ml=280,
            away_ml=400,
        )

    assert payload is not None
    # Display probs are market-calibrated raw model output (between the raw 40%
    # and the market-implied 60%), never the decorrelated pick probabilities.
    assert 40.0 <= payload["home_win_probability"] <= 60.0
    assert payload["home_win_probability"] != payload["pick_home_win_probability"]
    assert payload["pick_home_win_probability"] == 20.0
    assert payload["pick_away_win_probability"] == 55.0
    assert payload["market_decorrelated"] is True


def test_display_favorite_not_flipped_by_decorrelation() -> None:
    """When raw slightly favors home, decorrelation must not flip displayed favorite."""
    fake_prediction = type(
        "Pred",
        (),
        {
            "home_win_probability": 37.0,
            "draw_probability": 29.0,
            "away_win_probability": 34.0,
            "expected_home_goals": 1.3,
            "expected_away_goals": 1.2,
            "elo_home": 1510.0,
            "elo_away": 1505.0,
            "pi_expected_gd": 0.05,
            "source": "test",
        },
    )()
    context = {
        "team_keys": ["hom", "awy"],
        "team_game_counts": {"hom": 12, "awy": 12},
        "temperature": 1.0,
    }

    with patch(
        "web.soccer_pred_model.get_soccer_pred_context",
        return_value=context,
    ), patch(
        "web.soccer_pred_model.predict_matchup_from_model",
        return_value=fake_prediction,
    ), patch(
        "web.soccer_pred_model.devig_threeway_from_odds",
        return_value=(62.0, 21.0, 17.0),
    ), patch(
        "web.soccer_pred_model.blend_with_market_cap",
        return_value=(29.0, 31.0, 40.0),
    ), patch(
        "web.soccer_pred_model.build_soccer_pick_signals",
        return_value={},
    ):
        payload = run_soccer_pred_model(
            "ligue1",
            "8-16-2026",
            "hom",
            "awy",
            home_ml=-165,
            draw_ml=280,
            away_ml=390,
        )

    assert payload is not None
    assert payload["home_win_probability"] >= payload["away_win_probability"]
    assert payload["pick_away_win_probability"] >= payload["pick_home_win_probability"]
