"""Tests for hockey Poisson prediction layer."""

from web.hockey_pred_model import (
    TeamMetrics,
    calculate_expected_goals,
    calculate_win_probability,
    predict_matchup_from_model,
)


def test_calculate_expected_goals_home_favorite() -> None:
    home = TeamMetrics("tor", goals_for_pg=3.4, goals_against_pg=2.8, games_played=10)
    away = TeamMetrics("mtl", goals_for_pg=2.9, goals_against_pg=3.3, games_played=10)
    home_xg, away_xg = calculate_expected_goals(home, away)
    assert home_xg > away_xg


def test_calculate_win_probability_sums_near_one() -> None:
    probs = calculate_win_probability(3.2, 2.8)
    assert abs(probs.home_win + probs.away_win - 1.0) < 0.02


def test_predict_matchup_from_model() -> None:
    model = {
        "team_metrics": {
            "bos": TeamMetrics("bos", 3.5, 2.6, 15),
            "mtl": TeamMetrics("mtl", 2.8, 3.2, 15),
        },
        "team_game_counts": {"bos": 15, "mtl": 15},
    }
    result = predict_matchup_from_model(model, "bos", "mtl")
    assert result is not None
    assert 0 < result["home_win_probability"] < 100
    assert result["expected_home_goals"] > result["expected_away_goals"]
