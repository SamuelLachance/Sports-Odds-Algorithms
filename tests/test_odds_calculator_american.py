"""Odds_Calculator American-odds pair must price underdogs correctly."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_team_comparison_v2_underdog_odds_mirror_favorite() -> None:
    """60% favorite is fair -150 / +150 (implied probs sum to 100%)."""
    from odds_calculator import Odds_Calculator
    from web.clv_service import american_to_implied_prob

    calc = Odds_Calculator("nba")
    algo_data = {
        "record_points": 0.0,
        "home_away_points": 0.0,
        "home_away_10_games_points": 0.0,
        "last_10_games_points": 0.0,
        "avg_points": 0.0,
        "avg_points_10_games": 0.0,
        "total": 60.0,
    }
    with (
        patch.object(calc.espn_scraper, "update_data"),
        patch.object(calc.universal, "load_data", return_value={}),
        patch.object(calc, "analyze2", return_value={}),
        patch("odds_calculator.Algo") as algo_cls,
    ):
        algo_cls.return_value.calculate_V2.return_value = algo_data
        lines = calc.team_comparison(
            "Algo_V2",
            ["bos", "Boston"],
            ["mia", "Miami"],
            "1-1-2024",
            "2024",
        )

    fav_line = next(x for x in lines if x.startswith("Favorable team odds:"))
    dog_line = next(x for x in lines if x.startswith("Underdog team odds:"))
    fav = float(re.search(r"-([0-9.]+)$", fav_line).group(1))
    dog = float(re.search(r"\+([0-9.]+)$", dog_line).group(1))
    assert fav == pytest.approx(150.0)
    assert dog == pytest.approx(150.0)
    fav_p = american_to_implied_prob(-int(round(fav)))
    dog_p = american_to_implied_prob(int(round(dog)))
    assert fav_p is not None and dog_p is not None
    assert fav_p + dog_p == pytest.approx(1.0)


def test_team_comparison_v2_clamps_100_pct_favorite() -> None:
    """Algo_V2 total ±100 must not ZeroDivisionError on American odds."""
    from odds_calculator import Odds_Calculator

    calc = Odds_Calculator("nba")
    algo_data = {
        "record_points": 0.0,
        "home_away_points": 0.0,
        "home_away_10_games_points": 0.0,
        "last_10_games_points": 0.0,
        "avg_points": 0.0,
        "avg_points_10_games": 0.0,
        "total": 100.0,
    }
    with (
        patch.object(calc.espn_scraper, "update_data"),
        patch.object(calc.universal, "load_data", return_value={}),
        patch.object(calc, "analyze2", return_value={}),
        patch("odds_calculator.Algo") as algo_cls,
    ):
        algo_cls.return_value.calculate_V2.return_value = algo_data
        lines = calc.team_comparison(
            "Algo_V2",
            ["bos", "Boston"],
            ["mia", "Miami"],
            "1-1-2024",
            "2024",
        )

    fav_line = next(x for x in lines if x.startswith("Favorable team odds:"))
    dog_line = next(x for x in lines if x.startswith("Underdog team odds:"))
    fav = float(re.search(r"-([0-9.]+)$", fav_line).group(1))
    dog = float(re.search(r"\+([0-9.]+)$", dog_line).group(1))
    assert fav == pytest.approx(10000.0)
    assert dog == pytest.approx(0.0)


def test_get_output_analysis_early_season_odd_last10() -> None:
    """Fewer than 10 games yields an odd W−L diff; must not KeyError on bucket map."""
    from odds_calculator import Odds_Calculator

    calc = Odds_Calculator("nba")
    buckets = {str(i): [0, 0] for i in range(-10, 11, 2)}
    returned = {
        "seasonal_records": [[5, 3], [2, 1]],
        "avg_game_points": {
            "avg_game_points": [100, 100],
            "avg_other_game_points": [98, 98],
            "avg_10_games": [100, 100],
            "avg_other_10_games": [98, 98],
        },
        "home_away_record": {
            "home_record": [[3, 1], [2, 1]],
            "home_10_games": [[3, 1], [2, 1]],
            "away_record": [[2, 2], [0, 0]],
            "away_10_games": [[2, 2], [0, 0]],
        },
        # W L W → net +1 (odd); full-window map only has even keys.
        "current_win_ratio": [
            ["1-1-2024", "x", 1],
            ["1-2-2024", "x", -1],
            ["1-3-2024", "x", 1],
        ],
        "10_game_win_ratio": [buckets, dict(buckets)],
        "win_loss_streaks_against": {
            "games_since_last_loss": 0,
            "games_since_last_win": 0,
            "other_team": "mia",
            "games_since_last_loss_home": 0,
            "games_since_last_win_home": 0,
            "games_since_last_loss_away": 0,
            "games_since_last_win_away": 0,
        },
    }
    lines = calc.get_output_analysis("", ["bos", "Boston"], returned, "home")
    assert any("10 Games: 2-1" in line for line in lines)


def test_get_output_analysis_empty_last10_is_zero_zero() -> None:
    """No games played must label 0-0, not the full-window 5-5 bucket."""
    from odds_calculator import Odds_Calculator

    calc = Odds_Calculator("nba")
    buckets = {str(i): [0, 0] for i in range(-10, 11, 2)}
    returned = {
        "seasonal_records": [[0, 0], [0, 0]],
        "avg_game_points": {
            "avg_game_points": [0, 0],
            "avg_other_game_points": [0, 0],
            "avg_10_games": [0, 0],
            "avg_other_10_games": [0, 0],
        },
        "home_away_record": {
            "home_record": [[0, 0], [0, 0]],
            "home_10_games": [[0, 0], [0, 0]],
            "away_record": [[0, 0], [0, 0]],
            "away_10_games": [[0, 0], [0, 0]],
        },
        "current_win_ratio": [],
        "10_game_win_ratio": [buckets, dict(buckets)],
        "win_loss_streaks_against": {
            "games_since_last_loss": 0,
            "games_since_last_win": 0,
            "other_team": "mia",
            "games_since_last_loss_home": 0,
            "games_since_last_win_home": 0,
            "games_since_last_loss_away": 0,
            "games_since_last_win_away": 0,
        },
    }
    lines = calc.get_output_analysis("", ["bos", "Boston"], returned, "home")
    assert any("10 Games: 0-0" in line for line in lines)
    assert not any("10 Games: 5-5" in line for line in lines)
