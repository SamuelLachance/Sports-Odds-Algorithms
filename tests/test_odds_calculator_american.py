"""Odds_Calculator American-odds pair must price underdogs correctly."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_team_comparison_v2_underdog_odds_not_favorite_mirror() -> None:
    """60% favorite is -150 / +66.67, not -150 / +150."""
    from odds_calculator import Odds_Calculator

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
    assert dog == pytest.approx((100.0 - 60.0) / 60.0 * 100.0)
    assert dog == pytest.approx(66.666666, abs=0.01)
    assert dog != pytest.approx(fav)
