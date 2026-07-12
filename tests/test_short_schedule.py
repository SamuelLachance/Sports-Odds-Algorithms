"""Short-schedule fixes for international soccer teams with <10 games."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algo import Algo  # noqa: E402
from odds_calculator import Odds_Calculator  # noqa: E402


def _short_schedule_data() -> list[dict]:
    return [
        {
            "year": "2026",
            "dates": ["6-10-2026", "6-15-2026"],
            "other_team": ["cpv", "egy"],
            "home_away": ["away", "home"],
            "game_scores": [[1, 0], [2, 1]],
            "period_scores": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "seasons_used": ["2026"],
            "used_prior_season": False,
        }
    ]


def test_get_avg_points_handles_fewer_than_ten_games() -> None:
    calc = Odds_Calculator("nhl")
    result = calc.get_avg_points(_short_schedule_data())
    assert result["avg_10_games"] == [1.5]
    assert result["avg_other_10_games"] == [0.5]


def test_algo_last_ten_games_handles_short_win_ratio() -> None:
    calc = Odds_Calculator("nhl")
    away = calc.analyze2(["ksa", "saudi-arabia"], ["cpv", "cape-verde"], _short_schedule_data(), "away")
    home = calc.analyze2(["cpv", "cape-verde"], ["ksa", "saudi-arabia"], _short_schedule_data(), "home")
    algo = Algo("nhl")
    result = algo.calculate_V2("6-26-2026", away, home)
    assert "total" in result


def test_get_output_analysis_single_season_one_game() -> None:
    """One season row + odd W-L diff must not IndexError / KeyError."""
    calc = Odds_Calculator("nhl")
    data = [
        {
            "year": "2026",
            "dates": ["6-10-2026"],
            "other_team": ["egy"],
            "home_away": ["home"],
            "game_scores": [[2, 1]],
            "period_scores": [[[1], [0]]],
        }
    ]
    returned = calc.analyze2(["ksa", "ksa"], "egy", data, "home")
    lines = calc.get_output_analysis("", ["ksa", "ksa"], returned, "home")
    season = next(x for x in lines if "Season:" in x)
    ten = next(x for x in lines if "10 Games:" in x)
    assert "n/a" in season
    assert "1-0" in ten


if __name__ == "__main__":
    test_get_avg_points_handles_fewer_than_ten_games()
    test_algo_last_ten_games_handles_short_win_ratio()
    test_get_output_analysis_single_season_one_game()
    print("test_short_schedule.py: all tests passed")
