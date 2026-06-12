"""Power model unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.power_model import (  # noqa: E402
    build_power_ratings,
    calc_home_win_probability,
    predict_matchup,
)


def _sample_games() -> list[tuple[str, str, str, str, int, int]]:
    return [
        ("a", "b", "Team A", "Team B", 80, 70),
        ("a", "c", "Team A", "Team C", 75, 72),
        ("b", "c", "Team B", "Team C", 68, 70),
        ("c", "a", "Team C", "Team A", 65, 78),
        ("b", "a", "Team B", "Team A", 60, 82),
        ("c", "b", "Team C", "Team B", 71, 69),
        ("a", "b", "Team A", "Team B", 88, 75),
        ("b", "a", "Team B", "Team A", 70, 85),
    ]


def test_build_power_ratings_assigns_higher_power_to_winner() -> None:
    teams, _games, param = build_power_ratings(_sample_games(), iterations=10)
    assert param is not None
    assert teams["a"].power > teams["b"].power
    assert teams["a"].power > teams["c"].power


def test_calc_home_win_probability_favors_stronger_home() -> None:
    teams, _games, param = build_power_ratings(_sample_games(), iterations=10)
    assert param is not None
    strong_home = calc_home_win_probability(teams["a"].power, teams["c"].power, param)
    weak_home = calc_home_win_probability(teams["c"].power, teams["a"].power, param)
    assert strong_home > 50.0
    assert weak_home < 50.0
    assert strong_home > weak_home


def test_predict_matchup_requires_min_games() -> None:
    teams, _games, param = build_power_ratings(_sample_games(), iterations=10)
    assert param is not None
    result = predict_matchup(teams, param, "a", "c")
    assert result is not None
    assert result["home_win_probability"] > 50


if __name__ == "__main__":
    test_build_power_ratings_assigns_higher_power_to_winner()
    test_calc_home_win_probability_favors_stronger_home()
    test_predict_matchup_requires_min_games()
    print("test_power_model.py: all tests passed")
