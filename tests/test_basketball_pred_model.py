"""Basketball matrix prediction model unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.basketball_pred_model import (  # noqa: E402
    build_basketball_model,
    is_basketball_league,
    predict_matchup_from_model,
    soft_impute,
)


def _sample_games() -> list[tuple[str, str, str, str, int, int]]:
    return [
        ("a", "b", "Team A", "Team B", 110, 100),
        ("a", "c", "Team A", "Team C", 105, 98),
        ("a", "d", "Team A", "Team D", 102, 99),
        ("b", "c", "Team B", "Team C", 95, 102),
        ("b", "d", "Team B", "Team D", 97, 101),
        ("c", "d", "Team C", "Team D", 94, 100),
        ("c", "a", "Team C", "Team A", 90, 108),
        ("b", "a", "Team B", "Team A", 88, 112),
        ("c", "b", "Team C", "Team B", 100, 97),
        ("d", "a", "Team D", "Team A", 91, 107),
        ("a", "b", "Team A", "Team B", 115, 101),
        ("b", "a", "Team B", "Team A", 92, 109),
        ("a", "c", "Team A", "Team C", 108, 99),
        ("c", "a", "Team C", "Team A", 91, 111),
        ("b", "c", "Team B", "Team C", 96, 104),
        ("c", "b", "Team C", "Team B", 99, 95),
        ("d", "b", "Team D", "Team B", 98, 96),
        ("a", "b", "Team A", "Team B", 112, 98),
        ("a", "c", "Team A", "Team C", 107, 100),
        ("b", "c", "Team B", "Team C", 94, 101),
        ("c", "a", "Team C", "Team A", 89, 110),
        ("b", "a", "Team B", "Team A", 90, 113),
        ("c", "b", "Team C", "Team B", 98, 96),
        ("d", "c", "Team D", "Team C", 93, 105),
        ("a", "b", "Team A", "Team B", 109, 103),
        ("b", "a", "Team B", "Team A", 93, 108),
    ]


def test_is_basketball_league() -> None:
    assert is_basketball_league("nba")
    assert is_basketball_league("wnba")
    assert is_basketball_league("cbb")
    assert not is_basketball_league("nhl")


def test_soft_impute_preserves_observed() -> None:
    raw = [
        [10.0, 0.0, 5.0],
        [0.0, 8.0, 0.0],
        [5.0, 0.0, 12.0],
    ]
    completed = soft_impute(raw, lambda_=1.0, max_iters=20, max_rank=2)
    assert completed[0][0] == 10.0
    assert completed[1][1] == 8.0
    assert completed[2][2] == 12.0
    assert completed[0][2] != 0.0


def test_build_basketball_model_favors_stronger_team() -> None:
    model = build_basketball_model(_sample_games(), "nba")
    assert model is not None
    strong_home = predict_matchup_from_model(model, "a", "c")
    weak_home = predict_matchup_from_model(model, "c", "a")
    assert strong_home is not None
    assert weak_home is not None
    assert strong_home["home_win_probability"] > 50.0
    assert weak_home["home_win_probability"] < 50.0
    assert strong_home["home_win_probability"] > weak_home["home_win_probability"]


if __name__ == "__main__":
    test_is_basketball_league()
    test_soft_impute_preserves_observed()
    test_build_basketball_model_favors_stronger_team()
    print("test_basketball_pred_model.py: all tests passed")
