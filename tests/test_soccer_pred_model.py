"""Soccer rating model unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import is_soccer_league
from web.soccer_pred_model import (  # noqa: E402
    _normalize_threeway,
    build_soccer_model,
    predict_matchup_from_model,
)


def _sample_soccer_games() -> list[tuple[str, str, str, str, int, int]]:
    return [
        ("a", "b", "Alpha", "Beta", 2, 1),
        ("a", "c", "Alpha", "Charlie", 1, 1),
        ("a", "d", "Alpha", "Delta", 3, 0),
        ("b", "c", "Beta", "Charlie", 0, 2),
        ("b", "d", "Beta", "Delta", 2, 2),
        ("c", "d", "Charlie", "Delta", 1, 0),
        ("c", "a", "Charlie", "Alpha", 0, 1),
        ("b", "a", "Beta", "Alpha", 1, 2),
        ("c", "b", "Charlie", "Beta", 2, 1),
        ("d", "a", "Delta", "Alpha", 0, 2),
        ("a", "b", "Alpha", "Beta", 2, 0),
        ("b", "a", "Beta", "Alpha", 1, 1),
        ("a", "c", "Alpha", "Charlie", 2, 1),
        ("c", "a", "Charlie", "Alpha", 1, 2),
        ("b", "c", "Beta", "Charlie", 1, 3),
        ("c", "b", "Charlie", "Beta", 2, 0),
        ("d", "b", "Delta", "Beta", 1, 1),
        ("a", "b", "Alpha", "Beta", 1, 0),
        ("a", "c", "Alpha", "Charlie", 3, 1),
        ("b", "c", "Beta", "Charlie", 0, 1),
        ("c", "a", "Charlie", "Alpha", 0, 2),
        ("b", "a", "Beta", "Alpha", 1, 2),
        ("c", "b", "Charlie", "Beta", 1, 1),
        ("d", "c", "Delta", "Charlie", 0, 2),
        ("a", "b", "Alpha", "Beta", 2, 1),
        ("b", "a", "Beta", "Alpha", 0, 1),
    ]


def test_is_soccer_league() -> None:
    assert is_soccer_league("epl")
    assert is_soccer_league("mls")
    assert not is_soccer_league("nba")


def test_normalize_threeway_sums_to_100() -> None:
    home, draw, away = _normalize_threeway(40.0, 28.0, 32.0)
    assert abs(home + draw + away - 100.0) < 0.02


def test_build_soccer_model_favors_stronger_home_team() -> None:
    model = build_soccer_model(_sample_soccer_games(), "epl")
    assert model is not None
    strong_home = predict_matchup_from_model(model, "a", "c")
    weak_home = predict_matchup_from_model(model, "c", "a")
    assert strong_home is not None
    assert weak_home is not None
    assert strong_home.home_win_probability > weak_home.home_win_probability
    total = (
        strong_home.home_win_probability
        + strong_home.draw_probability
        + strong_home.away_win_probability
    )
    assert abs(total - 100.0) < 0.05


if __name__ == "__main__":
    test_is_soccer_league()
    test_normalize_threeway_sums_to_100()
    test_build_soccer_model_favors_stronger_home_team()
    print("test_soccer_pred_model.py: all tests passed")
