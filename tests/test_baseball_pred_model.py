"""Baseball Elo/Pythagorean prediction model unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.baseball_pred_model import (  # noqa: E402
    build_baseball_model,
    is_baseball_league,
    predict_matchup_from_model,
)


def _sample_games() -> list[tuple[str, str, str, str, int, int]]:
    return [
        ("a", "b", "Team A", "Team B", 6, 3),
        ("a", "c", "Team A", "Team C", 5, 2),
        ("a", "d", "Team A", "Team D", 7, 4),
        ("b", "c", "Team B", "Team C", 2, 5),
        ("b", "d", "Team B", "Team D", 3, 6),
        ("c", "d", "Team C", "Team D", 4, 5),
        ("c", "a", "Team C", "Team A", 1, 8),
        ("b", "a", "Team B", "Team A", 2, 7),
        ("c", "b", "Team C", "Team B", 5, 3),
        ("d", "a", "Team D", "Team A", 2, 6),
        ("a", "b", "Team A", "Team B", 8, 2),
        ("b", "a", "Team B", "Team A", 3, 5),
        ("a", "c", "Team A", "Team C", 6, 1),
        ("c", "a", "Team C", "Team A", 2, 9),
        ("b", "c", "Team B", "Team C", 4, 5),
        ("c", "b", "Team C", "Team B", 3, 4),
        ("d", "b", "Team D", "Team B", 5, 2),
        ("a", "b", "Team A", "Team B", 7, 3),
        ("a", "c", "Team A", "Team C", 5, 2),
        ("b", "c", "Team B", "Team C", 3, 4),
        ("c", "a", "Team C", "Team A", 1, 7),
        ("b", "a", "Team B", "Team A", 2, 6),
        ("c", "b", "Team C", "Team B", 4, 3),
        ("d", "c", "Team D", "Team C", 2, 5),
        ("a", "b", "Team A", "Team B", 6, 4),
        ("b", "a", "Team B", "Team A", 3, 5),
    ]


def test_is_baseball_league() -> None:
    assert is_baseball_league("mlb")
    assert not is_baseball_league("nba")
    assert not is_baseball_league("nhl")


def test_build_baseball_model_favors_stronger_team() -> None:
    model = build_baseball_model(_sample_games(), "mlb")
    assert model is not None
    strong_home = predict_matchup_from_model(model, "a", "c", season=2025)
    weak_home = predict_matchup_from_model(model, "c", "a", season=2025)
    assert strong_home is not None
    assert weak_home is not None
    assert strong_home["home_win_probability"] > 50.0
    assert weak_home["home_win_probability"] < 50.0
    assert strong_home["home_win_probability"] > weak_home["home_win_probability"]


def test_sharp_baseball_algorithm_name() -> None:
    model = build_baseball_model(_sample_games(), "mlb")
    assert model is not None
    payload = predict_matchup_from_model(model, "a", "b", season=2025)
    assert payload is not None
    assert "mlb_api_component" in payload


def test_sharp_baseball_forwards_dh_game_number(monkeypatch) -> None:
    """Doubleheader G2 must not reuse G1 pitcher edge via default game_number=1."""
    from unittest.mock import MagicMock

    from web import baseball_pred_model as bpm

    model = build_baseball_model(_sample_games(), "mlb")
    assert model is not None

    seen: list[int | None] = []

    def _fake_margin(home, away, *, game_date=None, game_number=None):
        seen.append(game_number)
        return 0.4 if game_number == 1 else -0.4

    monkeypatch.setattr(
        "web.mlb_pitcher_edge.pitcher_matchup_margin",
        _fake_margin,
    )
    # Avoid MLB season API noise in this unit test.
    monkeypatch.setattr(
        "web.mlb_stats_api.mlb_api_matchup_edge",
        MagicMock(return_value=None),
    )

    g1 = predict_matchup_from_model(
        model, "a", "b", season=2025, game_date="2025-06-15", game_number=1
    )
    g2 = predict_matchup_from_model(
        model, "a", "b", season=2025, game_date="2025-06-15", game_number=2
    )
    assert g1 is not None and g2 is not None
    assert seen == [1, 2]
    assert g1["pitcher_component"] > 0
    assert g2["pitcher_component"] < 0
    assert g1["predicted_margin"] != g2["predicted_margin"]


if __name__ == "__main__":
    test_is_baseball_league()
    test_build_baseball_model_favors_stronger_team()
    print("test_baseball_pred_model.py: all tests passed")
