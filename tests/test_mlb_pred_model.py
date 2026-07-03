"""MLB RunCast prediction model unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.mlb_efficiency import margin_to_home_win_prob  # noqa: E402
from web.mlb_pred_model import (  # noqa: E402
    build_mlb_model,
    is_mlb_league,
    predict_matchup_from_mlb_model,
    run_mlb_pred_model,
)
from web.mlb_runs import simulate_home_win_probability  # noqa: E402


def _sample_dated_games() -> list[tuple[str, tuple[str, str, str, str, int, int]]]:
    rows = [
        ("a", "b", "A", "B", 5, 3),
        ("a", "c", "A", "C", 6, 2),
        ("a", "d", "A", "D", 4, 3),
        ("b", "c", "B", "C", 2, 5),
        ("b", "d", "B", "D", 3, 4),
        ("c", "d", "C", "D", 1, 6),
        ("c", "a", "C", "A", 2, 7),
        ("b", "a", "B", "A", 1, 8),
        ("c", "b", "C", "B", 4, 3),
        ("d", "a", "D", "A", 2, 5),
        ("a", "b", "A", "B", 7, 2),
        ("b", "a", "B", "A", 3, 6),
        ("a", "c", "A", "C", 5, 4),
        ("c", "a", "C", "A", 2, 6),
        ("b", "c", "B", "C", 4, 5),
        ("c", "b", "C", "B", 3, 4),
        ("d", "b", "D", "B", 2, 5),
        ("a", "b", "A", "B", 6, 1),
        ("a", "c", "A", "C", 5, 3),
        ("b", "c", "B", "C", 4, 4),
        ("c", "a", "C", "A", 1, 7),
        ("b", "a", "B", "A", 2, 5),
        ("c", "b", "C", "B", 3, 4),
        ("d", "c", "D", "C", 2, 6),
        ("a", "b", "A", "B", 8, 2),
        ("b", "a", "B", "A", 3, 5),
    ]
    return [(f"2024-04-{10 + i % 18:02d}", row) for i, row in enumerate(rows)]


def test_is_mlb_league() -> None:
    assert is_mlb_league("mlb")
    assert not is_mlb_league("npb")


def test_margin_to_home_win_prob() -> None:
    assert margin_to_home_win_prob(0.0) == 50.0
    assert margin_to_home_win_prob(1.0) > 50.0
    assert margin_to_home_win_prob(-1.0) < 50.0


def test_simulate_home_win_probability_favors_home() -> None:
    home_favored = simulate_home_win_probability(5.5, 3.5, seed=1)
    away_favored = simulate_home_win_probability(3.5, 5.5, seed=1)
    assert home_favored > away_favored


def test_build_mlb_model_favors_stronger_team() -> None:
    model = build_mlb_model(_sample_dated_games(), "2024-06-01")
    assert model is not None
    strong = predict_matchup_from_mlb_model(model, "a", "d")
    weak = predict_matchup_from_mlb_model(model, "d", "a")
    assert strong is not None and weak is not None
    assert float(strong["home_win_probability"]) > float(weak["home_win_probability"])


def test_mlb_runcast_algorithm_name() -> None:
    dated = _sample_dated_games()
    with patch("web.mlb_pred_model.get_mlb_pred_context") as mock_ctx:
        model = build_mlb_model(dated, "2024-06-01")
        mock_ctx.return_value = model
        with patch("web.mlb_pred_model.pitcher_matchup_margin", return_value=0.2):
            payload = run_mlb_pred_model("mlb", "06-01-2024", "a", "b")
    assert payload is not None
    assert payload["algorithm"] == "MLBRunCast"
    assert payload["source"] == "run-sim-xgb-calibrated"
    assert "mlb_pick_signals" in payload
