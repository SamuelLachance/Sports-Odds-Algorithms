"""WNBA Elo+XGB prediction model unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.cbb_calibrate import (  # noqa: E402
    apply_cbb_calibration,
    decorrelate_from_market,
    fit_best_calibrator,
)
from web.wnba_efficiency import margin_to_home_win_prob  # noqa: E402
from web.wnba_pred_model import (  # noqa: E402
    build_wnba_model,
    get_wnba_pred_context,
    is_wnba_league,
    predict_matchup_from_wnba_model,
    run_wnba_pred_model,
    wnba_unavailable_reason,
)


def _sample_dated_games() -> list[tuple[str, tuple[str, str, str, str, int, int]]]:
    rows = [
        ("a", "b", "A", "B", 82, 78),
        ("a", "c", "A", "C", 85, 72),
        ("a", "d", "A", "D", 80, 74),
        ("b", "c", "B", "C", 76, 79),
        ("b", "d", "B", "D", 78, 77),
        ("c", "d", "C", "D", 74, 81),
        ("c", "a", "C", "A", 70, 88),
        ("b", "a", "B", "A", 72, 86),
        ("c", "b", "C", "B", 80, 75),
        ("d", "a", "D", "A", 71, 84),
        ("a", "b", "A", "B", 83, 79),
        ("b", "a", "B", "A", 73, 87),
        ("a", "c", "A", "C", 81, 76),
        ("c", "a", "C", "A", 69, 89),
        ("b", "c", "B", "C", 75, 80),
        ("c", "b", "C", "B", 79, 77),
        ("d", "b", "D", "B", 76, 74),
        ("a", "b", "A", "B", 86, 80),
        ("a", "c", "A", "C", 84, 78),
        ("b", "c", "B", "C", 77, 82),
        ("c", "a", "C", "A", 68, 90),
        ("b", "a", "B", "A", 74, 88),
        ("c", "b", "C", "B", 81, 79),
        ("d", "c", "D", "C", 75, 83),
        ("a", "b", "A", "B", 85, 81),
        ("b", "a", "B", "A", 76, 86),
    ]
    return [(f"2025-06-{10 + i % 18:02d}", row) for i, row in enumerate(rows)]


def test_is_wnba_league() -> None:
    assert is_wnba_league("wnba")
    assert not is_wnba_league("nba")


def test_wnba_unavailable_reason_not_wnba() -> None:
    assert "Not a WNBA league" in wnba_unavailable_reason("nba", "06-15-2025", "a", "b")


@patch("web.wnba_pred_model.load_league_completed_games")
def test_wnba_unavailable_reason_insufficient_games(mock_games) -> None:
    mock_games.return_value = []
    reason = wnba_unavailable_reason("wnba", "06-15-2025", "a", "b")
    assert "Insufficient completed games" in reason


def test_margin_to_home_win_prob() -> None:
    assert margin_to_home_win_prob(0.0) == 50.0
    assert margin_to_home_win_prob(5.0) > 50.0
    assert margin_to_home_win_prob(-5.0) < 50.0


def test_decorrelate_from_market() -> None:
    adjusted = decorrelate_from_market(0.68, 0.55, weight=0.12)
    assert adjusted > 0.68
    assert adjusted > 0.55


def test_fit_best_calibrator_picks_method() -> None:
    raw = [60.0, 65.0, 70.0, 55.0, 62.0, 58.0] * 5
    outcomes = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0] * 5
    state = fit_best_calibrator(raw, outcomes)
    calibrated = apply_cbb_calibration(65.0, state)
    assert 1.0 <= calibrated <= 99.0


def test_build_wnba_model_from_sample_games() -> None:
    model = build_wnba_model(_sample_dated_games(), "06-15-2025")
    assert model is not None
    assert "calibrator" in model
    assert len(model["team_game_counts"]) >= 4


def test_predict_matchup_favors_stronger_team() -> None:
    model = build_wnba_model(_sample_dated_games(), "06-15-2025")
    assert model is not None
    strong_home = predict_matchup_from_wnba_model(model, "a", "c")
    weak_home = predict_matchup_from_wnba_model(model, "c", "a")
    assert strong_home is not None
    assert weak_home is not None
    assert strong_home["home_win_probability"] > weak_home["home_win_probability"]


@patch("web.wnba_pred_model.fetch_opening_spread_proxy")
@patch("web.wnba_pred_model.availability_home_shift_pp")
@patch("web.wnba_pred_model.load_wnba_dated_games")
@patch("web.wnba_pred_model.load_league_completed_games")
def test_run_wnba_pred_model_payload(
    mock_games,
    mock_dated,
    mock_avail,
    mock_opening,
) -> None:
    mock_games.return_value = [g for _d, g in _sample_dated_games()]
    mock_dated.return_value = _sample_dated_games()
    mock_avail.return_value = 0.0
    mock_opening.return_value = None
    get_wnba_pred_context.cache_clear()

    payload = run_wnba_pred_model(
        "wnba",
        "06-15-2025",
        "a",
        "c",
        market_spread=-6.5,
    )
    assert payload is not None
    assert payload["algorithm"] == "WNBAEloXGB"
    assert payload["source"] == "elo-efficiency-xgb-calibrated"
    assert "wnba_pick_signals" in payload
    assert payload["home_win_probability"] > 50.0


@patch("web.wnba_pred_model.fetch_opening_spread_proxy", return_value=None)
@patch("web.wnba_pred_model.availability_home_shift_pp")
@patch("web.wnba_pred_model.load_wnba_dated_games")
@patch("web.wnba_pred_model.load_league_completed_games")
def test_wnba_availability_shift_survives_margin_rebuild(
    mock_games, mock_dated, mock_avail, _mock_opening
) -> None:
    import pytest

    mock_games.return_value = [g for _d, g in _sample_dated_games()]
    mock_dated.return_value = _sample_dated_games()
    get_wnba_pred_context.cache_clear()

    mock_avail.return_value = 0.0
    base = run_wnba_pred_model("wnba", "06-15-2025", "a", "c")
    get_wnba_pred_context.cache_clear()
    mock_avail.return_value = 2.0
    shifted = run_wnba_pred_model("wnba", "06-15-2025", "a", "c")
    assert base is not None and shifted is not None
    assert shifted["availability_shift_pp"] == 2.0
    assert shifted["raw_home_win_probability"] == pytest.approx(
        base["raw_home_win_probability"] + 2.0, abs=0.05
    )


def test_wnba_xgb_training_skips_ties_not_cast_to_away_win() -> None:
    """Tied games must not enter XGB labels — dtype=int turns 0.5 into 0."""
    dated = _sample_dated_games()
    dated.append(("2025-05-20", ("a", "b", "A", "B", 80, 80)))

    captured: dict[str, list] = {}

    def _capture(x_rows, y_rows):
        captured["y"] = list(y_rows)
        captured["n"] = len(x_rows)
        return None

    with patch("web.wnba_pred_model._train_xgb_classifier", side_effect=_capture):
        with patch("web.wnba_pred_model.fit_best_calibrator", return_value=None):
            build_wnba_model(dated, "2025-06-01")

    assert "y" in captured
    assert 0.5 not in captured["y"]
    assert all(label in (0.0, 1.0) for label in captured["y"])
    assert captured["n"] == len(captured["y"])
