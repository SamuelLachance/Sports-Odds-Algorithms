"""CBB Torvik prediction model unit tests."""

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
from web.cbb_efficiency import (  # noqa: E402
    TeamEfficiency,
    expected_margin_from_ratings,
    margin_to_home_win_prob,
)
from web.cbb_pred_model import (  # noqa: E402
    build_cbb_model,
    is_cbb_league,
    predict_matchup_from_cbb_model,
    run_cbb_pred_model,
)
from web.cbb_torvik import TorvikTeamRatings, clear_torvik_cache  # noqa: E402


def _sample_dated_games() -> list[tuple[str, tuple[str, str, str, str, int, int]]]:
    rows = [
        ("a", "b", "A", "B", 78, 72),
        ("a", "c", "A", "C", 80, 70),
        ("a", "d", "A", "D", 75, 68),
        ("b", "c", "B", "C", 70, 74),
        ("b", "d", "B", "D", 72, 71),
        ("c", "d", "C", "D", 68, 73),
        ("c", "a", "C", "A", 65, 82),
        ("b", "a", "B", "A", 66, 85),
        ("c", "b", "C", "B", 76, 70),
        ("d", "a", "D", "A", 64, 79),
        ("a", "b", "A", "B", 81, 73),
        ("b", "a", "B", "A", 67, 84),
        ("a", "c", "A", "C", 79, 71),
        ("c", "a", "C", "A", 63, 86),
        ("b", "c", "B", "C", 69, 75),
        ("c", "b", "C", "B", 74, 72),
        ("d", "b", "D", "B", 71, 69),
        ("a", "b", "A", "B", 83, 74),
        ("a", "c", "A", "C", 77, 72),
        ("b", "c", "B", "C", 71, 76),
        ("c", "a", "C", "A", 62, 88),
        ("b", "a", "B", "A", 65, 87),
        ("c", "b", "C", "B", 73, 71),
        ("d", "c", "D", "C", 70, 77),
        ("a", "b", "A", "B", 80, 75),
        ("b", "a", "B", "A", 68, 83),
    ]
    return [(f"2024-11-{10 + i % 18:02d}", row) for i, row in enumerate(rows)]


def _mock_torvik() -> dict[str, TorvikTeamRatings]:
    return {
        "a": TorvikTeamRatings(adj_o=115.0, adj_d=95.0, adj_t=70.0, barthag=0.92, torvik_name="Team A"),
        "b": TorvikTeamRatings(adj_o=105.0, adj_d=100.0, adj_t=68.0, barthag=0.65, torvik_name="Team B"),
        "c": TorvikTeamRatings(adj_o=98.0, adj_d=102.0, adj_t=67.0, barthag=0.45, torvik_name="Team C"),
        "d": TorvikTeamRatings(adj_o=92.0, adj_d=108.0, adj_t=66.0, barthag=0.30, torvik_name="Team D"),
    }


def test_is_cbb_league() -> None:
    assert is_cbb_league("cbb")
    assert not is_cbb_league("nba")


def test_margin_to_home_win_prob() -> None:
    assert margin_to_home_win_prob(0.0) == 50.0
    assert margin_to_home_win_prob(6.0) > 50.0
    assert margin_to_home_win_prob(-6.0) < 50.0


def test_expected_margin_favors_stronger_team() -> None:
    strong = TorvikTeamRatings(115.0, 95.0, 70.0, 0.9, "Strong")
    weak = TorvikTeamRatings(95.0, 110.0, 68.0, 0.2, "Weak")
    _hs, _as, margin = expected_margin_from_ratings(strong, weak)
    assert margin > 0


def test_decorrelate_from_market() -> None:
    adjusted = decorrelate_from_market(0.70, 0.55, weight=0.12)
    assert adjusted > 0.70
    assert adjusted > 0.55


def test_fit_best_calibrator_picks_method() -> None:
    raw = [60.0, 65.0, 70.0, 55.0, 62.0, 58.0] * 5
    outcomes = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0] * 5
    state = fit_best_calibrator(raw, outcomes)
    calibrated = apply_cbb_calibration(65.0, state)
    assert 1.0 <= calibrated <= 99.0


@patch("web.cbb_pred_model.fetch_torvik_ratings")
def test_build_cbb_model_with_mock_torvik(mock_fetch) -> None:
    clear_torvik_cache()
    mock_fetch.return_value = _mock_torvik()
    model = build_cbb_model(_sample_dated_games(), "11-15-2024", 2025)
    assert model is not None
    assert "calibrator" in model
    assert len(model["team_game_counts"]) >= 4


@patch("web.cbb_pred_model.fetch_torvik_ratings")
def test_predict_matchup_favors_stronger_team(mock_fetch) -> None:
    clear_torvik_cache()
    mock_fetch.return_value = _mock_torvik()
    model = build_cbb_model(_sample_dated_games(), "11-15-2024", 2025)
    assert model is not None
    strong_home = predict_matchup_from_cbb_model(model, "a", "c")
    weak_home = predict_matchup_from_cbb_model(model, "c", "a")
    assert strong_home is not None
    assert weak_home is not None
    assert strong_home["home_win_probability"] > weak_home["home_win_probability"]


@patch("web.cbb_pred_model.fetch_torvik_ratings")
@patch("web.cbb_pred_model.load_cbb_dated_games")
@patch("web.cbb_pred_model.load_league_completed_games")
def test_run_cbb_pred_model_payload(mock_games, mock_dated, mock_torvik) -> None:
    from web.cbb_pred_model import get_cbb_pred_context

    clear_torvik_cache()
    mock_games.return_value = [g for _d, g in _sample_dated_games()]
    mock_dated.return_value = _sample_dated_games()
    mock_torvik.return_value = _mock_torvik()
    get_cbb_pred_context.cache_clear()

    payload = run_cbb_pred_model(
        "cbb",
        "11-15-2024",
        "a",
        "c",
        market_spread=-8.5,
    )
    assert payload is not None
    assert payload["algorithm"] == "CBBTorvik"
    assert payload["source"] == "torvik-efficiency-calibrated"
    assert "cbb_pick_signals" in payload
    assert payload["home_win_probability"] > 50.0


def test_fallback_efficiency_from_games() -> None:
    from web.cbb_efficiency import build_fallback_efficiency

    games = [g for _d, g in _sample_dated_games()]
    ratings = build_fallback_efficiency(games)
    assert "a" in ratings
    assert isinstance(ratings["a"], TeamEfficiency)
    assert ratings["a"].games > 0
