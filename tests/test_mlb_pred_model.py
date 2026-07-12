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
from web.bet_advisor import official_pick_binary_probs  # noqa: E402
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


def test_mlb_runcast_decorrelates_from_moneyline_not_spread() -> None:
    dated = _sample_dated_games()
    with patch("web.mlb_pred_model.get_mlb_pred_context") as mock_ctx:
        model = build_mlb_model(dated, "2024-06-01")
        mock_ctx.return_value = model
        with patch("web.mlb_pred_model.pitcher_matchup_margin", return_value=0.0):
            payload = run_mlb_pred_model(
                "mlb",
                "06-01-2024",
                "a",
                "b",
                market_spread=-1.5,
                home_ml=-140,
                away_ml=120,
            )
    assert payload is not None
    assert payload["market_decorrelated"] is True
    assert payload["market_decorrelation_source"] == "moneyline"
    pre = float(payload["pre_decorrelation_home_win_probability"])
    final = float(payload["home_win_probability"])
    assert final != pre
    assert payload["home_win_probability"] != payload["pre_decorrelation_home_win_probability"]


def test_mlb_runcast_invalid_ml_falls_back_to_spread_decorrelation() -> None:
    """Garbage ML must not block spread-based Hubáček decorrelation."""
    dated = _sample_dated_games()
    with patch("web.mlb_pred_model.get_mlb_pred_context") as mock_ctx:
        model = build_mlb_model(dated, "2024-06-01")
        mock_ctx.return_value = model
        with patch("web.mlb_pred_model.pitcher_matchup_margin", return_value=0.0):
            payload = run_mlb_pred_model(
                "mlb",
                "06-01-2024",
                "a",
                "b",
                market_spread=-1.5,
                home_ml=-50,  # |odds| < 100 → reject
                away_ml=120,
            )
    assert payload is not None
    assert payload["market_decorrelated"] is True
    assert payload["market_decorrelation_source"] == "spread"
    assert float(payload["home_win_probability"]) != float(
        payload["pre_decorrelation_home_win_probability"]
    )


def test_official_pick_binary_probs_ignores_spread_decor_for_mlb() -> None:
    away, home = official_pick_binary_probs(
        {
            "blended_home_win_probability": 61.0,
            "total_score": -61.0,
            "baseball_pred": {
                "market_decorrelated": True,
                "market_decorrelation_source": "spread",
                "pre_decorrelation_home_win_probability": 61.0,
                "home_win_probability": 61.5,
            },
        },
        -61.0,
        league="mlb",
        away_market=130,
        home_market=-150,
        consensus_spread=-1.5,
    )
    assert home != 61.5
    assert away == 100.0 - home


def test_run_mlb_v2_keeps_zero_predicted_margin(monkeypatch) -> None:
    """A true pick'em margin of 0.0 must not be replaced by the logit fallback."""
    import web.mlb_pred_model as mlb

    monkeypatch.setattr(
        "web.mlb_v2.live.predict_matchup_v2",
        lambda *_a, **_k: {
            "home_win_probability": 50.0,
            "predicted_margin": 0.0,
            "model_version": "v2",
            "home_probable_pitcher": None,
            "away_probable_pitcher": None,
        },
    )
    out = mlb._run_mlb_v2("mlb", "07-12-2026", "nyy", "bos")
    assert out is not None
    assert float(out["predicted_margin"]) == 0.0
    # Without market spread, disagreement stays off — margin itself is the contract.
    assert out["mlb_pick_signals"]["disagreement_runs"] is None


def test_runcast_xgb_blend_keeps_margin_aligned_with_runs() -> None:
    """XGB win% blend must not invent a logit margin that disagrees with run totals."""
    import numpy as np
    import pytest

    class _FakeXgb:
        def predict_proba(self, _x):
            return np.array([[0.2, 0.8]])

    model = {
        "sigma": 1.5,
        "xgb_model": _FakeXgb(),
        "efficiency": object(),
        "last_game_date": {"nyy": "07-01-2026", "bos": "07-01-2026"},
        "team_game_counts": {"nyy": 80, "bos": 80},
        "hca": 0.1,
        "cutoff_date": "07-12-2026",
    }
    with patch(
        "web.mlb_pred_model.raw_prob_from_runs_sim",
        return_value=(5.0, 4.0, 1.0, 58.0),
    ), patch(
        "web.mlb_pred_model.build_matchup_features",
        return_value={},
    ), patch(
        "web.mlb_pred_model.features_to_vector",
        return_value=np.zeros(5),
    ):
        out = predict_matchup_from_mlb_model(
            model, "nyy", "bos", game_date="07-12-2026"
        )
    assert out is not None
    assert float(out["predicted_home_runs"]) == 5.0
    assert float(out["predicted_away_runs"]) == 4.0
    assert float(out["predicted_margin"]) == pytest.approx(1.0)
    # Win% may move with XGB, but margin stays with the sim runs.
    assert float(out["home_win_probability"]) != 58.0


def test_mlb_xgb_training_skips_ties_not_cast_to_away_win() -> None:
    """Tied games must not enter XGB labels — dtype=int turns 0.5 into 0."""
    dated = _sample_dated_games()
    # Force a walk-forward row that is a true tie.
    dated.append(("2024-05-20", ("a", "b", "A", "B", 3, 3)))

    captured: dict[str, list] = {}

    def _capture(x_rows, y_rows):
        captured["y"] = list(y_rows)
        captured["n"] = len(x_rows)
        return None

    with patch("web.mlb_pred_model._train_xgb_classifier", side_effect=_capture):
        with patch("web.mlb_pred_model.fit_best_calibrator", return_value=None):
            build_mlb_model(dated, "2024-06-01")

    assert "y" in captured
    assert 0.5 not in captured["y"]
    assert all(label in (0.0, 1.0) for label in captured["y"])
    # Sample rows include a 4-4 earlier; that and the appended 3-3 must be excluded.
    assert captured["n"] == len(captured["y"])
