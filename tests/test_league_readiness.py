"""League readiness gating for baseball and hockey three-layer model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web import league_readiness as readiness_module  # noqa: E402
from web.league_readiness import (  # noqa: E402
    assess_mlb_readiness,
    assess_soccer_readiness,
    assess_three_layer_readiness,
    is_league_ready_for_daily_slate,
    uses_three_layer_readiness_gate,
)


def test_basketball_leagues_skip_readiness_gate() -> None:
    assert uses_three_layer_readiness_gate("nba") is False
    assert is_league_ready_for_daily_slate("nba", "6-11-2026") is True


def test_baseball_leagues_use_readiness_gate() -> None:
    assert uses_three_layer_readiness_gate("mlb") is False
    assert uses_three_layer_readiness_gate("ncaabb") is True


def test_soccer_leagues_use_path_a_readiness() -> None:
    assert uses_three_layer_readiness_gate("epl") is False
    assert uses_three_layer_readiness_gate("mls") is False


def test_soccer_readiness_when_path_a_available(monkeypatch) -> None:
    games = [
        ("ars", "liv", "Arsenal", "Liverpool", 2, 1),
        ("liv", "ars", "Liverpool", "Arsenal", 1, 1),
    ] * 10

    monkeypatch.setattr(
        readiness_module,
        "load_league_completed_games",
        lambda *_a, **_k: games,
    )
    monkeypatch.setattr(
        readiness_module,
        "get_soccer_pred_context",
        lambda *_a, **_k: {
            "team_game_counts": {"ars": 10, "liv": 10, "che": 8, "mun": 8},
        },
    )

    result = assess_soccer_readiness("epl", "6-11-2026")
    assert result["ready"] is True
    assert is_league_ready_for_daily_slate("epl", "6-11-2026") is True


def test_readiness_false_when_insufficient_games(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness_module,
        "load_league_completed_games",
        lambda *_a, **_k: [],
    )
    result = assess_three_layer_readiness("ncaabb", "6-11-2026")
    assert result["ready"] is False
    assert result["game_count"] == 0
    assert "completed games" in result["reason"]


def test_mlb_readiness_when_runcast_available(monkeypatch) -> None:
    games = [
        ("bos", "nyy", "Boston", "New York", 5, 3),
        ("nyy", "bos", "New York", "Boston", 2, 4),
    ] * 12

    monkeypatch.setattr(
        readiness_module,
        "load_league_completed_games",
        lambda *_a, **_k: games,
    )
    monkeypatch.setattr(
        readiness_module,
        "get_mlb_pred_context",
        lambda *_a, **_k: {
            "team_game_counts": {"bos": 10, "nyy": 10, "lad": 8, "sf": 8},
        },
    )

    result = assess_mlb_readiness("mlb", "6-11-2026")
    assert result["ready"] is True
    assert is_league_ready_for_daily_slate("mlb", "6-11-2026") is True


def test_readiness_true_when_power_and_third_layer_available(monkeypatch) -> None:
    games = [
        ("bos", "nyy", "Boston", "New York", 5, 3),
        ("nyy", "bos", "New York", "Boston", 2, 4),
    ] * 12

    monkeypatch.setattr(
        readiness_module,
        "load_league_completed_games",
        lambda *_a, **_k: games,
    )
    monkeypatch.setattr(
        readiness_module,
        "get_league_power_context",
        lambda *_a, **_k: (
            {"bos": object(), "nyy": object(), "lad": object(), "sf": object()},
            games,
            1.5,
        ),
    )
    monkeypatch.setattr(
        readiness_module,
        "get_baseball_pred_context",
        lambda *_a, **_k: {
            "team_game_counts": {"bos": 10, "nyy": 10, "lad": 8, "sf": 8},
        },
    )

    result = assess_three_layer_readiness("ncaabb", "6-11-2026")
    assert result["ready"] is True
    assert result["power"] is True
    assert result["third_layer"] is True
