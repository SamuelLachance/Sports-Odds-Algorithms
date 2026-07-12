"""League readiness gating for baseball and hockey three-layer model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web import league_readiness as readiness_module  # noqa: E402
from web.league_readiness import (  # noqa: E402
    assess_league_readiness,
    assess_mlb_readiness,
    assess_soccer_readiness,
    assess_three_layer_readiness,
    is_league_ready_for_daily_slate,
    uses_three_layer_readiness_gate,
)


def test_hockey_readiness_when_algo_v1_data_available(monkeypatch) -> None:
    games = [
        ("bos", "mtl", "Boston", "Montreal", 4, 2),
        ("mtl", "bos", "Montreal", "Boston", 3, 4),
        ("bos", "tor", "Boston", "Toronto", 5, 3),
        ("tor", "mtl", "Toronto", "Montreal", 2, 3),
        ("bos", "nyr", "Boston", "New York", 4, 4),
        ("nyr", "tor", "New York", "Toronto", 3, 2),
    ] * 2

    monkeypatch.setattr(
        readiness_module,
        "load_league_completed_games",
        lambda *_a, **_k: games,
    )

    for league in ("nhl", "ncaah", "ncaawh"):
        result = readiness_module.assess_hockey_readiness(league, "4-12-2017")
        assert result["ready"] is True
        assert is_league_ready_for_daily_slate(league, "4-12-2017") is True


def test_basketball_leagues_use_matrix_readiness(monkeypatch) -> None:
    games = [
        ("bos", "mia", "Boston", "Miami", 110, 100),
        ("mia", "bos", "Miami", "Boston", 98, 105),
    ] * 12

    monkeypatch.setattr(
        readiness_module,
        "load_league_completed_games",
        lambda *_a, **_k: games,
    )
    monkeypatch.setattr(
        readiness_module,
        "get_basketball_pred_context",
        lambda *_a, **_k: {
            "team_game_counts": {"bos": 10, "mia": 10, "lal": 8, "gsw": 8},
        },
    )

    for league in ("nba", "wnba", "cbb"):
        result = readiness_module.assess_basketball_readiness(league, "6-11-2026")
        assert result["ready"] is True
        assert is_league_ready_for_daily_slate(league, "6-11-2026") is True


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


def test_assess_league_readiness_routes_by_sport(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness_module,
        "assess_hockey_readiness",
        lambda *_a, **_k: {"ready": True, "reason": "Algo V1 ready"},
    )
    monkeypatch.setattr(
        readiness_module,
        "assess_mlb_readiness",
        lambda *_a, **_k: {"ready": False, "reason": "Need more games"},
    )
    assert assess_league_readiness("nhl", "6-11-2026")["ready"] is True
    assert assess_league_readiness("mlb", "6-11-2026")["reason"] == "Need more games"


def test_list_leagues_metadata_matches_live_stacks() -> None:
    from web.league_profiles import list_leagues_metadata

    by_id = {row["id"]: row["description"] for row in list_leagues_metadata()}
    assert "GradientBoost v2" in by_id["nba"]
    assert "BasketballMatrix" in by_id["nba"]
    assert "predictions only" in by_id["nfl"].lower()
    assert "predictions only" in by_id["cbb"].lower()
    assert "SoccerPathA" in by_id["epl"]
    assert "MLBRunCast" in by_id["mlb"]
    assert "Algo V1" in by_id["nhl"]
    assert "Algo V2 + power ratings" not in by_id["nba"]
