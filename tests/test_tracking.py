"""Tracking service unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.tracking_service import (  # noqa: E402
    build_tracking_response,
    calculate_units,
    grade_bet,
    record_from_slate,
)


def test_calculate_units() -> None:
    assert calculate_units(1, 140, "win") == 1.4
    assert calculate_units(1, 140, "loss") == -1
    assert calculate_units(1, -110, "push") == 0


def test_record_and_grade() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [
            {
                "side": "home",
                "team_name": "Pirates",
                "team_slug": "pittsburgh-pirates",
                "strategy": "value",
                "strategy_label": "Value bet",
                "confidence": "medium",
                "edge": 12,
                "model_projection": 120,
                "market_odds": 141,
                "win_probability": 55,
                "reason": "Edge",
                "league": "mlb",
                "league_name": "MLB",
                "event_id": "401815712",
                "matchup": "Dodgers @ Pirates",
            }
        ],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert len(store["bets"]) == 1
    graded = grade_bet(store["bets"][0], 2, 5)
    assert graded["status"] == "win"
    response = build_tracking_response({"version": 1, "bets": [graded]})
    assert response["summary"]["wins"] == 1


if __name__ == "__main__":
    test_calculate_units()
    test_record_and_grade()
    print("test_tracking.py: all tests passed")
