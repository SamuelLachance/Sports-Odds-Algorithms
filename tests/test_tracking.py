"""Tracking service unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import MIN_RECOMMENDED_EDGE  # noqa: E402
from web.tracking_service import (  # noqa: E402
    build_tracking_response,
    calculate_units,
    grade_bet,
    prune_below_min_edge,
    record_from_slate,
)


def _sample_pick(*, edge: float = MIN_RECOMMENDED_EDGE, event_id: str = "401815712") -> dict:
    return {
        "side": "home",
        "team_name": "Pirates",
        "team_slug": "pittsburgh-pirates",
        "strategy": "value",
        "strategy_label": "Value bet",
        "confidence": "medium",
        "edge": edge,
        "model_projection": 120,
        "market_odds": 141,
        "win_probability": 55,
        "reason": "Edge",
        "league": "mlb",
        "league_name": "MLB",
        "event_id": event_id,
        "matchup": "Dodgers @ Pirates",
    }


def test_calculate_units() -> None:
    assert calculate_units(1, 140, "win") == 1.4
    assert calculate_units(1, 140, "loss") == -1
    assert calculate_units(1, -110, "push") == 0


def test_record_and_grade() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_sample_pick()],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert len(store["bets"]) == 1
    graded = grade_bet(store["bets"][0], 2, 5)
    assert graded["status"] == "win"
    response = build_tracking_response({"version": 1, "bets": [graded]})
    assert response["summary"]["wins"] == 1


def test_rejects_sub_min_edge() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_sample_pick(edge=39)],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert store["bets"] == []


def test_accepts_min_edge() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_sample_pick(edge=MIN_RECOMMENDED_EDGE)],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert len(store["bets"]) == 1
    assert store["bets"][0]["edge"] == MIN_RECOMMENDED_EDGE


def test_ignores_game_recommendations_not_in_recommended() -> None:
    """Per-game recommendations must not be tracked unless listed in recommended_bets."""
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [],
        "games": [
            {
                "league": "mlb",
                "league_name": "MLB",
                "event_id": "401815712",
                "matchup": {"away": {"name": "Dodgers"}, "home": {"name": "Pirates"}},
                "start_time": "2026-06-11T23:40Z",
                "recommendations": [_sample_pick(edge=60)],
            }
        ],
    }
    store = record_from_slate(store, slate)
    assert store["bets"] == []


def _spread_pick(*, edge: float = MIN_RECOMMENDED_EDGE, event_id: str = "401859967") -> dict:
    return {
        "side": "home",
        "team_name": "Spurs",
        "team_slug": "san-antonio-spurs",
        "strategy": "value",
        "strategy_label": "Value bet",
        "confidence": "medium",
        "edge": edge,
        "model_projection": 60,
        "market_odds": -108,
        "win_probability": 72,
        "reason": "Spread edge",
        "bet_type": "spread",
        "spread_line": -5.5,
        "spread_odds": -108,
        "consensus_spread": -5.5,
        "consensus_odds": -108,
        "model_margin": 8.0,
        "league": "nba",
        "league_name": "NBA",
        "event_id": event_id,
        "matchup": "Knicks @ Spurs",
    }


def test_grade_spread_cover_win() -> None:
    bet = {
        "side": "home",
        "bet_type": "spread",
        "consensus_spread": -5.5,
        "spread_odds": -108,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 98, 110)
    assert graded["status"] == "win"


def test_grade_spread_push() -> None:
    bet = {
        "side": "home",
        "bet_type": "spread",
        "consensus_spread": -6.0,
        "spread_odds": -110,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 100, 106)
    assert graded["status"] == "push"
    assert graded["units"] == 0.0


def test_grade_spread_away_cover() -> None:
    bet = {
        "side": "away",
        "bet_type": "spread",
        "consensus_spread": -5.5,
        "spread_odds": -112,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 102, 105)
    assert graded["status"] == "win"


def test_record_spread_bet_fields() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_spread_pick()],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert len(store["bets"]) == 1
    bet = store["bets"][0]
    assert bet["bet_type"] == "spread"
    assert bet["consensus_spread"] == -5.5
    assert bet["spread_line"] == -5.5


def test_grade_soccer_draw_bet_win() -> None:
    bet = {
        "side": "draw",
        "bet_type": "moneyline",
        "league": "epl",
        "market_odds": 250,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 1, 1)
    assert graded["status"] == "win"
    assert graded["units"] == 2.5


def test_grade_soccer_draw_bet_loss() -> None:
    bet = {
        "side": "draw",
        "bet_type": "moneyline",
        "league": "epl",
        "market_odds": 250,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 0, 2)
    assert graded["status"] == "loss"


def test_grade_soccer_home_ml_loses_on_draw() -> None:
    bet = {
        "side": "home",
        "bet_type": "moneyline",
        "league": "epl",
        "market_odds": -120,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 1, 1)
    assert graded["status"] == "loss"


def test_grade_mlb_moneyline_pushes_on_tie() -> None:
    bet = {
        "side": "home",
        "bet_type": "moneyline",
        "league": "mlb",
        "market_odds": -120,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 5, 5)
    assert graded["status"] == "push"


def test_prune_below_min_edge() -> None:
    store = {
        "version": 1,
        "bets": [
            _sample_pick(edge=39, event_id="401815712"),
            _sample_pick(edge=30, event_id="401815713"),
            _sample_pick(edge=MIN_RECOMMENDED_EDGE, event_id="401815714"),
            _sample_pick(edge=MIN_RECOMMENDED_EDGE + 10, event_id="401815715"),
        ],
    }
    # record_from_slate expects full bet shape; prune works on stored bets
    store["bets"] = [
        {
            "id": f"2026-06-11:{p['event_id']}:{p['side']}",
            "date": "2026-06-11",
            "event_id": p["event_id"],
            "side": p["side"],
            "edge": p["edge"],
            "status": "pending",
            "units": 0.0,
            "stake_units": 1.0,
        }
        for p in store["bets"]
    ]
    pruned = prune_below_min_edge(store)
    edges = [b["edge"] for b in pruned["bets"]]
    assert 30 not in edges
    assert 39 not in edges
    assert all(e >= MIN_RECOMMENDED_EDGE for e in edges)
    assert len(pruned["bets"]) == 2


if __name__ == "__main__":
    test_calculate_units()
    test_record_and_grade()
    test_rejects_sub_min_edge()
    test_accepts_min_edge()
    test_ignores_game_recommendations_not_in_recommended()
    test_grade_spread_cover_win()
    test_grade_spread_push()
    test_grade_spread_away_cover()
    test_record_spread_bet_fields()
    test_grade_soccer_draw_bet_win()
    test_grade_soccer_draw_bet_loss()
    test_grade_soccer_home_ml_loses_on_draw()
    test_grade_mlb_moneyline_pushes_on_tie()
    test_prune_below_min_edge()
    print("test_tracking.py: all tests passed")
