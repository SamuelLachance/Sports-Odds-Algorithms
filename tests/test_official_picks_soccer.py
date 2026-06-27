"""Official picks exclude soccer while predictions remain on the slate."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import MIN_RECOMMENDED_EDGE, eligible_for_official_picks  # noqa: E402
from web.tracking_service import (  # noqa: E402
    _official_tracked_bets,
    build_tracking_response,
    record_from_slate,
)


def _soccer_pick(*, edge: float = MIN_RECOMMENDED_EDGE) -> dict:
    return {
        "side": "home",
        "team_name": "Arsenal",
        "team_slug": "arsenal",
        "strategy": "value",
        "strategy_label": "Value bet",
        "confidence": "medium",
        "edge": edge,
        "model_projection": 120,
        "market_odds": 141,
        "win_probability": 55,
        "reason": "Edge",
        "league": "epl",
        "league_name": "Premier League",
        "event_id": "401999001",
        "matchup": "Chelsea @ Arsenal",
    }


def test_eligible_for_official_picks() -> None:
    assert not eligible_for_official_picks("epl")
    assert not eligible_for_official_picks("mls")
    assert eligible_for_official_picks("nba")
    assert eligible_for_official_picks("mlb")


def test_soccer_game_keeps_model_clears_official_picks() -> None:
    """Soccer slate games keep predictions but expose no official pick fields."""
    picks = [_soccer_pick(edge=55)]
    official = picks if eligible_for_official_picks("epl") else []
    game = {
        "league": "epl",
        "eligible_for_official_picks": eligible_for_official_picks("epl"),
        "model": {
            "threeway": True,
            "home_win_probability": 48.0,
            "draw_probability": 27.0,
            "away_win_probability": 25.0,
            "soccer_pred": {"expected_home_goals": 1.6, "expected_away_goals": 1.1},
        },
        "recommendations": official,
        "top_pick": official[0] if official else None,
    }
    assert game["model"]["threeway"] is True
    assert game["eligible_for_official_picks"] is False
    assert game["recommendations"] == []
    assert game["top_pick"] is None


def test_slate_recommendation_rollup_skips_soccer() -> None:
    soccer_game = {
        "league": "epl",
        "league_name": "Premier League",
        "event_id": "401999001",
        "eligible_for_official_picks": False,
        "recommendations": [_soccer_pick(edge=60)],
        "matchup": {"away": {"name": "Chelsea"}, "home": {"name": "Arsenal"}},
        "start_time": "2026-06-27T19:00Z",
        "model": {"model_agreement": {}},
    }
    mlb_game = {
        "league": "mlb",
        "league_name": "MLB",
        "event_id": "401815712",
        "eligible_for_official_picks": True,
        "recommendations": [
            {
                **_soccer_pick(edge=45),
                "league": "mlb",
                "league_name": "MLB",
                "event_id": "401815712",
            }
        ],
        "matchup": {"away": {"name": "Dodgers"}, "home": {"name": "Pirates"}},
        "start_time": "2026-06-27T23:40Z",
        "model": {"model_agreement": {}},
    }

    recommendations = []
    for game in [soccer_game, mlb_game]:
        if not game.get("eligible_for_official_picks", True):
            continue
        for pick in game.get("recommendations") or []:
            recommendations.append(
                {
                    **pick,
                    "league": game["league"],
                    "event_id": game["event_id"],
                }
            )

    assert len(recommendations) == 1
    assert recommendations[0]["league"] == "mlb"


def test_record_from_slate_skips_soccer() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-27",
        "recommended_bets": [_soccer_pick()],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert store["bets"] == []


def test_tracking_rollups_exclude_soccer() -> None:
    soccer_bet = {
        "id": "2026-06-01:401999001:home",
        "date": "2026-06-01",
        "event_id": "401999001",
        "league": "epl",
        "side": "home",
        "status": "win",
        "units": 1.4,
        "edge": MIN_RECOMMENDED_EDGE,
    }
    mlb_bet = {
        "id": "2026-06-01:401815712:home",
        "date": "2026-06-01",
        "event_id": "401815712",
        "league": "mlb",
        "side": "home",
        "status": "loss",
        "units": -1.0,
        "edge": MIN_RECOMMENDED_EDGE,
    }
    store = {"version": 1, "bets": [soccer_bet, mlb_bet]}
    assert len(_official_tracked_bets(store["bets"])) == 1
    response = build_tracking_response(store)
    assert response["summary"]["bets"] == 1
    assert response["summary"]["record"] == "0-1"


def run_all() -> None:
    test_eligible_for_official_picks()
    test_soccer_game_keeps_model_clears_official_picks()
    test_slate_recommendation_rollup_skips_soccer()
    test_record_from_slate_skips_soccer()
    test_tracking_rollups_exclude_soccer()
    print("test_official_picks_soccer.py: all tests passed")


if __name__ == "__main__":
    run_all()
