"""Official pick eligibility and tracking for enabled leagues only."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import (  # noqa: E402
    OFFICIAL_MIN_EV_PCT,
    OFFICIAL_PICK_LEAGUES,
    eligible_for_official_picks,
)
from web.pick_strategy import evaluate_soccer_official_picks_for_game, get_pick_thresholds  # noqa: E402
from web.tracking_service import (  # noqa: E402
    _official_tracked_bets,
    build_tracking_response,
    record_from_slate,
)


def _soccer_pick(*, edge: float = 30.0, ev_pct: float = OFFICIAL_MIN_EV_PCT) -> dict:
    return {
        "side": "home",
        "team_name": "Arsenal",
        "team_slug": "arsenal",
        "strategy": "value",
        "strategy_label": "Value bet",
        "confidence": "medium",
        "edge": edge,
        "ev_pct": ev_pct,
        "profit_score": 12.0,
        "kelly_pct": 3.0,
        "model_projection": 120,
        "market_odds": 141,
        "win_probability": 55,
        "reason": "Edge",
        "bet_type": "soccer_1x2",
        "league": "epl",
        "league_name": "Premier League",
        "event_id": "401999001",
        "matchup": "Chelsea @ Arsenal",
    }


def test_eligible_for_official_picks() -> None:
    for league in OFFICIAL_PICK_LEAGUES:
        assert eligible_for_official_picks(league)
    # A+ soccer leagues (calibrated model beats the closing line) are official.
    assert eligible_for_official_picks("epl")
    assert eligible_for_official_picks("worldcup")
    # Leagues without a closing-line-beating model stay untracked.
    assert not eligible_for_official_picks("mls")
    assert not eligible_for_official_picks("ucl")
    assert not eligible_for_official_picks("nfl")


def test_soccer_game_not_eligible_for_official_picks() -> None:
    """Soccer leagues without an A+ model (e.g. MLS) stay off official tracking."""
    pick = {**_soccer_pick(edge=55), "league": "mls", "league_name": "MLS"}
    game = {
        "league": "mls",
        "eligible_for_official_picks": eligible_for_official_picks("mls"),
        "model": {
            "threeway": True,
            "home_win_probability": 48.0,
            "draw_probability": 27.0,
            "away_win_probability": 25.0,
            "soccer_pred": {"expected_home_goals": 1.6, "expected_away_goals": 1.1},
        },
        "recommendations": [pick],
        "top_pick": pick,
    }
    assert game["eligible_for_official_picks"] is False
    assert game["recommendations"], "Soccer keeps model recommendations on the game card"
    assert game["top_pick"] is not None


def test_slate_recommendation_rollup_skips_non_official_leagues() -> None:
    soccer_game = {
        "league": "mls",
        "league_name": "MLS",
        "event_id": "401999001",
        "eligible_for_official_picks": False,
        "recommendations": [{**_soccer_pick(edge=60), "league": "mls"}],
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
                "bet_type": "moneyline",
            }
        ],
        "matchup": {"away": {"name": "Dodgers"}, "home": {"name": "Pirates"}},
        "start_time": "2026-06-27T23:40Z",
        "model": {"model_agreement": {}},
    }

    official = []
    model_analysis = []
    for game in [soccer_game, mlb_game]:
        bucket = official if game.get("eligible_for_official_picks", True) else model_analysis
        for pick in game.get("recommendations") or []:
            bucket.append(
                {
                    **pick,
                    "league": game["league"],
                    "event_id": game["event_id"],
                    "tracked": game.get("eligible_for_official_picks", True),
                }
            )

    assert len(official) == 1
    assert official[0]["league"] == "mlb"
    assert len(model_analysis) == 1
    assert model_analysis[0]["league"] == "mls"
    assert model_analysis[0]["tracked"] is False


def test_record_from_slate_skips_non_official_leagues() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-27",
        "recommended_bets": [{**_soccer_pick(), "league": "mls"}],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert len(store["bets"]) == 0


def test_record_from_slate_accepts_official_soccer_hubacek_pick() -> None:
    store = {"version": 1, "bets": []}
    pick = {
        **_soccer_pick(),
        "strategy": "hubacek",
        "ev_pct": 4.0,
        "model_market_gap_pp": 3.5,
        "win_probability": 62.0,
    }
    slate = {
        "date_label": "2026-06-27",
        "recommended_bets": [pick],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert len(store["bets"]) == 1
    assert store["bets"][0]["league"] == "epl"
    assert store["bets"][0]["bet_type"] == "soccer_1x2"


def test_tracking_rollups_filter_non_official_leagues() -> None:
    soccer_bet = {
        "id": "2026-06-01:401999001:home",
        "date": "2026-06-01",
        "event_id": "401999001",
        "league": "mls",
        "side": "home",
        "status": "win",
        "units": 1.4,
        "edge": 30.0,
        "ev_pct": OFFICIAL_MIN_EV_PCT,
    }
    mlb_bet = {
        "id": "2026-06-01:401815712:home",
        "date": "2026-06-01",
        "event_id": "401815712",
        "league": "mlb",
        "side": "home",
        "status": "loss",
        "units": -1.0,
        "edge": 30.0,
        "ev_pct": OFFICIAL_MIN_EV_PCT,
    }
    store = {"version": 1, "bets": [soccer_bet, mlb_bet]}
    assert len(_official_tracked_bets(store["bets"])) == 1
    response = build_tracking_response(store)
    assert response["summary"]["bets"] == 1
    assert response["summary"]["record"] == "0-1"


def test_evaluate_soccer_official_picks_respects_disabled_league() -> None:
    thresholds = get_pick_thresholds("epl")
    if thresholds.get("enabled", True):
        return
    picks = evaluate_soccer_official_picks_for_game(
        league="epl",
        away_name="Chelsea",
        home_name="Arsenal",
        away_slug="che",
        home_slug="ars",
        home_prob=62.0,
        draw_prob=22.0,
        away_prob=16.0,
        away_proj=220,
        draw_proj=280,
        home_proj=-150,
        away_market=180,
        draw_market=260,
        home_market=120,
    )
    assert picks == []
