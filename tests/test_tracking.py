"""Tracking service unit tests."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hubacek_picks import HUBACEK_MIN_WIN_CONFIDENCE_PP  # noqa: E402
from web.tracking_service import (  # noqa: E402
    _fetch_event_result,
    _scoreboard_dates_for_bet,
    build_tracking_response,
    calculate_units,
    grade_bet,
    grade_pending,
    load_store,
    prune_below_min_edge,
    record_from_slate,
    save_store,
)


def _sample_pick(
    *,
    edge: float = 30.0,
    ev_pct: float = 5.0,
    gap_pp: float = 7.0,
    win_probability: float = 72.0,
    event_id: str = "401815712",
) -> dict:
    return {
        "side": "home",
        "team_name": "Pirates",
        "team_slug": "pittsburgh-pirates",
        "strategy": "hubacek",
        "strategy_label": "Hubáček spot",
        "confidence": "medium",
        "edge": edge,
        "ev_pct": ev_pct,
        "model_market_gap_pp": gap_pp,
        "model_projection": 120,
        "market_odds": 141,
        "win_probability": win_probability,
        "reason": "Hubáček decorrelation gap",
        "league": "mlb",
        "league_name": "MLB",
        "event_id": event_id,
        "matchup": "Dodgers @ Pirates",
    }


def test_calculate_units() -> None:
    assert calculate_units(1, 140, "win") == 1.4
    assert calculate_units(1, 140, "loss") == -1
    assert calculate_units(1, -110, "push") == 0
    assert abs(calculate_units(1, -110, "win") - (100 / 110)) < 1e-9


def test_roi_excludes_pushes_from_denominator() -> None:
    from web.tracking_service import _summarize_bets

    summary = _summarize_bets(
        [
            {"status": "win", "units": 0.91},
            {"status": "loss", "units": -1.0},
            {"status": "push", "units": 0.0},
        ]
    )
    assert summary["roi_percent"] == round(-0.09 / 2 * 100, 2)


def test_roi_is_stake_weighted() -> None:
    """A single 3u loss at 3u staked is −100% ROI, not −300%."""
    from web.tracking_service import _summarize_bets

    summary = _summarize_bets([{"status": "loss", "units": -3.0, "stake_units": 3.0}])
    assert summary["roi_percent"] == -100.0
    assert summary["staked_units"] == 3.0


def test_roi_stake_weighted_mixed_stakes() -> None:
    """ROI divides settled units by staked units of decided bets only."""
    from web.tracking_service import _summarize_bets

    summary = _summarize_bets(
        [
            {"status": "win", "units": 1.41, "stake_units": 1.0},
            {"status": "loss", "units": -3.0, "stake_units": 3.0},
            {"status": "push", "units": 0.0, "stake_units": 2.0},
            {"status": "pending", "units": 0.0, "stake_units": 1.5},
        ]
    )
    assert summary["staked_units"] == 4.0  # pushes/pending excluded
    assert summary["roi_percent"] == round((1.41 - 3.0) / 4.0 * 100, 2)


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


def test_rejects_low_confidence() -> None:
    """Confidence bar applies in leagues that keep one (MLB's is 0 by backtest)."""
    low_conf_nhl = {**_sample_pick(win_probability=55), "league": "nhl", "league_name": "NHL"}
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [low_conf_nhl],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert store["bets"] == []


def test_rejects_non_positive_ev() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_sample_pick(ev_pct=0)],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert store["bets"] == []


def test_accepts_hubacek_pick() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_sample_pick()],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert len(store["bets"]) == 1
    assert store["bets"][0]["strategy"] == "hubacek"


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
                "recommendations": [_sample_pick(ev_pct=60)],
            }
        ],
    }
    store = record_from_slate(store, slate)
    assert store["bets"] == []


def _future_start() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%MZ")


def _past_start() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%MZ")


def test_new_bet_skipped_when_game_already_started() -> None:
    """A pick first seen post-kickoff was never actionable — do not record it."""
    store = {"version": 1, "bets": []}
    pick = {**_sample_pick(), "start_time": _past_start()}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    assert store["bets"] == []


def test_new_bet_recorded_pre_start_with_flag() -> None:
    store = {"version": 1, "bets": []}
    pick = {**_sample_pick(), "start_time": _future_start()}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    assert len(store["bets"]) == 1
    assert store["bets"][0]["recorded_pre_start"] is True


def test_new_bet_recorded_when_start_time_unparseable() -> None:
    """Defensive parsing: a bad start_time must not block recording."""
    store = {"version": 1, "bets": []}
    pick = {**_sample_pick(), "start_time": "TBD"}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    assert len(store["bets"]) == 1


def test_pending_bet_still_updated_after_start() -> None:
    """Post-start slate runs still refresh closing odds on existing pending bets."""
    store = {"version": 1, "bets": []}
    pick = {**_sample_pick(), "start_time": _future_start()}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    assert len(store["bets"]) == 1

    started = {**_sample_pick(), "start_time": _past_start(), "market_odds": 120}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [started], "games": []}
    )
    assert len(store["bets"]) == 1
    bet = store["bets"][0]
    assert bet["market_odds"] == 141  # frozen at record time
    assert bet["closing_market_odds"] == 120
    assert bet["closing_source"] == "espn"


def test_closing_snapshot_prefers_consensus_moneyline() -> None:
    """Multi-book consensus beats the single ESPN price for the closing snapshot."""
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_sample_pick()],
        "games": [],
    }
    store = record_from_slate(store, slate)

    moved = _sample_pick()
    moved["market_odds"] = 120
    game = {
        "event_id": "401815712",
        "market": {"consensus_home_ml": 112, "consensus_away_ml": -125},
    }
    store = record_from_slate(
        store,
        {"date_label": "2026-06-11", "recommended_bets": [moved], "games": [game]},
    )
    bet = store["bets"][0]
    assert bet["closing_market_odds"] == 112  # home-side consensus, not ESPN 120
    assert bet["closing_source"] == "consensus"


def test_closing_snapshot_falls_back_to_espn_without_consensus() -> None:
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_sample_pick()],
        "games": [],
    }
    store = record_from_slate(store, slate)

    moved = _sample_pick()
    moved["market_odds"] = 120
    game = {"event_id": "401815712", "market": {"provider": "espn"}}
    store = record_from_slate(
        store,
        {"date_label": "2026-06-11", "recommended_bets": [moved], "games": [game]},
    )
    bet = store["bets"][0]
    assert bet["closing_market_odds"] == 120
    assert bet["closing_source"] == "espn"


def _spread_pick(
    *,
    edge: float = 30.0,
    ev_pct: float = 5.0,
    gap_pp: float = 5.0,
    win_probability: float = 72.0,
    event_id: str = "401859967",
) -> dict:
    return {
        "side": "home",
        "team_name": "Spurs",
        "team_slug": "san-antonio-spurs",
        "strategy": "hubacek",
        "strategy_label": "Hubáček spot",
        "confidence": "medium",
        "edge": edge,
        "ev_pct": ev_pct,
        "model_market_gap_pp": gap_pp,
        "model_projection": 60,
        "market_odds": -108,
        "win_probability": win_probability,
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


def test_scoreboard_dates_use_start_time_not_record_date() -> None:
    bet = {
        "date": "2026-06-12",
        "start_time": "2026-06-14T00:30:00Z",
    }
    dates = _scoreboard_dates_for_bet(bet)
    iso_dates = [d.isoformat() for d in dates]
    assert "2026-06-13" in iso_dates
    assert "2026-06-12" in iso_dates


def test_fetch_event_result_grades_completed_nba_final() -> None:
    scores = _fetch_event_result("nba", "401859967")
    assert scores == (94, 90)


def test_grade_pending_resolves_stale_event_by_id() -> None:
    store = {
        "version": 1,
        "bets": [
            {
                "id": "2026-06-12:401859967:away:spread",
                "date": "2026-06-12",
                "event_id": "401859967",
                "league": "nba",
                "side": "away",
                "bet_type": "spread",
                "consensus_spread": -5.5,
                "spread_odds": -112,
                "stake_units": 1.0,
                "status": "pending",
                "units": 0.0,
                "start_time": "2026-06-14T00:30:00Z",
            }
        ],
    }
    graded_store = grade_pending(store)
    bet = graded_store["bets"][0]
    assert bet["status"] in {"win", "loss", "push"}
    assert bet.get("final_score")


def test_recorded_odds_frozen_and_closing_snapshot_updates() -> None:
    """Re-running the slate must not rewrite recorded odds, only the closing snapshot."""
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [_sample_pick()],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert store["bets"][0]["market_odds"] == 141

    moved = _sample_pick()
    moved["market_odds"] = 120
    moved["ev_pct"] = 9.9
    store = record_from_slate(
        store,
        {"date_label": "2026-06-11", "recommended_bets": [moved], "games": []},
    )
    bet = store["bets"][0]
    assert len(store["bets"]) == 1
    assert bet["market_odds"] == 141  # frozen at record time
    assert bet["ev_pct"] == 5.0  # frozen at record time
    assert bet["closing_market_odds"] == 120


def test_clv_computed_at_grading() -> None:
    from web.clv_service import clv_vs_market_pct

    bet = {
        "side": "home",
        "bet_type": "moneyline",
        "league": "mlb",
        "market_odds": 141,
        "closing_market_odds": 120,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 2, 5)
    assert graded["status"] == "win"
    # Implied-prob CLV is the industry standard (positive = better than close).
    assert graded["clv_pct"] == clv_vs_market_pct(141, 120)
    # Legacy decimal-payout ratio kept for backward compatibility.
    assert graded["clv_payout_pct"] == round((2.41 / 2.20 - 1.0) * 100.0, 2)


def test_stake_units_quarter_kelly() -> None:
    from web.tracking_service import stake_units_from_kelly

    assert stake_units_from_kelly(None) == 1.0
    assert stake_units_from_kelly(0.0) == 1.0
    assert stake_units_from_kelly(4.0) == 1.0
    assert stake_units_from_kelly(0.5) == 0.25
    assert stake_units_from_kelly(25.0) == 3.0


def test_prune_keeps_graded_bets() -> None:
    """Graded bets are immutable history even if thresholds change."""
    graded = {
        "id": "2026-06-11:401:home",
        "date": "2026-06-11",
        "event_id": "401",
        "side": "home",
        "strategy": "value",
        "ev_pct": 0.5,
        "win_probability": 51,
        "status": "loss",
        "units": -1.0,
        "stake_units": 1.0,
    }
    store = {"version": 1, "bets": [graded]}
    pruned = prune_below_min_edge(store)
    assert len(pruned["bets"]) == 1


def test_prune_below_hubacek_threshold() -> None:
    qualifying = _sample_pick(event_id="401815714")
    low_conf = _sample_pick(win_probability=55, event_id="401815712")
    legacy = _sample_pick(event_id="401815713")
    legacy["strategy"] = "value"
    store = {
        "version": 1,
        "bets": [qualifying, low_conf, legacy],
    }
    store["bets"] = [
        {
            "id": f"2026-06-11:{p['event_id']}:{p['side']}",
            "date": "2026-06-11",
            "event_id": p["event_id"],
            "side": p["side"],
            "edge": p["edge"],
            "ev_pct": p["ev_pct"],
            "strategy": p["strategy"],
            "win_probability": p.get("win_probability"),
            "model_market_gap_pp": p.get("model_market_gap_pp"),
            "status": "pending",
            "units": 0.0,
            "stake_units": 1.0,
        }
        for p in store["bets"]
    ]
    pruned = prune_below_min_edge(store)
    assert len(pruned["bets"]) == 1
    assert pruned["bets"][0]["event_id"] == "401815714"


if __name__ == "__main__":
    test_calculate_units()
    test_roi_excludes_pushes_from_denominator()
    test_roi_is_stake_weighted()
    test_roi_stake_weighted_mixed_stakes()
    test_new_bet_skipped_when_game_already_started()
    test_new_bet_recorded_pre_start_with_flag()
    test_new_bet_recorded_when_start_time_unparseable()
    test_pending_bet_still_updated_after_start()
    test_closing_snapshot_prefers_consensus_moneyline()
    test_closing_snapshot_falls_back_to_espn_without_consensus()
    test_record_and_grade()
    test_rejects_low_confidence()
    test_rejects_non_positive_ev()
    test_accepts_hubacek_pick()
    test_ignores_game_recommendations_not_in_recommended()
    test_grade_spread_cover_win()
    test_grade_spread_push()
    test_grade_spread_away_cover()
    test_record_spread_bet_fields()
    test_grade_soccer_draw_bet_win()
    test_grade_soccer_draw_bet_loss()
    test_grade_soccer_home_ml_loses_on_draw()
    test_grade_mlb_moneyline_pushes_on_tie()
    test_scoreboard_dates_use_start_time_not_record_date()
    test_fetch_event_result_grades_completed_nba_final()
    test_grade_pending_resolves_stale_event_by_id()
    test_recorded_odds_frozen_and_closing_snapshot_updates()
    test_clv_computed_at_grading()
    test_stake_units_quarter_kelly()
    test_prune_keeps_graded_bets()
    test_prune_below_hubacek_threshold()
    print("test_tracking.py: all tests passed")
