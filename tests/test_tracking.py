"""Tracking service unit tests."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hubacek_picks import HUBACEK_MIN_WIN_CONFIDENCE_PP  # noqa: E402
from web.tracking_service import (  # noqa: E402
    _bet_units,
    _fetch_event_result,
    _scoreboard_dates_for_bet,
    _summarize_bets,
    build_tracking_response,
    calculate_units,
    grade_bet,
    grade_pending,
    load_store,
    prune_below_min_edge,
    record_from_slate,
    save_store,
    stake_units_from_kelly,
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


def test_calculate_units_pending_does_not_pay() -> None:
    """Pending / unknown results must not fall through as wins."""
    assert calculate_units(1, 140, "pending") == 0.0
    assert calculate_units(1, -110, "pending") == 0.0
    assert calculate_units(1, 140, "foo") == 0.0  # type: ignore[arg-type]


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


def test_empty_store_and_push_only_roi_is_null() -> None:
    """Empty seasons / push-only windows must not invent a flat 0% ROI book."""
    from web.tracking_service import _normalize_store, _summarize_bets

    empty = _summarize_bets([])
    assert empty["roi_percent"] is None
    assert empty["staked_units"] == 0.0
    assert empty["bets"] == 0
    assert empty["record"] == "0-0"

    push_only = _summarize_bets(
        [{"status": "push", "units": 0.0, "stake_units": 2.0}]
    )
    assert push_only["roi_percent"] is None
    assert push_only["staked_units"] == 0.0

    response = build_tracking_response(_normalize_store({"version": 1}))
    assert response["tracking_since"] is None
    assert response["all_time"]["roi_percent"] is None
    assert response["yearly"] == []
    assert response["daily"] == []
    note = response.get("note") or ""
    # Do not hardcode a flat ≥2 pp rule — live gates are per-league (MLB 6.7, etc.).
    assert "≥2 pp" not in note
    assert ">=2 pp" not in note
    assert "per-league" in note
    assert "NFL/CFB/CBB" in note
    assert response.get("thresholds_scope") == "global_baseline"


def test_normalize_store_tolerates_corrupt_payload() -> None:
    from web.tracking_service import _normalize_store, record_from_slate

    assert _normalize_store(None)["bets"] == []
    assert _normalize_store({"bets": "nope"})["bets"] == []
    assert _normalize_store({"bets": [1, {"side": "home"}]})["bets"] == [{"side": "home"}]

    # Recording into a missing-bets store must not raise.
    store = record_from_slate({"version": 1}, {"date_label": "2026-07-12", "recommended_bets": []})
    assert store["bets"] == []


def test_calculate_units_guards_bad_odds() -> None:
    assert calculate_units(1, 0, "win") == 1.0
    assert calculate_units("bad", -110, "loss") == -1.0
    assert calculate_units(-5, 150, "win") == 1.5  # negative stake → default 1u
    # Invalid |odds| < 100 (other than EVEN/0) must not invent huge payouts.
    assert calculate_units(1, 50, "win") == 0.0
    assert calculate_units(1, -50, "win") == 0.0
    assert calculate_units(1, "oops", "win") == 0.0
    # Non-finite stake must not poison P&L with ±inf / NaN.
    assert calculate_units(float("inf"), -110, "win") == 100.0 / 110.0
    assert calculate_units(float("nan"), 150, "loss") == -1.0


def test_non_finite_units_do_not_poison_roi() -> None:
    """Corrupt ±inf settled units used to make all-time ROI infinite."""
    assert _bet_units({"units": float("inf")}) == 0.0
    assert _bet_units({"units": float("-inf")}) == 0.0
    summary = _summarize_bets(
        [{"status": "win", "units": float("inf"), "stake_units": 1.0}]
    )
    assert summary["units"] == 0.0
    assert summary["roi_percent"] == 0.0
    # NaN EV must not Kelly-size a full stake via portfolio path.
    assert stake_units_from_kelly(8.0, ev_pct=float("nan")) == 0.25


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


def test_tracking_response_sorts_same_day_by_edge_desc() -> None:
    """Within a calendar day, higher-edge bets should list first."""
    low = _sample_pick()
    low["date"] = "2026-06-11"
    low["event_id"] = "low"
    low["edge"] = 2.0
    high = _sample_pick()
    high["date"] = "2026-06-11"
    high["event_id"] = "high"
    high["edge"] = 8.0
    newer = _sample_pick()
    newer["date"] = "2026-06-12"
    newer["event_id"] = "newer"
    newer["edge"] = 1.0
    response = build_tracking_response({"version": 1, "bets": [low, high, newer]})
    assert [b["event_id"] for b in response["bets"]] == ["newer", "high", "low"]


def test_tracking_response_tolerates_corrupt_edge_for_sort() -> None:
    """String/list edge used to TypeError when mixed with numeric edges."""
    junk = _sample_pick()
    junk["date"] = "2026-06-11"
    junk["event_id"] = "junk"
    junk["edge"] = "high"
    junk["status"] = "win"
    junk["units"] = 1.0
    numeric = _sample_pick()
    numeric["date"] = "2026-06-11"
    numeric["event_id"] = "numeric"
    numeric["edge"] = 8.0
    numeric["status"] = "win"
    numeric["units"] = 1.0
    response = build_tracking_response({"version": 1, "bets": [junk, numeric]})
    assert [b["event_id"] for b in response["bets"]] == ["numeric", "junk"]


def test_tracking_response_excludes_legacy_non_hubacek_rows() -> None:
    """Graded legacy strategy rows stay on disk but must not poison Hubáček ROI."""
    hubacek = _sample_pick()
    hubacek.update(
        {
            "id": "2026-06-11:hub:home",
            "date": "2026-06-11",
            "league": "mlb",
            "status": "win",
            "units": 1.0,
            "stake_units": 1.0,
            "strategy": "hubacek",
        }
    )
    legacy = {
        "id": "2026-06-11:legacy:home",
        "date": "2026-06-11",
        "event_id": "legacy",
        "league": "nba",
        "side": "home",
        "strategy": "value",
        "status": "win",
        "units": 5.0,
        "stake_units": 1.0,
        "edge": 9.0,
    }
    response = build_tracking_response({"version": 1, "bets": [hubacek, legacy]})
    assert [b["event_id"] for b in response["bets"]] == ["401815712"]
    assert response["summary"]["bets"] == 1
    assert response["summary"]["units"] == 1.0
    assert response["all_time"]["units"] == 1.0


def test_stake_units_from_kelly_rejects_bool() -> None:
    """False kelly must not coerce to 0% and fall through to default 1u."""
    from web.tracking_service import stake_units_from_kelly

    assert stake_units_from_kelly(False) == 0.25  # type: ignore[arg-type]
    assert stake_units_from_kelly(True) == 0.25  # type: ignore[arg-type]
    assert stake_units_from_kelly(None) == 1.0


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


def test_pending_bet_closing_freezes_after_start() -> None:
    """Post-tip slate runs must not overwrite closing with live prices."""
    store = {"version": 1, "bets": []}
    pick = {**_sample_pick(), "start_time": _future_start()}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    assert len(store["bets"]) == 1
    assert store["bets"][0]["closing_market_odds"] == 141  # seeded at record

    # Pre-tip refresh still moves the closing snapshot.
    pre_tip = {**_sample_pick(), "start_time": _future_start(), "market_odds": 130}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pre_tip], "games": []}
    )
    assert store["bets"][0]["closing_market_odds"] == 130

    started = {**_sample_pick(), "start_time": _past_start(), "market_odds": 120}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [started], "games": []}
    )
    assert len(store["bets"]) == 1
    bet = store["bets"][0]
    assert bet["market_odds"] == 141  # frozen at record time
    assert bet["closing_market_odds"] == 130  # last pre-tip close, not live 120


def test_pending_closing_does_not_refresh_when_start_time_unparseable() -> None:
    """Without a tip to freeze on, closing must not keep moving (fail closed)."""
    store = {"version": 1, "bets": []}
    pick = {**_sample_pick(), "start_time": "TBD", "market_odds": -110}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    assert store["bets"][0]["closing_market_odds"] == -110

    moved = {**_sample_pick(), "start_time": "TBD", "market_odds": 150}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [moved], "games": []}
    )
    assert store["bets"][0]["closing_market_odds"] == -110
    assert store["bets"][0]["market_odds"] == -110


def test_consensus_closing_ml_rejects_invalid_american() -> None:
    from web.tracking_service import _consensus_closing_ml

    game = {
        "event_id": "1",
        "market": {"consensus_home_ml": 50, "consensus_away_ml": -110},
    }
    assert _consensus_closing_ml(game, "home") is None
    assert _consensus_closing_ml(game, "away") == -110
    even = {"market": {"consensus_home_ml": 0}}
    assert _consensus_closing_ml(even, "home") == 100
    # Float strings from JSON/CSV must still seed CLV closing snapshots.
    floatish = {"market": {"consensus_home_ml": "-115.0"}}
    assert _consensus_closing_ml(floatish, "home") == -115


def test_grade_bet_leaves_ungraded_on_invalid_american_odds() -> None:
    """Invalid |odds| < 100 must not book a 0u win/loss (same as missing juice)."""
    bet = {
        "side": "home",
        "bet_type": "moneyline",
        "league": "mlb",
        "market_odds": 50,
        "stake_units": 1.0,
        "status": "pending",
    }
    graded = grade_bet(bet, 2, 5)
    assert graded.get("status") == "pending"
    assert "units" not in graded or graded.get("units") is None


def test_calculate_units_rejects_bool_odds() -> None:
    """int(False)==0 must not pay EVEN units on a win."""
    from web.tracking_service import calculate_units

    assert calculate_units(1.0, False, "win") == 0.0  # type: ignore[arg-type]
    assert calculate_units(1.0, True, "win") == 0.0  # type: ignore[arg-type]
    assert calculate_units(1.0, 0, "win") == 1.0


def test_grade_bet_even_zero_moneyline_still_grades() -> None:
    bet = {
        "side": "home",
        "bet_type": "moneyline",
        "league": "mlb",
        "market_odds": 0,
        "stake_units": 1.0,
        "status": "pending",
    }
    graded = grade_bet(bet, 2, 5)
    assert graded["status"] == "win"
    assert graded["units"] == 1.0


def test_record_dedupes_across_slate_date_labels() -> None:
    """Same event/side across consecutive slate days must not create a second pending row."""
    store = {"version": 1, "bets": []}
    pick = {**_sample_pick(), "start_time": _future_start()}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    assert len(store["bets"]) == 1
    first_id = store["bets"][0]["id"]

    moved = {**_sample_pick(), "start_time": _future_start(), "market_odds": 155}
    store = record_from_slate(
        store, {"date_label": "2026-07-11", "recommended_bets": [moved], "games": []}
    )
    assert len(store["bets"]) == 1
    bet = store["bets"][0]
    assert bet["id"] == first_id
    assert bet["date"] == "2026-07-10"  # original slate day preserved
    assert bet["market_odds"] == 141  # frozen at first record
    assert bet["closing_market_odds"] == 155


def test_normalize_store_collapses_legacy_duplicate_keys() -> None:
    """Legacy same-event rows must not inflate ROI after normalize/save."""
    from web.tracking_service import _normalize_store, _summarize_bets

    store = _normalize_store(
        {
            "version": 1,
            "bets": [
                {
                    "id": "a",
                    "event_id": "401815712",
                    "side": "home",
                    "bet_type": "moneyline",
                    "status": "pending",
                    "units": 0.0,
                    "stake_units": 1.0,
                    "recorded_at": "2026-07-10T12:00:00+00:00",
                    "league": "mlb",
                },
                {
                    "id": "b",
                    "event_id": "401815712",
                    "side": "home",
                    "bet_type": "moneyline",
                    "status": "win",
                    "units": 1.4,
                    "stake_units": 1.0,
                    "recorded_at": "2026-07-11T12:00:00+00:00",
                    "league": "mlb",
                },
            ],
        }
    )
    assert len(store["bets"]) == 1
    assert store["bets"][0]["id"] == "b"
    summary = _summarize_bets(store["bets"])
    assert summary["bets"] == 1
    assert summary["roi_percent"] == 140.0


def test_record_from_slate_does_not_reopen_settled_when_pending_dupe_present() -> None:
    """Settled wins must stay settled even if a legacy pending twin is still in the file."""
    from unittest.mock import patch

    from web.tracking_service import record_from_slate

    settled = {
        "id": "401815712:home",
        "event_id": "401815712",
        "side": "home",
        "bet_type": "moneyline",
        "status": "win",
        "units": 1.2,
        "stake_units": 1.0,
        "market_odds": -120,
        "recorded_at": "2026-07-11T18:00:00+00:00",
        "league": "mlb",
        "league_name": "MLB",
        "team_name": "Yankees",
        "team_slug": "nyy",
        "matchup": "BOS @ NYY",
        "strategy": "hubacek",
        "ev_pct": 8.0,
        "kelly_pct": 2.0,
    }
    pending_twin = {
        **settled,
        "id": "401815712:home:legacy",
        "status": "pending",
        "units": 0.0,
        "recorded_at": "2026-07-10T12:00:00+00:00",
    }
    # Bypass normalize so both rows exist in the working list; index must prefer settled.
    store = {"version": 1, "bets": [pending_twin, settled]}
    slate = {
        "date_label": "2026-07-12",
        "recommended_bets": [
            {
                "event_id": "401815712",
                "side": "home",
                "bet_type": "moneyline",
                "league": "mlb",
                "league_name": "MLB",
                "team_name": "Yankees",
                "team_slug": "nyy",
                "matchup": "BOS @ NYY",
                "strategy": "hubacek",
                "market_odds": -110,
                "ev_pct": 9.0,
                "kelly_pct": 2.5,
                "start_time": "2099-07-12T23:00:00+00:00",
            }
        ],
    }
    with patch("web.tracking_service.eligible_for_official_picks", return_value=True), patch(
        "web.tracking_service.passes_hubacek_tracked_pick", return_value=True
    ), patch(
        "web.tracking_service._normalize_store",
        side_effect=lambda s: {"version": 1, "bets": list(s.get("bets") or [])},
    ):
        out = record_from_slate(store, slate)
    wins = [b for b in out["bets"] if b.get("event_id") == "401815712"]
    assert any(b.get("status") == "win" for b in wins)
    assert not any(
        b.get("status") == "pending" and b.get("closing_market_odds") == -110 for b in wins
    )


def test_closing_snapshot_prefers_consensus_moneyline() -> None:
    """Multi-book consensus beats the single ESPN price for the closing snapshot."""
    store = {"version": 1, "bets": []}
    slate = {
        "date_label": "2026-06-11",
        "recommended_bets": [{**_sample_pick(), "start_time": _future_start()}],
        "games": [],
    }
    store = record_from_slate(store, slate)

    moved = {**_sample_pick(), "start_time": _future_start(), "market_odds": 120}
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
        "recommended_bets": [{**_sample_pick(), "start_time": _future_start()}],
        "games": [],
    }
    store = record_from_slate(store, slate)

    moved = {**_sample_pick(), "start_time": _future_start(), "market_odds": 120}
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
    gap_pp: float = 10.5,
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


def test_grade_spread_legacy_title_case_side_still_home() -> None:
    """Dedupe can keep side='Home'; must not invert ATS grading to away."""
    bet = {
        "side": "Home",
        "bet_type": "spread",
        "consensus_spread": -5.5,
        "spread_odds": -110,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 98, 110)
    assert graded["status"] == "win"


def test_grade_soccer_draw_title_case_side() -> None:
    bet = {
        "side": "Draw",
        "bet_type": "soccer_1x2",
        "league": "ucl",
        "market_odds": 320,
        "stake_units": 1.0,
    }
    graded = grade_bet(bet, 1, 1)
    assert graded["status"] == "win"


def test_soccer_aet_uses_regulation_linescores() -> None:
    """1X2 must settle on FT, not ESPN board score after extra time."""
    from web.tracking_service import _scores_from_event

    event = {
        "competitions": [
            {
                "status": {"type": {"completed": True, "detail": "FT-AET"}},
                "competitors": [
                    {
                        "homeAway": "away",
                        "score": "1",
                        "linescores": [{"value": 1}, {"value": 0}, {"value": 0}],
                    },
                    {
                        "homeAway": "home",
                        "score": "2",
                        "linescores": [{"value": 1}, {"value": 0}, {"value": 1}],
                    },
                ],
            }
        ]
    }
    away, home, completed = _scores_from_event(event, soccer_regulation=True)
    assert completed is True
    assert away == 1
    assert home == 1


def test_scores_from_event_parses_float_string_board_scores() -> None:
    """ESPN float/string scores must grade; int('1.0') previously failed closed."""
    from web.tracking_service import _scores_from_event

    event = {
        "competitions": [
            {
                "status": {"type": {"completed": True}},
                "competitors": [
                    {"homeAway": "away", "score": {"value": "98.0"}},
                    {"homeAway": "home", "score": 110.0},
                ],
            }
        ]
    }
    away, home, completed = _scores_from_event(event)
    assert completed is True
    assert away == 98
    assert home == 110


def test_soccer_aet_without_linescores_leaves_ungraded() -> None:
    from web.tracking_service import _scores_from_event

    event = {
        "competitions": [
            {
                "status": {"type": {"completed": True, "description": "After Extra Time"}},
                "competitors": [
                    {"homeAway": "away", "score": "1"},
                    {"homeAway": "home", "score": "2"},
                ],
            }
        ]
    }
    away, home, completed = _scores_from_event(event, soccer_regulation=True)
    assert completed is True
    assert away is None
    assert home is None


def test_grade_spread_without_juice_leaves_ungraded() -> None:
    """Missing posted spread juice must not invent −110 for P&L."""
    bet = {
        "side": "home",
        "bet_type": "spread",
        "consensus_spread": -5.5,
        "stake_units": 1.0,
        "status": "pending",
    }
    graded = grade_bet(bet, 98, 110)
    assert graded.get("status") == "pending"
    assert "units" not in graded


def test_grade_spread_does_not_settle_at_moneyline_market_odds() -> None:
    """Spread P&L must not fall back to ``market_odds`` (usually the ML)."""
    from web.tracking_service import _resolve_grading_odds

    bet = {
        "side": "home",
        "bet_type": "spread",
        "consensus_spread": -5.5,
        "spread_odds": None,
        "consensus_odds": None,
        "market_odds": -150,
        "stake_units": 1.0,
        "status": "pending",
    }
    assert _resolve_grading_odds(bet) is None
    graded = grade_bet(bet, 100, 110)
    assert graded.get("status") == "pending"
    assert "units" not in graded


def test_recorded_spread_odds_preserves_even_zero() -> None:
    """ESPN EVEN (0) must not fall through to consensus via falsy `or`."""
    from web.tracking_service import _recorded_and_closing_odds

    rec, close = _recorded_and_closing_odds(
        {
            "bet_type": "spread",
            "spread_odds": 0,
            "consensus_odds": -110,
            "closing_spread_odds": -105,
        }
    )
    # Posted EVEN is selected over consensus, then normalized to +100 for CLV/units.
    assert rec == 100
    assert close == -105


def test_recorded_and_closing_odds_accept_float_strings() -> None:
    """CLV must not drop when JSON/CSV stores American prices as float strings."""
    from web.clv_service import clv_vs_market_pct
    from web.tracking_service import _clv_pct, _recorded_and_closing_odds

    rec, close = _recorded_and_closing_odds(
        {
            "bet_type": "moneyline",
            "market_odds": "150.0",
            "closing_market_odds": "130.0",
        }
    )
    assert rec == 150
    assert close == 130
    assert _clv_pct(
        {"bet_type": "moneyline", "market_odds": "150.0", "closing_market_odds": "130.0"}
    ) == clv_vs_market_pct(150, 130)


def test_grade_and_clv_use_same_spread_odds_priority() -> None:
    """P&L and CLV must settle on the same recorded juice when fields differ."""
    from web.clv_service import clv_vs_market_pct
    from web.tracking_service import _resolve_grading_odds, calculate_units

    bet = {
        "side": "home",
        "bet_type": "spread",
        "league": "nba",
        "consensus_spread": -3.5,
        "spread_odds": -105,
        "consensus_odds": -120,
        "closing_spread_odds": -110,
        "stake_units": 1.0,
        "status": "pending",
    }
    assert _resolve_grading_odds(bet) == -105
    graded = grade_bet(bet, 100, 110)  # home covers -3.5
    assert graded["status"] == "win"
    assert graded["units"] == calculate_units(1.0, -105, "win")
    assert graded["clv_pct"] == clv_vs_market_pct(-105, -110)


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


def test_grade_spread_rejects_bool_nan_and_pk_lines() -> None:
    """False→0.0 / NaN→always-loss / PK→ValueError must not settle wrongly."""
    from web.tracking_service import _grading_spread_line

    assert _grading_spread_line({"side": "home", "consensus_spread": False}) is None
    assert (
        _grading_spread_line({"side": "home", "consensus_spread": float("nan")}) is None
    )
    assert _grading_spread_line({"side": "home", "consensus_spread": "PK"}) == 0.0

    false_bet = {
        "side": "home",
        "bet_type": "spread",
        "consensus_spread": False,
        "spread_odds": -110,
        "stake_units": 1.0,
        "status": "pending",
    }
    assert grade_bet(false_bet, 100, 105)["status"] == "pending"

    nan_bet = {
        "side": "home",
        "bet_type": "spread",
        "consensus_spread": float("nan"),
        "spread_odds": -110,
        "stake_units": 1.0,
        "status": "pending",
    }
    assert grade_bet(nan_bet, 100, 105)["status"] == "pending"

    pk_bet = {
        "side": "home",
        "bet_type": "spread",
        "consensus_spread": "PK",
        "spread_odds": -110,
        "stake_units": 1.0,
    }
    graded = grade_bet(pk_bet, 100, 105)
    assert graded["status"] == "win"


def test_grade_pending_survives_corrupt_spread_line(monkeypatch) -> None:
    """One bad pending bet must not abort grading of the rest of the store."""
    import web.tracking_service as ts

    store = {
        "version": 1,
        "bets": [
            {
                "id": "bad",
                "event_id": "1",
                "league": "nba",
                "side": "home",
                "bet_type": "spread",
                "status": "pending",
                "consensus_spread": "GARBAGE",
                "spread_odds": -110,
                "stake_units": 1,
                "date": "2026-07-12",
            },
            {
                "id": "good",
                "event_id": "2",
                "league": "nba",
                "side": "home",
                "bet_type": "moneyline",
                "status": "pending",
                "market_odds": -110,
                "stake_units": 1,
                "date": "2026-07-12",
            },
        ],
    }
    monkeypatch.setattr(ts, "_fetch_event_result", lambda league, eid: (100, 110))
    out = grade_pending(store)
    by_id = {b["id"]: b for b in out["bets"]}
    assert by_id["bad"]["status"] == "pending"
    assert by_id["good"]["status"] == "win"


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


def test_grade_soccer_1x2_home_loses_on_draw_without_league() -> None:
    """soccer_1x2 must lose on draws even when league is missing/wrong."""
    graded = grade_bet(
        {
            "side": "home",
            "bet_type": "soccer_1x2",
            "market_odds": -120,
            "stake_units": 1.0,
        },
        1,
        1,
    )
    assert graded["status"] == "loss"


def test_grade_bet_leaves_unknown_side_pending() -> None:
    """Corrupt side strings must not settle as an implicit home moneyline."""
    bet = {
        "side": "hame",
        "bet_type": "moneyline",
        "league": "nba",
        "market_odds": -110,
        "stake_units": 1.0,
        "status": "pending",
    }
    graded = grade_bet(bet, 100, 110)
    assert graded.get("status") == "pending"
    assert "graded_at" not in graded or graded.get("graded_at") is None


def test_dedupe_pending_prefers_earlier_opening_odds() -> None:
    from web.tracking_service import _normalize_store

    store = _normalize_store(
        {
            "version": 1,
            "bets": [
                {
                    "event_id": "1",
                    "side": "home",
                    "bet_type": "moneyline",
                    "status": "pending",
                    "market_odds": 150,
                    "recorded_at": "2026-01-01T10:00:00+00:00",
                },
                {
                    "event_id": "1",
                    "side": "home",
                    "bet_type": "moneyline",
                    "status": "pending",
                    "market_odds": 130,
                    "recorded_at": "2026-01-02T10:00:00+00:00",
                },
            ],
        }
    )
    assert len(store["bets"]) == 1
    assert store["bets"][0]["market_odds"] == 150


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
        "recommended_bets": [{**_sample_pick(), "start_time": _future_start()}],
        "games": [],
    }
    store = record_from_slate(store, slate)
    assert store["bets"][0]["market_odds"] == 141

    moved = {**_sample_pick(), "start_time": _future_start(), "market_odds": 120, "ev_pct": 9.9}
    store = record_from_slate(
        store,
        {"date_label": "2026-06-11", "recommended_bets": [moved], "games": []},
    )
    bet = store["bets"][0]
    assert len(store["bets"]) == 1
    assert bet["market_odds"] == 141  # frozen at record time
    assert bet["ev_pct"] == 5.0  # frozen at record time
    assert bet["closing_market_odds"] == 120


def test_pending_closing_snapshot_preserves_prior_when_slate_loses_juice() -> None:
    """A later slate with missing odds must not wipe a good closing CLV snapshot."""
    store = {"version": 1, "bets": []}
    pick = {
        **_sample_pick(),
        "bet_type": "spread",
        "spread_line": -3.5,
        "spread_odds": -105,
        "consensus_spread": -3.5,
        "start_time": _future_start(),
    }
    slate = {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    store = record_from_slate(store, slate)
    # Re-run with juice present so the pending updater seeds closing_* fields.
    store = record_from_slate(store, slate)
    assert store["bets"][0]["closing_spread_odds"] == -105
    assert store["bets"][0]["closing_market_odds"] == 141

    wiped = {
        **pick,
        "market_odds": None,
        "spread_odds": None,
        "consensus_spread": None,
    }
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [wiped], "games": []}
    )
    bet = store["bets"][0]
    assert bet["closing_spread_odds"] == -105
    assert bet["closing_consensus_spread"] == -3.5
    assert bet["closing_market_odds"] == 141


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
    # Explicit correlation haircut shrinks classic quarter-Kelly.
    raw = stake_units_from_kelly(4.0, ev_pct=5.0, correlation_penalty=0.0)
    haircut = stake_units_from_kelly(4.0, ev_pct=5.0, correlation_penalty=0.15)
    assert raw == 1.0
    assert haircut == 0.85
    # Non-numeric EV must not raise (slate recording path).
    assert stake_units_from_kelly(4.0, ev_pct="bad") == 1.0


def test_record_from_slate_applies_correlation_penalty_on_multi_pick_day() -> None:
    """Same-slate multi-pick days must not size at raw quarter-Kelly."""
    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    a = _sample_pick(event_id="a")
    a["kelly_pct"] = 4.0
    a["start_time"] = future
    b = _sample_pick(event_id="b")
    b["kelly_pct"] = 4.0
    b["start_time"] = future
    store = record_from_slate(
        {"version": 1, "bets": []},
        {"date_label": "2026-07-12", "recommended_bets": [a, b], "games": []},
    )
    assert len(store["bets"]) == 2
    assert all(bet["stake_units"] == 0.85 for bet in store["bets"])

    solo = _sample_pick(event_id="solo")
    solo["kelly_pct"] = 4.0
    solo["start_time"] = future
    solo_store = record_from_slate(
        {"version": 1, "bets": []},
        {"date_label": "2026-07-12", "recommended_bets": [solo], "games": []},
    )
    assert solo_store["bets"][0]["stake_units"] == 1.0


def test_correlation_penalty_ignores_gate_failed_companions() -> None:
    """A second recommended row that fails Hubáček must not haircut the survivor."""
    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    good = _sample_pick(event_id="good")
    good["kelly_pct"] = 4.0
    good["start_time"] = future
    failed = _sample_pick(event_id="failed", win_probability=51.0, gap_pp=0.5)
    failed["kelly_pct"] = 4.0
    failed["start_time"] = future
    store = record_from_slate(
        {"version": 1, "bets": []},
        {"date_label": "2026-07-12", "recommended_bets": [good, failed], "games": []},
    )
    assert len(store["bets"]) == 1
    assert store["bets"][0]["stake_units"] == 1.0


def test_pending_closing_rejects_garbage_american_overwrite() -> None:
    """Invalid |odds| < 100 must not wipe a valid pre-tip closing CLV snapshot."""
    store = {"version": 1, "bets": []}
    pick = {**_sample_pick(), "start_time": _future_start(), "market_odds": -120}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    assert store["bets"][0]["closing_market_odds"] == -120

    garbage = {**_sample_pick(), "start_time": _future_start(), "market_odds": 50}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [garbage], "games": []}
    )
    assert store["bets"][0]["closing_market_odds"] == -120
    assert store["bets"][0]["market_odds"] == -120


def test_record_from_slate_skips_invalid_american_odds() -> None:
    """New official rows must never store raw |odds| < 100 garbage juice."""
    store = record_from_slate(
        {"version": 1, "bets": []},
        {
            "date_label": "2026-07-12",
            "recommended_bets": [
                {**_sample_pick(), "start_time": _future_start(), "market_odds": 50}
            ],
            "games": [],
        },
    )
    assert store["bets"] == []

    spread = {
        **_sample_pick(),
        "start_time": _future_start(),
        "bet_type": "spread",
        "league": "nba",
        "spread_odds": 50,
        "spread_line": -3.5,
        "consensus_spread": -3.5,
        "model_market_gap_pp": 10.0,
        "win_probability": 56.0,
        "market_odds": -110,
    }
    store = record_from_slate(
        {"version": 1, "bets": []},
        {"date_label": "2026-07-12", "recommended_bets": [spread], "games": []},
    )
    assert store["bets"] == []


def test_pending_refresh_updates_start_time_when_postponed() -> None:
    """Postponed tip-offs must refresh stored start_time for scoreboard dating."""
    store = {"version": 1, "bets": []}
    early = _future_start()
    pick = {**_sample_pick(), "start_time": early}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [pick], "games": []}
    )
    later = (
        datetime.now(timezone.utc) + timedelta(hours=30)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    moved = {**_sample_pick(), "start_time": later, "market_odds": 130}
    store = record_from_slate(
        store, {"date_label": "2026-07-10", "recommended_bets": [moved], "games": []}
    )
    assert store["bets"][0]["start_time"] == later
    assert store["bets"][0]["closing_market_odds"] == 130


def test_dedupe_collapses_side_case_variants() -> None:
    """Home vs home must not create twin pending rows for the same event."""
    from web.tracking_service import _normalize_store

    store = _normalize_store(
        {
            "version": 1,
            "bets": [
                {
                    "event_id": "1",
                    "side": "Home",
                    "bet_type": "moneyline",
                    "status": "pending",
                    "market_odds": 150,
                    "recorded_at": "2026-01-01T10:00:00+00:00",
                },
                {
                    "event_id": "1",
                    "side": "home",
                    "bet_type": "moneyline",
                    "status": "pending",
                    "market_odds": 130,
                    "recorded_at": "2026-01-02T10:00:00+00:00",
                },
            ],
        }
    )
    assert len(store["bets"]) == 1
    assert store["bets"][0]["market_odds"] == 150


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
            "market_odds": p.get("market_odds", -150),
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
    test_empty_store_and_push_only_roi_is_null()
    test_normalize_store_tolerates_corrupt_payload()
    test_calculate_units_guards_bad_odds()
    test_new_bet_skipped_when_game_already_started()
    test_new_bet_recorded_pre_start_with_flag()
    test_new_bet_recorded_when_start_time_unparseable()
    test_pending_bet_closing_freezes_after_start()
    test_pending_closing_does_not_refresh_when_start_time_unparseable()
    test_grade_bet_leaves_ungraded_on_invalid_american_odds()
    test_grade_bet_even_zero_moneyline_still_grades()
    test_record_dedupes_across_slate_date_labels()
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
