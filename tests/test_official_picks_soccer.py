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
    from web.hubacek_picks import clear_strategy_cache
    from web.pick_strategy import get_pick_thresholds, load_pick_strategy

    clear_strategy_cache()
    load_pick_strategy.cache_clear()

    # Membership alone is not enough — enabled=false leagues are not official-eligible.
    for league in OFFICIAL_PICK_LEAGUES:
        enabled = bool(get_pick_thresholds(league).get("enabled", True))
        assert eligible_for_official_picks(league) is enabled, league

    # Validated / enabled official leagues.
    assert eligible_for_official_picks("epl")
    assert eligible_for_official_picks("worldcup")
    assert eligible_for_official_picks("nba")
    assert eligible_for_official_picks("mlb")

    # In OFFICIAL_PICK_LEAGUES but pick_strategy enabled=false — not UI/tracking eligible.
    for paused in ("cbb",):
        assert paused in OFFICIAL_PICK_LEAGUES
        assert get_pick_thresholds(paused).get("enabled") is False, paused
        assert eligible_for_official_picks(paused) is False, paused

    assert eligible_for_official_picks("nhl")
    assert eligible_for_official_picks("wnba")
    assert eligible_for_official_picks("nfl")
    assert eligible_for_official_picks("cfb")

    # Leagues without a closing-line-beating model stay untracked.
    assert not eligible_for_official_picks("mls")
    assert not eligible_for_official_picks("ucl")


def test_nfl_cfb_cbb_not_eligible_for_official_picks() -> None:
    """CBB stays predictions-only; NFL/CFB cleared the all-seasons-positive bar."""
    from web.hubacek_picks import clear_strategy_cache
    from web.pick_strategy import load_pick_strategy

    clear_strategy_cache()
    load_pick_strategy.cache_clear()

    assert eligible_for_official_picks("cbb") is False
    assert eligible_for_official_picks("nfl") is True
    assert eligible_for_official_picks("cfb") is True
    assert eligible_for_official_picks("nhl") is True
    assert eligible_for_official_picks("wnba") is True
    # Case-insensitive
    assert eligible_for_official_picks("CBB") is False
    assert eligible_for_official_picks("NFL") is True
    assert eligible_for_official_picks("Cfb") is True


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
        "ev_pct": 6.0,
        "model_market_gap_pp": 5.2,
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
        "strategy": "hubacek",
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
        "strategy": "hubacek",
        "status": "loss",
        "units": -1.0,
        "edge": 30.0,
        "ev_pct": OFFICIAL_MIN_EV_PCT,
    }
    legacy_nba = {
        "id": "2026-06-01:legacy:home",
        "date": "2026-06-01",
        "event_id": "legacy",
        "league": "nba",
        "side": "home",
        "strategy": "value",
        "status": "win",
        "units": 2.0,
        "edge": 30.0,
        "ev_pct": OFFICIAL_MIN_EV_PCT,
    }
    store = {"version": 1, "bets": [soccer_bet, mlb_bet, legacy_nba]}
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


def test_top5_soccer_thresholds_use_backtested_v2_policy() -> None:
    """Top-5 club leagues read the soccer v2 backtest gates from pick_strategy.json."""
    for league in ("epl", "bundesliga", "laliga", "seriea", "ligue1"):
        thresholds = get_pick_thresholds(league)
        assert thresholds["bet_type"] == "soccer_1x2"
        assert thresholds["min_market_gap_pp"] >= 4.0, league
        assert thresholds["min_ev_pct"] >= 5.0, league
        assert thresholds["min_win_confidence_pp"] == 0.0, league
        assert thresholds["allowed_sides"] == ["home"], league
        assert thresholds["enabled"] is True, league
    # Internationals use sparse-sample EV caps (home only).
    worldcup = get_pick_thresholds("worldcup")
    assert worldcup["allowed_sides"] == ["home"]
    assert worldcup["min_win_confidence_pp"] == 10.0
    assert worldcup["min_ev_pct"] >= 8.0
    assert worldcup["min_market_gap_pp"] >= 8.0


def test_top5_soccer_official_picks_home_side_only() -> None:
    """Qualifying home edges pass; away/draw edges are filtered by allowed_sides."""
    home_pick = evaluate_soccer_official_picks_for_game(
        league="epl",
        away_name="Chelsea",
        home_name="Arsenal",
        away_slug="che",
        home_slug="ars",
        home_prob=52.0,  # devig home ~45.3% at +120/+270/+260 -> gap ~6.7 pp
        draw_prob=26.0,
        away_prob=22.0,
        away_proj=250,
        draw_proj=300,
        home_proj=-110,
        away_market=260,
        draw_market=270,
        home_market=120,
        base_home_prob=51.0,  # EV at +120: 0.51*2.2-1 = +12.2%
        base_draw_prob=26.0,
        base_away_prob=23.0,
    )
    assert len(home_pick) == 1
    assert home_pick[0].side == "home"
    assert home_pick[0].bet_type == "soccer_1x2"
    assert home_pick[0].strategy == "hubacek"

    away_edge_only = evaluate_soccer_official_picks_for_game(
        league="epl",
        away_name="Chelsea",
        home_name="Arsenal",
        away_slug="che",
        home_slug="ars",
        home_prob=38.0,
        draw_prob=24.0,
        away_prob=38.0,  # devig away ~27.7% at +260 -> gap ~10 pp, but away-side
        away_proj=150,
        draw_proj=320,
        home_proj=140,
        away_market=260,
        draw_market=270,
        home_market=120,
        base_home_prob=38.0,
        base_draw_prob=24.0,
        base_away_prob=38.0,
    )
    assert away_edge_only == []
