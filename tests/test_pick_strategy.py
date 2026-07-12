"""Official pick strategy (backtest-tuned spread vs moneyline routing)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import (  # noqa: E402
    kelly_fraction,
    pick_profit_score,
)
from web.pick_strategy import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    _closing_market_fields,
    _evaluate_backtest_pick,
    grade_moneyline_bet,
    grade_spread_bet,
    official_bet_type,
    simulate_market_moneylines,
    simulate_market_spread,
    simulate_market_threeway,
)


def test_official_bet_type_by_sport() -> None:
    assert official_bet_type("nba") == "spread"
    assert official_bet_type("cbb") == "spread"
    assert official_bet_type("nfl") == "spread"
    assert official_bet_type("cfb") == "spread"
    assert official_bet_type("nhl") == "moneyline"
    assert official_bet_type("mlb") == "moneyline"
    assert official_bet_type("epl") == "soccer_1x2"


def test_pick_thresholds_cbb_nfl_cfb() -> None:
    from web.hubacek_picks import clear_strategy_cache
    from web.pick_strategy import get_pick_thresholds, load_pick_strategy

    clear_strategy_cache()
    load_pick_strategy.cache_clear()

    cbb = get_pick_thresholds("cbb")
    assert cbb["bet_type"] == "spread"
    assert cbb["min_spread_cover_gap_pp"] == 7.0
    assert cbb["min_spread_point_edge"] == 5.0
    assert cbb["min_spread_confidence_pp"] == 5.0
    assert cbb["min_ev_pct"] == 2.0

    nfl = get_pick_thresholds("nfl")
    assert nfl["bet_type"] == "spread"
    assert nfl["min_spread_cover_gap_pp"] == 5.5
    assert nfl["min_spread_point_edge"] == 3.0
    assert nfl["min_spread_confidence_pp"] == 5.0
    assert nfl["min_ev_pct"] == 2.5

    cfb = get_pick_thresholds("cfb")
    assert cfb["bet_type"] == "spread"
    assert cfb["min_spread_cover_gap_pp"] == 6.5
    assert cfb["min_spread_point_edge"] == 4.0
    assert cfb["min_spread_confidence_pp"] == 5.0
    assert cfb["min_ev_pct"] == 2.5


def test_disabled_leagues_produce_no_official_picks() -> None:
    """NFL/CFB failed walk-forward validation; CBB paused — enabled flag enforced."""
    from web.hubacek_picks import clear_strategy_cache
    from web.league_profiles import OFFICIAL_PICK_LEAGUES, eligible_for_official_picks
    from web.pick_strategy import (
        evaluate_official_picks_for_game,
        get_pick_thresholds,
        load_pick_strategy,
    )

    clear_strategy_cache()
    load_pick_strategy.cache_clear()

    for league in ("nfl", "cfb", "cbb"):
        assert league in OFFICIAL_PICK_LEAGUES
        assert get_pick_thresholds(league)["enabled"] is False, league
        assert eligible_for_official_picks(league) is False, league
        assert eligible_for_official_picks(league.upper()) is False, league
        picks = evaluate_official_picks_for_game(
            league=league,
            away_name="Away Team",
            home_name="Home Team",
            away_slug="away-team",
            home_slug="home-team",
            total_score=-70.0,
            win_probability=70.0,
            blended={
                "blended_home_win_probability": 70.0,
                "market_decorrelated": True,
                "home_spread_margin": -9.0,
            },
            away_market=200,
            home_market=-240,
            consensus_spread=-3.0,
            away_spread_odds=-110,
            home_spread_odds=-110,
        )
        assert picks == [], league

    # Validated leagues stay enabled and official-eligible.
    assert get_pick_thresholds("mlb")["enabled"] is True
    assert get_pick_thresholds("nba")["enabled"] is True
    assert eligible_for_official_picks("mlb") is True
    assert eligible_for_official_picks("nba") is True


def test_grade_spread_home_covers() -> None:
    assert grade_spread_bet("home", 110, 100, -5.5) == "win"
    assert grade_spread_bet("home", 105, 100, -5.5) == "loss"
    assert grade_spread_bet("away", 105, 100, -5.5) == "win"


def test_grade_moneyline() -> None:
    assert grade_moneyline_bet("home", 3, 2) == "win"
    assert grade_moneyline_bet("away", 3, 2) == "loss"
    assert grade_moneyline_bet("home", 2, 2) == "push"


def test_simulate_market_helpers() -> None:
    spread = simulate_market_spread(-8.0, "nba")
    assert spread != 0.0
    away_ml, home_ml = simulate_market_moneylines(62.0)
    assert away_ml != home_ml
    assert abs(away_ml) >= 100 and abs(home_ml) >= 100
    assert "min_edge" in DEFAULT_THRESHOLDS


def test_simulate_market_moneylines_near_even_stay_valid_american() -> None:
    """Vig near 50/50 must not emit invalid |odds| < 100."""
    for home_prob in (48.0, 49.0, 50.0, 50.5, 51.0, 52.0):
        away_ml, home_ml = simulate_market_moneylines(home_prob)
        assert abs(away_ml) >= 100, (home_prob, away_ml)
        assert abs(home_ml) >= 100, (home_prob, home_ml)


def test_evaluate_backtest_pick_uses_real_spread_when_provided() -> None:
    thresholds = {
        **DEFAULT_THRESHOLDS,
        "min_edge": 0.0,
        "min_ev_pct": 0.0,
        "min_spread_point_edge": 0.0,
        "min_profit_score": -999.0,
        "min_kelly_pct": 0.0,
    }
    simulated = _evaluate_backtest_pick(
        league="nba",
        bet_type="spread",
        blended_home=85.0,
        model_margin=-15.0,
        power_margin=-12.0,
        power_home=80.0,
        home_goals=105,
        away_goals=100,
        thresholds=thresholds,
    )
    real_line = _evaluate_backtest_pick(
        league="nba",
        bet_type="spread",
        blended_home=85.0,
        model_margin=-15.0,
        power_margin=-12.0,
        power_home=80.0,
        home_goals=105,
        away_goals=100,
        thresholds=thresholds,
        market_spread=-1.5,
        home_spread_odds=-110,
        away_spread_odds=-110,
    )
    assert simulated is not None
    assert real_line is not None
    assert real_line[1] == "win"


def test_kelly_and_profit_score_positive_ev() -> None:
    kelly = kelly_fraction(55.0, -110)
    assert kelly > 0
    score = pick_profit_score(model_prob_pct=55.0, american_odds=-110, edge=20.0)
    assert score > 0


def test_simulate_market_threeway_renormalizes_nonsimplex_bases() -> None:
    """Power-home mixed with model draw/away must not invent non-simplex books."""
    # Same shape as _evaluate_backtest_pick when power_home is set:
    # market_home=60, market_draw=26, market_away=22 from blended home → sum 108.
    away_ml, draw_ml, home_ml = simulate_market_threeway(
        52.0,
        26.0,
        22.0,
        market_home=60.0,
        market_draw=26.0,
        market_away=22.0,
    )
    # After renormalize+shrink, fair moneylines should still be valid American.
    for odds in (away_ml, draw_ml, home_ml):
        assert abs(odds) >= 100
    # Home base is largest after scale → shortest home price among the three.
    assert home_ml < away_ml
    assert home_ml < draw_ml
