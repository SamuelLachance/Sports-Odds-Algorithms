"""Official pick strategy (backtest-tuned spread vs moneyline routing)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_official_mlb_picks_include_kelly_after_enrich() -> None:
    """Live official path must attach Kelly — slate recording sizes off kelly_pct."""
    from web.pick_strategy import evaluate_official_picks_for_game

    picks = evaluate_official_picks_for_game(
        league="mlb",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-72.0,
        win_probability=72.0,
        blended={
            "blended_home_win_probability": 72.0,
            "market_decorrelated": True,
            "home_win_probability": 72.0,
            "away_win_probability": 28.0,
        },
        away_market=200,
        home_market=-150,
        consensus_spread=None,
        away_spread_odds=None,
        home_spread_odds=None,
        home_prob=72.0,
        away_prob=28.0,
    )
    assert picks
    assert picks[0].extra.get("kelly_pct") is not None
    assert picks[0].profit_score > 0


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


def test_apply_synthetic_vig_worsens_even_money_price() -> None:
    """Near-even dogs must take juice, not snap back to fair +100."""
    from web.clv_service import american_to_implied_prob
    from web.pick_strategy import _apply_synthetic_vig

    assert _apply_synthetic_vig(-100) == -104
    juiced = _apply_synthetic_vig(100)
    assert abs(juiced) >= 100
    assert american_to_implied_prob(juiced) > american_to_implied_prob(100)
    # Asymmetric pick'em markets should juice both sides.
    away_ml, home_ml = simulate_market_moneylines(50.0)
    assert american_to_implied_prob(away_ml) > 0.5
    assert american_to_implied_prob(home_ml) > 0.5


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


def test_evaluate_backtest_pick_skips_real_spread_without_juice() -> None:
    """Closing line without juice must not invent -110 for walk-forward ROI."""
    thresholds = {
        **DEFAULT_THRESHOLDS,
        "min_edge": 0.0,
        "min_ev_pct": 0.0,
        "min_spread_point_edge": 0.0,
        "min_profit_score": -999.0,
        "min_kelly_pct": 0.0,
    }
    assert (
        _evaluate_backtest_pick(
            league="nba",
            bet_type="spread",
            blended_home=85.0,
            model_margin=-15.0,
            power_margin=-12.0,
            power_home=80.0,
            home_goals=105,
            away_goals=100,
            thresholds=thresholds,
            market_spread=-4.0,
            home_spread_odds=None,
            away_spread_odds=None,
        )
        is None
    )


def test_kelly_and_profit_score_positive_ev() -> None:
    kelly = kelly_fraction(55.0, -110)
    assert kelly > 0
    score = pick_profit_score(model_prob_pct=55.0, american_odds=-110, edge=20.0)
    assert score > 0


def test_simulate_market_threeway_renormalizes_nonsimplex_bases() -> None:
    """Power-home mixed with model draw must complete the power-home simplex."""
    # Correct call shape from _evaluate_backtest_pick:
    # market_away = 100 - power_home - draw (14), not 100 - model_home - draw.
    away_ml, draw_ml, home_ml = simulate_market_threeway(
        52.0,
        26.0,
        22.0,
        market_home=60.0,
        market_draw=26.0,
        market_away=14.0,
    )
    for odds in (away_ml, draw_ml, home_ml):
        assert abs(odds) >= 100
    assert home_ml < away_ml
    assert home_ml < draw_ml

    buggy_away = simulate_market_threeway(
        52.0, 26.0, 22.0, market_home=60.0, market_draw=26.0, market_away=22.0
    )
    assert (away_ml, draw_ml, home_ml) != buggy_away


def test_simulate_market_threeway_clamps_negative_away_mass() -> None:
    """power_home + draw > 100 must not invent a negative away leg / absurd ML."""
    away_ml, draw_ml, home_ml = simulate_market_threeway(
        40.0,
        28.0,
        32.0,
        market_home=95.0,
        market_draw=50.0,
        market_away=-45.0,
    )
    for odds in (away_ml, draw_ml, home_ml):
        assert abs(odds) >= 100
        assert abs(odds) < 5000
    # Away mass was floored; home remains the shortest price of the three.
    assert home_ml < away_ml
    assert home_ml < draw_ml


def test_evaluate_backtest_soccer_passes_power_home_simplex(monkeypatch) -> None:
    """Synthetic soccer market must use 100 - power_home - draw for away mass."""
    captured: dict = {}

    def _capture(home_prob, draw_prob, away_prob, **kwargs):
        captured["kwargs"] = kwargs
        return (-150, 250, -120)

    monkeypatch.setattr("web.pick_strategy.simulate_market_threeway", _capture)
    monkeypatch.setattr(
        "web.pick_strategy.evaluate_soccer_picks",
        lambda **_k: [],
    )
    result = _evaluate_backtest_pick(
        league="epl",
        bet_type="soccer_1x2",
        blended_home=52.0,
        model_margin=0.0,
        power_home=60.0,
        home_goals=1,
        away_goals=0,
        thresholds={**DEFAULT_THRESHOLDS, "min_edge": 999.0},
        home_prob=52.0,
        draw_prob=26.0,
        away_prob=22.0,
    )
    assert result is None  # no picks / gate
    assert captured["kwargs"]["market_home"] == 60.0
    assert captured["kwargs"]["market_draw"] == 26.0
    assert captured["kwargs"]["market_away"] == pytest.approx(14.0)


def test_backtest_roi_excludes_pushes_from_denominator(monkeypatch) -> None:
    """Pushes must not dilute ROI (same rule as tracking_service)."""
    from web.pick_strategy import _backtest_samples

    outcomes = iter(
        [
            (0.91, "win"),
            (-1.0, "loss"),
            (0.0, "push"),
        ]
    )
    monkeypatch.setattr(
        "web.pick_strategy._evaluate_backtest_pick",
        lambda **_k: next(outcomes),
    )
    samples = [
        {
            "blended_home": 55.0,
            "model_margin": 2.0,
            "home_goals": 100,
            "away_goals": 90,
        }
        for _ in range(3)
    ]
    summary = _backtest_samples(
        "nba",
        samples,
        {**DEFAULT_THRESHOLDS, "min_edge": 0.0},
        "spread",
    )
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["pushes"] == 1
    assert summary["bets"] == 3
    # -0.09 / 2 decided bets ≈ -4.5%, not -0.09 / 3 ≈ -3.0%
    assert summary["roi_pct"] == pytest.approx(-4.5, abs=0.05)


def test_simulate_market_threeway_applies_synthetic_vig() -> None:
    """Docstring promises vig; overround must match two-way synthetic books."""
    from web.bet_advisor import american_implied_prob

    away_2, home_2 = simulate_market_moneylines(60.0)
    assert american_implied_prob(away_2) + american_implied_prob(home_2) > 1.0

    away_ml, draw_ml, home_ml = simulate_market_threeway(45.0, 28.0, 27.0)
    overround = (
        american_implied_prob(away_ml)
        + american_implied_prob(draw_ml)
        + american_implied_prob(home_ml)
    )
    assert overround > 1.02


def test_closing_market_fields_includes_draw_close_ml(monkeypatch) -> None:
    """Soccer closes must surface draw ML, not only home/away."""
    monkeypatch.setattr(
        "web.pick_strategy.closing_odds_lookup",
        lambda *_a, **_k: {
            "home_close_ml": -120,
            "draw_close_ml": 240,
            "away_close_ml": 320,
        },
    )
    fields = _closing_market_fields("epl", "2024-01-15", "ars", "che")
    assert fields["market_home_odds"] == -120
    assert fields["market_draw_odds"] == 240
    assert fields["market_away_odds"] == 320


def test_settle_spread_juice_is_side_aware() -> None:
    """Missing pick juice must not borrow the opposite side's price."""
    from web.pick_strategy import _settle_spread_juice

    assert (
        _settle_spread_juice(
            "away",
            None,
            home_spread_odds=-115,
            away_spread_odds=-105,
            market_spread=-3.5,
        )
        == -105
    )
    assert (
        _settle_spread_juice(
            "home",
            None,
            home_spread_odds=-115,
            away_spread_odds=-105,
            market_spread=-3.5,
        )
        == -115
    )
    # Away pick with only home juice present → fail closed.
    assert (
        _settle_spread_juice(
            "away",
            None,
            home_spread_odds=-115,
            away_spread_odds=None,
            market_spread=-3.5,
        )
        is None
    )
    # Synthetic market (no closing line) may use historical -110.
    assert (
        _settle_spread_juice(
            "away",
            None,
            home_spread_odds=-115,
            away_spread_odds=None,
            market_spread=None,
        )
        == -110
    )
    assert (
        _settle_spread_juice(
            "away",
            -108,
            home_spread_odds=-115,
            away_spread_odds=-105,
            market_spread=-3.5,
        )
        == -108
    )
