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
)


def test_official_bet_type_by_sport() -> None:
    assert official_bet_type("nba") == "spread"
    assert official_bet_type("nfl") == "spread"
    assert official_bet_type("nhl") == "moneyline"
    assert official_bet_type("mlb") == "moneyline"
    assert official_bet_type("epl") == "soccer_1x2"


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
    assert "min_edge" in DEFAULT_THRESHOLDS


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
