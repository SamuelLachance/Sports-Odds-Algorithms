"""Official pick strategy (backtest-tuned spread vs moneyline routing)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.pick_strategy import (  # noqa: E402
    DEFAULT_THRESHOLDS,
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
    assert official_bet_type("epl") == "none"


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
