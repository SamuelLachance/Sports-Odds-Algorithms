"""Bet advisor unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import (  # noqa: E402
    evaluate_spread_picks,
    model_home_margin,
    spread_line_for_side,
    spread_point_edge,
)
from web.league_profiles import MIN_RECOMMENDED_EDGE  # noqa: E402


def test_spread_line_for_side() -> None:
    assert spread_line_for_side(-5.5, "home") == -5.5
    assert spread_line_for_side(-5.5, "away") == 5.5


def test_spread_point_edge_home_favorite() -> None:
    # Model home by 8, book home -5.5 → 2.5 pt cushion
    margin = 8.0
    assert spread_point_edge(margin, -5.5, "home") == 2.5
    assert spread_point_edge(margin, -5.5, "away") < 0


def test_evaluate_spread_picks_meets_edge_threshold() -> None:
    # total_score < 0 → home favorite; high win prob → large margin
    picks = evaluate_spread_picks(
        league="nba",
        away_name="Knicks",
        home_name="Spurs",
        away_slug="new-york-knicks",
        home_slug="san-antonio-spurs",
        total_score=-95.0,
        win_probability=95.0,
        consensus_spread=-3.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
    )
    assert picks
    assert picks[0].bet_type == "spread"
    assert picks[0].side == "home"
    assert picks[0].edge >= MIN_RECOMMENDED_EDGE
    assert picks[0].consensus_spread == -3.5
    assert picks[0].spread_line == -3.5


def test_evaluate_spread_picks_skips_without_consensus() -> None:
    picks = evaluate_spread_picks(
        league="nba",
        away_name="A",
        home_name="B",
        away_slug="a",
        home_slug="b",
        total_score=-60.0,
        win_probability=60.0,
        consensus_spread=None,
    )
    assert picks == []


def test_model_home_margin_sign() -> None:
    assert model_home_margin(-60.0, "nba") > 0
    assert model_home_margin(60.0, "nba") < 0


def test_spread_point_edge_away_favorite_small_margin() -> None:
    # Model away by 1.2, book home +9.5 → fade away -9.5, take home +9.5
    margin = model_home_margin(60.04, "wnba")
    assert margin < 0
    assert spread_point_edge(margin, 9.5, "home") > 8.0
    assert spread_point_edge(margin, 9.5, "away") < 0


def test_evaluate_spread_picks_favors_underdog_when_market_overlays() -> None:
    picks = evaluate_spread_picks(
        league="wnba",
        away_name="Golden State Valkyries",
        home_name="Seattle Storm",
        away_slug="golden-state-valkyries",
        home_slug="seattle-storm",
        total_score=60.04,
        win_probability=60.04,
        consensus_spread=9.5,
        away_spread_odds=-105,
        home_spread_odds=-115,
    )
    assert picks
    assert picks[0].side == "home"
    assert picks[0].team_name == "Seattle Storm"
    assert picks[0].spread_line == 9.5


if __name__ == "__main__":
    test_spread_line_for_side()
    test_spread_point_edge_home_favorite()
    test_evaluate_spread_picks_meets_edge_threshold()
    test_evaluate_spread_picks_skips_without_consensus()
    test_model_home_margin_sign()
    test_spread_point_edge_away_favorite_small_margin()
    test_evaluate_spread_picks_favors_underdog_when_market_overlays()
    print("test_bet_advisor.py: all tests passed")
