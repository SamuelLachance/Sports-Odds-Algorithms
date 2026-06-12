"""Soccer 3-way moneyline model and bet advisor tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import (  # noqa: E402
    evaluate_soccer_picks,
    soccer_model_moneylines,
    soccer_threeway_probs,
)
from web.espn_client import _extract_draw_moneyline  # noqa: E402
from web.league_profiles import MIN_RECOMMENDED_EDGE  # noqa: E402


def test_soccer_threeway_probs_sum_to_100() -> None:
    for total in (-75.0, -55.0, 0.0, 45.0, 80.0):
        home, draw, away = soccer_threeway_probs(total, "epl")
        total_prob = round(home + draw + away, 4)
        assert abs(total_prob - 100.0) < 0.01
        assert 18.0 <= draw <= 35.0


def test_closer_matchups_raise_draw_probability() -> None:
    _, draw_even, _ = soccer_threeway_probs(50.0, "epl")
    _, draw_blowout, _ = soccer_threeway_probs(-85.0, "epl")
    assert draw_even > draw_blowout


def test_soccer_model_moneylines_from_probs() -> None:
    home, draw, away = soccer_threeway_probs(-62.0, "worldcup")
    away_ml, draw_ml, home_ml = soccer_model_moneylines(home, draw, away)
    assert draw_ml > 0
    assert away_ml > 0
    assert isinstance(home_ml, int)
    assert isinstance(draw_ml, int)


def test_evaluate_soccer_picks_includes_draw() -> None:
    home, draw, away = soccer_threeway_probs(-55.0, "epl")
    away_ml, draw_ml, home_ml = soccer_model_moneylines(home, draw, away)
    picks = evaluate_soccer_picks(
        away_name="Arsenal",
        home_name="Chelsea",
        away_slug="arsenal",
        home_slug="chelsea",
        total_score=-55.0,
        home_prob=home,
        draw_prob=draw,
        away_prob=away,
        away_proj=away_ml,
        draw_proj=draw_ml,
        home_proj=home_ml,
        away_market=away_ml + 80,
        draw_market=draw_ml + 80,
        home_market=home_ml + 80,
    )
    assert picks
    assert len(picks) == 1
    assert picks[0].edge >= MIN_RECOMMENDED_EDGE
    assert picks[0].bet_type == "moneyline"


def test_evaluate_soccer_picks_single_best_when_multiple_qualify() -> None:
    home, draw, away = soccer_threeway_probs(-52.0, "epl")
    away_ml, draw_ml, home_ml = soccer_model_moneylines(home, draw, away)
    picks = evaluate_soccer_picks(
        away_name="Arsenal",
        home_name="Chelsea",
        away_slug="arsenal",
        home_slug="chelsea",
        total_score=-52.0,
        home_prob=home,
        draw_prob=draw,
        away_prob=away,
        away_proj=away_ml,
        draw_proj=draw_ml,
        home_proj=home_ml,
        away_market=away_ml + 120,
        draw_market=draw_ml + 90,
        home_market=home_ml + 60,
    )
    assert len(picks) == 1
    assert picks[0].side == "away"


def test_extract_draw_moneyline_from_espn_shape() -> None:
    odds_block = {
        "moneyline": {
            "draw": {"close": {"odds": "+250"}},
        }
    }
    assert _extract_draw_moneyline(odds_block) == 250

    odds_block_alt = {"drawOdds": {"moneyLine": "+220"}}
    assert _extract_draw_moneyline(odds_block_alt) == 220


if __name__ == "__main__":
    test_soccer_threeway_probs_sum_to_100()
    test_closer_matchups_raise_draw_probability()
    test_soccer_model_moneylines_from_probs()
    test_evaluate_soccer_picks_includes_draw()
    test_evaluate_soccer_picks_single_best_when_multiple_qualify()
    test_extract_draw_moneyline_from_espn_shape()
    print("test_soccer_threeway.py: all tests passed")
