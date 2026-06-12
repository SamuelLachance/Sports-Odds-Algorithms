"""Regression tests: win probability, favorite side, and American odds stay aligned."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import (  # noqa: E402
    model_moneylines,
    projections_from_win_probs,
    soccer_model_moneylines,
)
from web.blend_service import home_win_prob_to_total_score  # noqa: E402


def _favorite_side_from_projections(away_proj: int, home_proj: int) -> str:
    if away_proj < 0 and home_proj > 0:
        return "away"
    if home_proj < 0 and away_proj > 0:
        return "home"
    raise AssertionError(f"Invalid favorite/underdog signs: away={away_proj}, home={home_proj}")


def test_away_favorite_rays_at_angels_regression() -> None:
    """Rays @ Angels: away favored at 55.68% → away gets negative ML."""
    away_prob = 55.68
    home_prob = 100.0 - away_prob
    total, win_prob = home_win_prob_to_total_score(home_prob)
    assert total > 0
    assert win_prob == away_prob

    away_proj, home_proj = projections_from_win_probs(home_prob, away_prob)
    assert away_proj < 0
    assert home_proj > 0
    assert _favorite_side_from_projections(away_proj, home_proj) == "away"
    assert abs(away_proj) == abs(home_proj)

    away_ml, home_ml = model_moneylines(total)
    assert away_ml == away_proj
    assert home_ml == home_proj


def test_home_favorite_moneylines() -> None:
    total, win_prob = home_win_prob_to_total_score(62.0)
    assert total < 0
    assert win_prob == 62.0

    away_proj, home_proj = model_moneylines(total)
    assert home_proj < 0
    assert away_proj > 0
    assert _favorite_side_from_projections(away_proj, home_proj) == "home"


def test_nba_spread_away_favorite_consistent_with_total_score() -> None:
    """Spread leagues still encode away favorite with positive total_score."""
    total = 58.5
    away_proj, home_proj = model_moneylines(total)
    assert away_proj < 0
    assert home_proj > 0
    assert _favorite_side_from_projections(away_proj, home_proj) == "away"


def test_soccer_threeway_projections_match_probs() -> None:
    home_prob, draw_prob, away_prob = 38.0, 27.0, 35.0
    away_ml, draw_ml, home_ml = soccer_model_moneylines(home_prob, draw_prob, away_prob)

    assert away_ml > 0 and draw_ml > 0 and home_ml > 0
    assert home_ml < away_ml < draw_ml

    home_fav, draw_mid, away_dog = 55.0, 25.0, 20.0
    away_fav_ml, draw_fav_ml, home_fav_ml = soccer_model_moneylines(
        home_fav, draw_mid, away_dog
    )
    assert home_fav_ml < 0
    assert away_fav_ml > 0
    assert draw_fav_ml > 0


if __name__ == "__main__":
    test_away_favorite_rays_at_angels_regression()
    test_home_favorite_moneylines()
    test_nba_spread_away_favorite_consistent_with_total_score()
    test_soccer_threeway_projections_match_probs()
    print("test_model_projections.py: all tests passed")
