"""Unit tests for web.context_signals (pure logic, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.context_signals import (  # noqa: E402
    MAX_FLB_SHIFT_PP,
    MAX_NEWS_SHIFT_PP,
    apply_context_to_blend,
    favorite_longshot_adjustment,
    is_sparse_sample_league,
    news_sentiment_shift,
    sparse_sample_ev_cap,
    steam_line_movement_shift,
)


def test_flb_zero_without_odds() -> None:
    assert favorite_longshot_adjustment(None, 140, 60.0) == 0.0
    assert favorite_longshot_adjustment(-180, None, 60.0) == 0.0


def test_flb_nudges_toward_underdog_when_home_favorite() -> None:
    shift = favorite_longshot_adjustment(-200, 170, 62.0)
    assert shift < 0
    assert abs(shift) <= MAX_FLB_SHIFT_PP


def test_flb_nudges_toward_underdog_when_away_favorite() -> None:
    shift = favorite_longshot_adjustment(160, -190, 38.0)
    assert shift > 0
    assert abs(shift) <= MAX_FLB_SHIFT_PP


def test_flb_skips_when_model_disagrees_with_chalk() -> None:
    # Book: home favorite, model: away favorite → no FLB correction.
    assert favorite_longshot_adjustment(-200, 170, 40.0) == 0.0


def test_flb_skips_pick_em_prices() -> None:
    assert favorite_longshot_adjustment(-120, 100, 55.0) == 0.0


def test_flb_heavier_favorite_larger_nudge() -> None:
    mild = abs(favorite_longshot_adjustment(-160, 140, 58.0))
    heavy = abs(favorite_longshot_adjustment(-400, 300, 72.0))
    assert heavy >= mild


def test_news_injury_hurts_named_side() -> None:
    shift = news_sentiment_shift(
        ["Celtics star sidelined with ankle injury"],
        home_names=["Celtics", "Boston"],
        away_names=["Heat", "Miami"],
    )
    assert shift < 0
    assert abs(shift) <= MAX_NEWS_SHIFT_PP


def test_news_hot_streak_helps_named_side() -> None:
    shift = news_sentiment_shift(
        ["Miami Heat on a hot streak ahead of the series"],
        home_names=["Celtics"],
        away_names=["Heat", "Miami"],
    )
    assert shift < 0  # away hot → home prob down
    assert abs(shift) <= MAX_NEWS_SHIFT_PP


def test_news_empty_or_unmatched_is_zero() -> None:
    assert news_sentiment_shift([], ["A"], ["B"]) == 0.0
    assert news_sentiment_shift(["Random league news"], ["Celtics"], ["Heat"]) == 0.0


def test_steam_zero_without_opens() -> None:
    assert steam_line_movement_shift(-150, 130) == 0.0


def test_steam_toward_home_when_implied_rises() -> None:
    # Open home +150 (~40%), close -130 (~56.5%) → steam home.
    shift = steam_line_movement_shift(-130, 110, open_home_ml=150, open_away_ml=-170)
    assert shift > 0


def test_sparse_league_detection() -> None:
    assert is_sparse_sample_league("worldcup")
    assert is_sparse_sample_league("fifa_friendlies")
    assert is_sparse_sample_league("copa_america")
    assert is_sparse_sample_league("concacaf_wcq")
    assert is_sparse_sample_league("CONCACAF_GOLD")
    assert not is_sparse_sample_league("nba")
    assert not is_sparse_sample_league("epl")


def test_sparse_ev_cap_blocks_absurd_world_cup_ev() -> None:
    capped = sparse_sample_ev_cap("worldcup", None, 99.0)
    assert capped < 30.0
    assert capped == sparse_sample_ev_cap("worldcup", 3, 99.0)


def test_sparse_ev_cap_relaxes_with_more_games() -> None:
    thin = sparse_sample_ev_cap("copa_america", 4, 50.0)
    thicker = sparse_sample_ev_cap("copa_america", 20, 50.0)
    assert thin <= thicker
    assert thicker <= 50.0


def test_sparse_ev_cap_passthrough_for_dense_nba() -> None:
    assert sparse_sample_ev_cap("nba", 40, 12.5) == 12.5


def test_sparse_ev_cap_soft_for_thin_regular_league() -> None:
    capped = sparse_sample_ev_cap("nba", 3, 80.0)
    assert capped < 80.0
    assert capped <= 55.0


def test_apply_context_stores_pre_context_and_updates_probs() -> None:
    blended = {
        "blended_home_win_probability": 62.0,
        "total_score": -62.0,
        "win_probability": 62.0,
        "favorite_side": "home",
    }
    market = {"home_moneyline": -200, "away_moneyline": 170}
    out = apply_context_to_blend(
        blended,
        market=market,
        headlines=["Celtics star sidelined with injury"],
        home_names=["Celtics"],
        away_names=["Heat"],
        league="nba",
    )
    assert "pre_context_home_win_probability" in out
    assert out["pre_context_home_win_probability"] == 62.0
    assert "context_adjustment_pp" in out
    assert out["blended_home_win_probability"] != 62.0 or out["context_adjustment_pp"] == 0.0
    # FLB (home fav) + injury on home → home prob should drop.
    assert out["blended_home_win_probability"] < 62.0


def test_apply_context_noop_without_signals() -> None:
    blended = {
        "blended_home_win_probability": 55.0,
        "total_score": -55.0,
        "win_probability": 55.0,
        "favorite_side": "home",
    }
    out = apply_context_to_blend(blended, market={}, league="nba")
    assert out["context_adjustment_pp"] == 0.0
    assert out["blended_home_win_probability"] == 55.0


def test_apply_context_threeway_renormalizes() -> None:
    blended = {
        "threeway": True,
        "home_win_probability": 40.0,
        "draw_probability": 28.0,
        "away_win_probability": 32.0,
        "blended_home_win_probability": 40.0,
        "total_score": 60.0,
        "win_probability": 60.0,
        "favorite_side": "away",
    }
    out = apply_context_to_blend(
        blended,
        market={"home_moneyline": 180, "away_moneyline": -210},
        league="epl",
    )
    total = (
        float(out["home_win_probability"])
        + float(out["draw_probability"])
        + float(out["away_win_probability"])
    )
    assert total == pytest.approx(100.0, abs=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
