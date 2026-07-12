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
    _resolve_home_prob,
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


def test_news_ignores_looking_out_for_false_positive() -> None:
    """Former 'out for' stem matched 'looking out for' draft copy."""
    assert (
        news_sentiment_shift(
            ["Celtics looking out for young talent in the draft"],
            home_names=["Celtics"],
            away_names=["Heat"],
        )
        == 0.0
    )


def test_news_ignores_rolling_and_dominant_false_positives() -> None:
    assert (
        news_sentiment_shift(
            ["Heat rolling out new uniforms ahead of tip"],
            home_names=["Celtics"],
            away_names=["Heat"],
        )
        == 0.0
    )
    assert (
        news_sentiment_shift(
            ["Historically dominant Celtics franchise since 2008"],
            home_names=["Celtics"],
            away_names=["Heat"],
        )
        == 0.0
    )


def test_news_ignores_injury_report_header_without_availability() -> None:
    assert (
        news_sentiment_shift(
            ["Celtics injury report: no changes ahead of tip-off"],
            home_names=["Celtics"],
            away_names=["Heat"],
        )
        == 0.0
    )


def test_news_injury_report_still_counts_sidelined() -> None:
    shift = news_sentiment_shift(
        ["Celtics injury report: star sidelined for tonight"],
        home_names=["Celtics"],
        away_names=["Heat"],
    )
    assert shift < 0


def test_news_team_token_requires_word_boundary() -> None:
    """Short nicknames must not match substrings (Heat⊂heating, Sun⊂Sunday)."""
    assert (
        news_sentiment_shift(
            ["Rivalry heating up ahead of Sunday night"],
            home_names=["Heat"],
            away_names=["Sun"],
        )
        == 0.0
    )
    assert (
        news_sentiment_shift(
            ["Heat star sidelined with ankle sprain"],
            home_names=["Heat"],
            away_names=["Sun"],
        )
        < 0
    )


def test_steam_zero_without_opens() -> None:
    assert steam_line_movement_shift(-150, 130) == 0.0


def test_steam_toward_home_when_implied_rises() -> None:
    # Open home +150 (~40%), close -130 (~56.5%) → steam home.
    shift = steam_line_movement_shift(-130, 110, open_home_ml=150, open_away_ml=-170)
    assert shift > 0


def test_steam_rejects_invalid_american_magnitude() -> None:
    """Garbage |odds| < 100 must not invent a steam nudge."""
    assert (
        steam_line_movement_shift(-110, 100, open_home_ml=50, open_away_ml=-150) == 0.0
    )
    assert (
        steam_line_movement_shift(-50, 130, open_home_ml=-110, open_away_ml=100) == 0.0
    )


def test_steam_rejects_bool_false_as_even_open() -> None:
    """Missing open ML as False must not become EVEN (+100) and invent steam."""
    poisoned = steam_line_movement_shift(
        -150, 130, open_home_ml=False, open_away_ml=130  # type: ignore[arg-type]
    )
    assert poisoned == 0.0
    assert steam_line_movement_shift(-150, 130, open_home_ml=None, open_away_ml=130) == 0.0


def test_flb_rejects_bool_false_as_even_dog() -> None:
    """Missing side as False must not become EVEN and invent an FLB nudge."""
    assert favorite_longshot_adjustment(-200, False, 65.0) == 0.0  # type: ignore[arg-type]
    assert favorite_longshot_adjustment(False, -200, 40.0) == 0.0  # type: ignore[arg-type]
    assert favorite_longshot_adjustment(-200, None, 65.0) == 0.0


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


def test_games_played_proxy_from_blend_reads_sport_pred_and_context() -> None:
    from web.context_signals import games_played_proxy_from_blend

    assert games_played_proxy_from_blend(
        {"basketball_pred": {"home_games": 4, "away_games": 7}}
    ) == 4
    assert games_played_proxy_from_blend(
        {
            "context_signals": {"games_played_proxy": 3},
            "basketball_pred": {"home_games": 40, "away_games": 40},
        }
    ) == 3
    assert games_played_proxy_from_blend({}) is None


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


def test_apply_context_steam_activates_from_market_opens() -> None:
    """Multi-book opens in the market dict switch on the steam signal."""
    blended = {
        "blended_home_win_probability": 55.0,
        "total_score": -55.0,
        "win_probability": 55.0,
        "favorite_side": "home",
    }
    # -140 close (~58.3% implied) vs +115 open (~46.5%) → strong home steam.
    # -140 is above the FLB threshold so steam is the only active signal.
    market = {
        "home_moneyline": -140,
        "away_moneyline": 120,
        "open_home_moneyline": 115,
        "open_away_moneyline": -135,
    }
    out = apply_context_to_blend(blended, market=market, league="nba")
    signals = out["context_signals"]
    assert signals["steam_pp"] > 0
    assert signals["flb_pp"] == 0.0
    assert signals["news_pp"] == 0.0
    assert out["context_adjustment_pp"] == signals["steam_pp"]
    assert out["blended_home_win_probability"] > 55.0


def test_apply_context_flb_uses_opening_moneylines_not_close() -> None:
    """FLB follows open-like chalk; a shortened close must not erase the bias."""
    blended = {
        "blended_home_win_probability": 62.0,
        "total_score": -62.0,
        "win_probability": 62.0,
        "favorite_side": "home",
    }
    # Open heavy home favorite (−200) → FLB; close shortened to −140 (no FLB).
    market = {
        "home_moneyline": -140,
        "away_moneyline": 120,
        "open_home_moneyline": -200,
        "open_away_moneyline": 170,
    }
    out = apply_context_to_blend(blended, market=market, league="nba")
    assert out["context_signals"]["flb_pp"] < 0
    # Live-only FLB would be 0 at −140; opens must drive the nudge.
    assert favorite_longshot_adjustment(-140, 120, 62.0) == 0.0


def test_apply_context_steam_inactive_without_opens() -> None:
    blended = {
        "blended_home_win_probability": 55.0,
        "total_score": -55.0,
        "win_probability": 55.0,
        "favorite_side": "home",
    }
    market = {"home_moneyline": -140, "away_moneyline": 120}
    out = apply_context_to_blend(blended, market=market, league="nba")
    assert out["context_signals"]["steam_pp"] == 0.0
    assert out["blended_home_win_probability"] == 55.0


def test_opening_steam_does_not_flip_computed_steam_sign() -> None:
    """Sport-model opening_steam must floor magnitude, not invert ML steam."""
    from unittest.mock import patch

    blended = {
        "blended_home_win_probability": 55.0,
        "total_score": -55.0,
        "win_probability": 55.0,
        "favorite_side": "home",
        "opening_steam": {"steam_signal": True, "steam_direction": "home"},
    }
    market = {"home_moneyline": -110, "away_moneyline": -110}
    with patch("web.context_signals.steam_line_movement_shift", return_value=-0.8):
        with patch("web.context_signals.favorite_longshot_adjustment", return_value=0.0):
            with patch("web.context_signals.news_sentiment_shift", return_value=0.0):
                out = apply_context_to_blend(blended, market=market, league="nba")
    # Conflict: computed steam is away (−0.8) while opening says home — keep −0.8.
    assert out["context_signals"]["steam_pp"] == pytest.approx(-0.8)
    assert out["blended_home_win_probability"] < 55.0


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
    # Draw is held fixed; only home↔away move.
    assert float(out["draw_probability"]) == pytest.approx(28.0, abs=0.01)


def test_apply_context_threeway_holds_draw_when_clamped() -> None:
    """Extreme home clamp must not rescale draw mass."""
    from unittest.mock import patch

    blended = {
        "threeway": True,
        "home_win_probability": 2.0,
        "draw_probability": 28.0,
        "away_win_probability": 70.0,
        "blended_home_win_probability": 2.0,
        "total_score": 70.0,
        "win_probability": 70.0,
        "favorite_side": "away",
    }
    with (
        patch("web.context_signals.favorite_longshot_adjustment", return_value=-5.0),
        patch("web.context_signals.steam_line_movement_shift", return_value=0.0),
        patch("web.context_signals.news_sentiment_shift", return_value=0.0),
    ):
        out = apply_context_to_blend(blended, market={}, league="epl")
    assert float(out["draw_probability"]) == pytest.approx(28.0, abs=0.01)
    assert (
        float(out["home_win_probability"])
        + float(out["draw_probability"])
        + float(out["away_win_probability"])
    ) == pytest.approx(100.0, abs=0.05)
    assert float(out["home_win_probability"]) < 2.0


def test_apply_context_preserves_draw_favorite() -> None:
    """Draw-favorite threeways must not be forced to home after a context shift."""
    blended = {
        "threeway": True,
        "home_win_probability": 28.0,
        "draw_probability": 44.0,
        "away_win_probability": 28.0,
        "blended_home_win_probability": 28.0,
        "total_score": 0.0,
        "win_probability": 44.0,
        "favorite_side": "draw",
    }
    # Small FLB/steam shift that keeps draw as the mode after renormalize.
    out = apply_context_to_blend(
        blended,
        market={
            "home_moneyline": 150,
            "away_moneyline": 150,
            "open_home_moneyline": 140,
            "open_away_moneyline": 160,
        },
        league="epl",
    )
    assert out["context_adjustment_pp"] != 0.0
    assert out["favorite_side"] == "draw"
    assert float(out["draw_probability"]) > float(out["home_win_probability"])
    assert float(out["draw_probability"]) > float(out["away_win_probability"])


def test_resolve_home_prob_rejects_draw_favorite_without_threeway() -> None:
    """Legacy draw payloads must not invert draw% into a fake home win%."""
    assert (
        _resolve_home_prob({"win_probability": 44.0, "favorite_side": "draw"})
        is None
    )
    assert _resolve_home_prob(
        {"win_probability": 44.0, "favorite_side": "away"}
    ) == pytest.approx(56.0)
    assert _resolve_home_prob(
        {"win_probability": 44.0, "favorite_side": "home"}
    ) == pytest.approx(44.0)


def test_apply_context_shifts_pre_decorrelation_and_sport_pred() -> None:
    """Context must move Honest-EV base and nested sport probs with the board."""
    blended = {
        "blended_home_win_probability": 60.0,
        "total_score": -60.0,
        "win_probability": 60.0,
        "favorite_side": "home",
        "pre_decorrelation_home_win_probability": 63.0,
        "market_decorrelated": True,
        "baseball_pred": {
            "home_win_probability": 60.0,
            "pre_decorrelation_home_win_probability": 63.0,
            "market_decorrelated": True,
        },
    }
    # Strong home steam from open → close.
    market = {
        "home_moneyline": -140,
        "away_moneyline": 120,
        "open_home_moneyline": 115,
        "open_away_moneyline": -135,
    }
    out = apply_context_to_blend(blended, market=market, league="mlb")
    assert out["context_adjustment_pp"] != 0.0
    shift = float(out["context_adjustment_pp"])
    assert out["pre_decorrelation_home_win_probability"] == pytest.approx(
        63.0 + shift, abs=0.05
    )
    assert out["baseball_pred"]["home_win_probability"] == pytest.approx(
        60.0 + shift, abs=0.05
    )


def test_apply_context_syncs_nested_away_win_probability() -> None:
    """Binary sport preds must keep away = 100 - home after a context shift."""
    blended = {
        "blended_home_win_probability": 60.0,
        "total_score": -60.0,
        "win_probability": 60.0,
        "favorite_side": "home",
        "away_win_probability": 40.0,
        "hockey_pred": {
            "home_win_probability": 60.0,
            "away_win_probability": 40.0,
        },
    }
    market = {
        "home_moneyline": -140,
        "away_moneyline": 120,
        "open_home_moneyline": 115,
        "open_away_moneyline": -135,
    }
    out = apply_context_to_blend(blended, market=market, league="nhl")
    assert out["context_adjustment_pp"] != 0.0
    home = float(out["hockey_pred"]["home_win_probability"])
    away = float(out["hockey_pred"]["away_win_probability"])
    assert away == pytest.approx(100.0 - home, abs=0.01)
    assert float(out["away_win_probability"]) == pytest.approx(
        100.0 - float(out["home_win_probability"]), abs=0.01
    )


def test_apply_context_syncs_nested_soccer_pred_threeway() -> None:
    """Three-way context shift must keep soccer_pred in step with the board."""
    blended = {
        "threeway": True,
        "home_win_probability": 40.0,
        "draw_probability": 28.0,
        "away_win_probability": 32.0,
        "blended_home_win_probability": 40.0,
        "total_score": -55.56,
        "win_probability": 55.56,
        "favorite_side": "home",
        "soccer_pred": {
            "home_win_probability": 40.0,
            "draw_probability": 28.0,
            "away_win_probability": 32.0,
        },
    }
    market = {
        "home_moneyline": 180,
        "away_moneyline": -210,
        "open_home_moneyline": 130,
        "open_away_moneyline": -150,
    }
    out = apply_context_to_blend(blended, market=market, league="epl")
    assert out["context_adjustment_pp"] != 0.0
    assert out["soccer_pred"]["home_win_probability"] == out["home_win_probability"]
    assert out["soccer_pred"]["draw_probability"] == out["draw_probability"]
    assert out["soccer_pred"]["away_win_probability"] == out["away_win_probability"]


def test_apply_context_syncs_blended_threeway() -> None:
    """Context shift must rewrite blended_threeway, not leave pre-shift probs."""
    blended = {
        "threeway": True,
        "home_win_probability": 40.0,
        "draw_probability": 28.0,
        "away_win_probability": 32.0,
        "blended_home_win_probability": 40.0,
        "total_score": -55.56,
        "win_probability": 55.56,
        "favorite_side": "home",
        "blended_threeway": {
            "home_win_probability": 40.0,
            "draw_probability": 28.0,
            "away_win_probability": 32.0,
        },
    }
    market = {
        "home_moneyline": 180,
        "away_moneyline": -210,
        "open_home_moneyline": 130,
        "open_away_moneyline": -150,
    }
    out = apply_context_to_blend(blended, market=market, league="epl")
    assert out["context_adjustment_pp"] != 0.0
    assert out["blended_threeway"]["home_win_probability"] == out["home_win_probability"]
    assert out["blended_threeway"]["draw_probability"] == out["draw_probability"]
    assert out["blended_threeway"]["away_win_probability"] == out["away_win_probability"]
    assert out["blended_threeway"]["home_win_probability"] != 40.0


def test_steam_open_even_zero_not_skipped_by_falsy_or() -> None:
    """American open ML 0 (EVEN) must not fall through via falsy `or`."""
    blended = {
        "blended_home_win_probability": 55.0,
        "total_score": -55.0,
        "win_probability": 55.0,
        "favorite_side": "home",
    }
    # Hard steam home vs EVEN open; compare against explicit +100 open.
    market_even = {
        "home_moneyline": -160,
        "away_moneyline": 140,
        "open_home_moneyline": 0,
        "open_away_moneyline": -135,
    }
    market_plus = {
        "home_moneyline": -160,
        "away_moneyline": 140,
        "open_home_moneyline": 100,
        "open_away_moneyline": -135,
    }
    out_even = apply_context_to_blend(blended, market=market_even, league="nba")
    out_plus = apply_context_to_blend(blended, market=market_plus, league="nba")
    assert out_even["context_signals"]["steam_pp"] == out_plus["context_signals"]["steam_pp"]
    assert out_even["context_signals"]["steam_pp"] != 0.0


def test_apply_context_threeway_pre_decorrelation_uses_board_clamp() -> None:
    """Honest-EV base must use non-draw mass clamps, not binary 5–95."""
    from unittest.mock import patch

    blended = {
        "threeway": True,
        "home_win_probability": 68.0,
        "draw_probability": 30.0,
        "away_win_probability": 2.0,
        "pre_decorrelation_home_win_probability": 68.0,
    }
    with (
        patch("web.context_signals.favorite_longshot_adjustment", return_value=3.0),
        patch("web.context_signals.steam_line_movement_shift", return_value=0.0),
        patch("web.context_signals.news_sentiment_shift", return_value=0.0),
    ):
        out = apply_context_to_blend(
            blended, market={}, headlines=[], home_names=[], away_names=[], league="epl"
        )
    # remaining non-draw mass = 70 → home hi = 69; both board and pre must match.
    assert out["home_win_probability"] == 69.0
    assert out["pre_decorrelation_home_win_probability"] == 69.0
    assert out["away_win_probability"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
