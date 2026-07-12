"""Soccer 3-way moneyline model and bet advisor tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import (  # noqa: E402
    MIN_RECOMMENDED_EDGE,
    evaluate_soccer_picks,
    soccer_model_moneylines,
    soccer_team_pick_blocked_by_projected_score,
    soccer_threeway_probs,
)
from web.espn_client import _extract_draw_moneyline  # noqa: E402
from web.soccer_blend import threeway_probs_to_total_score  # noqa: E402


def test_threeway_probs_to_total_score_draw_favorite() -> None:
    total, win_prob, favorite = threeway_probs_to_total_score(28.0, 44.0, 28.0)
    assert favorite == "draw"
    assert win_prob == 44.0
    assert total == 0.0


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
        home_prob=home,
        draw_prob=draw,
        away_prob=away,
        away_proj=away_ml,
        draw_proj=draw_ml,
        home_proj=home_ml,
        away_market=away_ml + 120,
        draw_market=draw_ml + 120,
        home_market=home_ml + 120,
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


def test_soccer_team_pick_blocked_when_projected_score_favors_other_side() -> None:
    assert soccer_team_pick_blocked_by_projected_score(
        "home", expected_home_goals=0.8, expected_away_goals=2.5
    )
    assert soccer_team_pick_blocked_by_projected_score(
        "away", expected_home_goals=2.4, expected_away_goals=0.7
    )
    assert not soccer_team_pick_blocked_by_projected_score(
        "home", expected_home_goals=1.1, expected_away_goals=2.4
    )
    assert not soccer_team_pick_blocked_by_projected_score(
        "draw", expected_home_goals=0.5, expected_away_goals=3.0
    )


def test_evaluate_soccer_picks_skips_team_when_projected_score_conflicts() -> None:
    home, draw, away = soccer_threeway_probs(-52.0, "epl")
    away_ml, draw_ml, home_ml = soccer_model_moneylines(home, draw, away)
    picks = evaluate_soccer_picks(
        away_name="Arsenal",
        home_name="Chelsea",
        away_slug="arsenal",
        home_slug="chelsea",
        home_prob=home,
        draw_prob=draw,
        away_prob=away,
        away_proj=away_ml,
        draw_proj=draw_ml,
        home_proj=home_ml,
        away_market=away_ml + 120,
        draw_market=draw_ml + 90,
        home_market=home_ml + 60,
        expected_home_goals=0.6,
        expected_away_goals=2.2,
    )
    assert picks
    assert all(pick.side != "home" for pick in picks)


def test_hubacek_soccer_fails_closed_without_full_1x2() -> None:
    """Incomplete 1X2 must not fall back to 0–1 implied prob vs 0–100 model probs."""
    picks = evaluate_soccer_picks(
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        home_prob=55.0,
        draw_prob=25.0,
        away_prob=20.0,
        away_proj=200,
        draw_proj=250,
        home_proj=-140,
        away_market=180,
        draw_market=None,
        home_market=-150,
        hubacek_only=True,
        min_market_gap_pp=2.0,
        min_win_confidence_pp=0.0,
        min_ev_pct=2.0,
        league="epl",
    )
    assert picks == []


def test_hubacek_soccer_fails_closed_when_devig_returns_none(monkeypatch) -> None:
    """Invalid full 1X2 boards must fail closed, not TypeError on unpack."""
    monkeypatch.setattr(
        "web.soccer_decorrelation.devig_threeway_from_odds",
        lambda *_a, **_k: None,
    )
    picks = evaluate_soccer_picks(
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        home_prob=55.0,
        draw_prob=25.0,
        away_prob=20.0,
        away_proj=200,
        draw_proj=260,
        home_proj=-140,
        away_market=200,
        draw_market=260,
        home_market=-140,
        hubacek_only=True,
        min_market_gap_pp=2.0,
        min_win_confidence_pp=0.0,
        min_ev_pct=2.0,
        league="epl",
    )
    assert picks == []


def test_hubacek_soccer_blocks_projected_score_conflict() -> None:
    picks = evaluate_soccer_picks(
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        home_prob=52.0,
        draw_prob=26.0,
        away_prob=22.0,
        away_proj=250,
        draw_proj=300,
        home_proj=-110,
        away_market=260,
        draw_market=270,
        home_market=120,
        expected_home_goals=0.5,
        expected_away_goals=2.4,
        base_home_prob=51.0,
        hubacek_only=True,
        min_market_gap_pp=4.0,
        min_win_confidence_pp=0.0,
        min_ev_pct=5.0,
        league="epl",
    )
    assert all(pick.side != "home" for pick in picks)


def test_soccer_non_hubacek_stores_market_implied_on_percent_scale() -> None:
    picks = evaluate_soccer_picks(
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        home_prob=40.0,
        draw_prob=30.0,
        away_prob=30.0,
        away_proj=150,
        draw_proj=220,
        home_proj=130,
        away_market=150,
        draw_market=None,
        home_market=130,
        min_edge=0.0,
        min_ev_pct=0.0,
    )
    assert picks
    for pick in picks:
        assert pick.market_implied_prob is not None
        assert pick.market_implied_prob > 1.0
        assert pick.market_implied_prob < 99.0


def test_extract_draw_moneyline_from_espn_shape() -> None:
    odds_block = {
        "moneyline": {
            "draw": {"close": {"odds": "+250"}},
        }
    }
    assert _extract_draw_moneyline(odds_block) == 250

    odds_block_alt = {"drawOdds": {"moneyLine": "+220"}}
    assert _extract_draw_moneyline(odds_block_alt) == 220


def test_blend_threeway_layers_empty_weights_is_safe() -> None:
    from web.soccer_blend import blend_threeway_layers

    assert blend_threeway_layers([], []) == (33.33, 33.33, 33.34)
    assert blend_threeway_layers([(50.0, 25.0, 25.0)], [0.0]) == (33.33, 33.33, 33.34)
    home, draw, away = blend_threeway_layers([(60.0, 20.0, 20.0)], [1.0])
    assert abs(home + draw + away - 100.0) < 0.01
    assert home == 60.0


def test_path_a_context_syncs_pick_probs_without_market_decorr(monkeypatch) -> None:
    """Path A ESPN context must refresh pick_* when not market-decorrelated."""
    from web.soccer_blend import _attach_soccer_context_layer

    monkeypatch.setattr(
        "web.soccer_blend.build_soccer_context_payload",
        lambda *args, **kwargs: {
            "home_win_probability": 48.0,
            "draw_probability": 26.0,
            "away_win_probability": 26.0,
        },
    )
    result = {
        "home_win_probability": 42.0,
        "draw_probability": 28.0,
        "away_win_probability": 30.0,
        "soccer_pred": {
            "home_win_probability": 42.0,
            "draw_probability": 28.0,
            "away_win_probability": 30.0,
            "pick_home_win_probability": 42.0,
            "pick_draw_probability": 28.0,
            "pick_away_win_probability": 30.0,
            "market_decorrelated": False,
        },
    }
    out = _attach_soccer_context_layer(
        result,
        league="epl",
        cutoff_date="2026-07-12",
        home_abbr="ars",
        away_abbr="che",
    )
    pred = out["soccer_pred"]
    assert pred["pick_home_win_probability"] == 48.0
    assert pred["pick_draw_probability"] == 26.0
    assert pred["pick_away_win_probability"] == 26.0


def test_path_a_context_preserves_decorrelated_pick_probs(monkeypatch) -> None:
    from web.soccer_blend import _attach_soccer_context_layer

    monkeypatch.setattr(
        "web.soccer_blend.build_soccer_context_payload",
        lambda *args, **kwargs: {
            "home_win_probability": 48.0,
            "draw_probability": 26.0,
            "away_win_probability": 26.0,
        },
    )
    result = {
        "home_win_probability": 42.0,
        "draw_probability": 28.0,
        "away_win_probability": 30.0,
        "soccer_pred": {
            "home_win_probability": 42.0,
            "draw_probability": 28.0,
            "away_win_probability": 30.0,
            "pick_home_win_probability": 44.0,
            "pick_draw_probability": 27.0,
            "pick_away_win_probability": 29.0,
            "market_decorrelated": True,
        },
    }
    out = _attach_soccer_context_layer(
        result,
        league="epl",
        cutoff_date="2026-07-12",
        home_abbr="ars",
        away_abbr="che",
    )
    pred = out["soccer_pred"]
    assert pred["pick_home_win_probability"] == 44.0
    assert pred["home_win_probability"] == 48.0


if __name__ == "__main__":
    test_soccer_threeway_probs_sum_to_100()
    test_closer_matchups_raise_draw_probability()
    test_soccer_model_moneylines_from_probs()
    test_evaluate_soccer_picks_includes_draw()
    test_evaluate_soccer_picks_single_best_when_multiple_qualify()
    test_soccer_team_pick_blocked_when_projected_score_favors_other_side()
    test_evaluate_soccer_picks_skips_team_when_projected_score_conflicts()
    test_hubacek_soccer_fails_closed_without_full_1x2()
    test_hubacek_soccer_blocks_projected_score_conflict()
    test_soccer_non_hubacek_stores_market_implied_on_percent_scale()
    test_extract_draw_moneyline_from_espn_shape()
    print("test_soccer_threeway.py: all tests passed")
