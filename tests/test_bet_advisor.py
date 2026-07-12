"""Bet advisor unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import (  # noqa: E402
    BetPick,
    _breakeven_american,
    _odds_edge,
    best_pick_only,
    evaluate_picks,
    evaluate_spread_picks,
    expected_value_pct,
    model_home_margin,
    model_moneylines,
    passes_moneyline_pick_gate,
    spread_line_for_side,
    spread_point_edge,
)
from web.league_profiles import MIN_EXPECTED_VALUE_PCT, MIN_RECOMMENDED_EDGE  # noqa: E402


def test_spread_line_for_side() -> None:
    assert spread_line_for_side(-5.5, "home") == -5.5
    assert spread_line_for_side(-5.5, "away") == 5.5


def test_spread_point_edge_home_favorite() -> None:
    # Model home -8 (home favored), book home -5.5 → 2.5 pt cushion
    margin = -8.0
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
        total_score=-99.0,
        win_probability=99.0,
        consensus_spread=-1.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
    )
    assert picks
    assert picks[0].bet_type == "spread"
    assert picks[0].side == "home"
    assert picks[0].edge >= MIN_RECOMMENDED_EDGE
    assert picks[0].consensus_spread == -1.5
    assert picks[0].spread_line == -1.5
    assert picks[0].extra.get("base_win_probability") == round(picks[0].win_probability, 2)


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
    """Spread convention: negative = home favored, positive = away favored."""
    assert model_home_margin(-60.0, "nba") < 0
    assert model_home_margin(60.0, "nba") > 0


def test_spread_point_edge_away_favorite_small_margin() -> None:
    # Model away favored (+1.2 home margin), book home +9.5 → take home +9.5
    margin = model_home_margin(60.04, "wnba")
    assert margin > 0
    assert spread_point_edge(margin, 9.5, "home") > 8.0
    assert spread_point_edge(margin, 9.5, "away") < 0


def test_moneyline_edge_same_sign_underdog() -> None:
    """Same-sign underdog lines compare directly (Bosnia screenshot)."""
    assert _odds_edge(253, 380, 28.33) == 127.0


def test_moneyline_edge_same_sign_favorite() -> None:
    """Same-sign favorite lines compare directly."""
    away_proj, home_proj = model_moneylines(55.68)
    assert away_proj < 0
    assert home_proj > 0
    assert _odds_edge(away_proj, -171, 55.68) == 0.0
    assert _odds_edge(away_proj, -110, 55.68) == float(-110 - away_proj)
    assert _odds_edge(home_proj, 120, 44.32) == 0.0


def test_moneyline_edge_cross_sign_model_dog_market_fav() -> None:
    """Model underdog priced as favorite: compare to breakeven, not raw subtraction."""
    edge = _odds_edge(122, -70, 45.0)
    fair_underdog = _breakeven_american(45.0, as_underdog=True)
    breakeven_fav = _breakeven_american(45.0, as_underdog=False)
    assert edge > 0.0
    assert edge == float(-70 - breakeven_fav)
    assert fair_underdog > 0


def test_pick_em_win_probs() -> None:
    away_proj, home_proj = model_moneylines(0.0)
    assert away_proj == 100
    assert home_proj == 100


def test_moneyline_edge_cross_sign_not_raw_subtraction() -> None:
    """Padres screenshot: +109 market vs -121 model is ~+26, not +230."""
    padres_prob = 54.79
    edge = _odds_edge(-121, 109, padres_prob)
    fair_underdog = _breakeven_american(padres_prob, as_underdog=True)
    assert abs(edge - (109 - fair_underdog)) < 0.5
    assert edge >= MIN_RECOMMENDED_EDGE
    assert edge != 230.0


def test_official_pick_threshold_is_hubacek_gap() -> None:
    """Official picks follow Hubáček gates with real gap/EV/confidence floors."""
    from web.hubacek_picks import (
        HUBACEK_MIN_EV_PCT,
        HUBACEK_MIN_MARKET_GAP_PP,
        HUBACEK_MIN_WIN_CONFIDENCE_PP,
        HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP,
    )
    from web.league_profiles import OFFICIAL_MIN_EV_PCT
    from web.pick_strategy import OFFICIAL_MIN_EDGE, get_pick_thresholds

    assert OFFICIAL_MIN_EDGE == 0.0
    assert OFFICIAL_MIN_EV_PCT == 0.0
    thresholds = get_pick_thresholds("nba")
    assert thresholds["min_edge"] == 0.0
    assert thresholds["min_ev_pct"] == HUBACEK_MIN_EV_PCT
    assert thresholds["pick_system"] == "hubacek"
    assert thresholds["min_market_gap_pp"] == HUBACEK_MIN_MARKET_GAP_PP
    assert thresholds["min_win_confidence_pp"] == HUBACEK_MIN_WIN_CONFIDENCE_PP
    assert thresholds["min_spread_confidence_pp"] == HUBACEK_SPREAD_MIN_WIN_CONFIDENCE_PP
    assert thresholds["enabled"] is True

    away_proj, _ = model_moneylines(28.33)
    assert _odds_edge(away_proj, 292, 28.33) == 39.0

    picks = evaluate_picks(
        away_name="Bosnia",
        home_name="Opponent",
        away_slug="bosnia",
        home_slug="opponent",
        total_score=28.33,
        win_probability=28.33,
        away_market=292,
        home_market=None,
        min_edge=25.0,
        min_ev_pct=0.0,
    )
    assert len(picks) == 1
    assert picks[0].edge == 39.0


def test_evaluate_picks_cross_sign_positive_ev_qualifies() -> None:
    """Model favorite priced as underdog: clears gate when edge >= 25."""
    picks = evaluate_picks(
        away_name="San Diego Padres",
        home_name="Cincinnati Reds",
        away_slug="san-diego-padres",
        home_slug="cincinnati-reds",
        total_score=54.79,
        win_probability=54.79,
        away_market=109,
        home_market=-130,
        min_edge=25.0,
        min_ev_pct=0.0,
    )
    assert len(picks) == 1
    assert picks[0].edge >= 25.0
    assert picks[0].strategy == "model_favorite"


def test_expected_value_pct_favorite() -> None:
    ev = expected_value_pct(54.79, 109)
    assert ev > MIN_EXPECTED_VALUE_PCT


def test_passes_moneyline_pick_gate_model_favorite() -> None:
    assert passes_moneyline_pick_gate(edge=20, ev_pct=4, strategy="model_favorite")
    assert not passes_moneyline_pick_gate(edge=20, ev_pct=2, strategy="model_favorite")


def test_evaluate_picks_same_sign_meets_threshold() -> None:
    """Large same-sign underdog overlay still clears MIN_RECOMMENDED_EDGE."""
    picks = evaluate_picks(
        away_name="Bosnia",
        home_name="Opponent",
        away_slug="bosnia",
        home_slug="opponent",
        total_score=28.33,
        win_probability=28.33,
        away_market=380,
        home_market=-150,
    )
    assert len(picks) == 1
    assert picks[0].side == "away"
    assert picks[0].edge >= MIN_RECOMMENDED_EDGE
    assert picks[0].edge == 127.0


def test_evaluate_picks_away_favorite_same_sign_soft_line() -> None:
    """Away favorite with softer same-sign market line yields positive edge."""
    away_proj, _ = model_moneylines(55.68)
    edge = _odds_edge(away_proj, -110, 55.68)
    assert edge == float(-110 - away_proj)
    assert edge < MIN_RECOMMENDED_EDGE


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
    # Storm screenshot: ~8.9 pt cushion × 20 ≈ 178 edge
    assert picks[0].edge >= MIN_RECOMMENDED_EDGE


def test_spread_edge_juice_adjustment() -> None:
    from web.bet_advisor import spread_odds_edge

    base = spread_odds_edge(2.0, -110)
    worse_juice = spread_odds_edge(2.0, -120)
    better_juice = spread_odds_edge(2.0, -105)
    assert base > 0
    assert worse_juice < base < better_juice


def test_spread_cover_probability_uses_empirical_sigma() -> None:
    """Cover probability follows Φ(edge/σ), not the old 5 pp/point line."""
    import math

    from web.bet_advisor import (
        SPREAD_MARGIN_SIGMA,
        spread_cover_probability,
        spread_margin_sigma,
    )

    assert spread_margin_sigma("nba") == SPREAD_MARGIN_SIGMA["nba"]
    sigma = spread_margin_sigma("nba")
    expected = 0.5 * (1.0 + math.erf(3.0 / (sigma * math.sqrt(2.0)))) * 100.0
    assert abs(spread_cover_probability(3.0, "nba") - expected) < 1e-9
    # A 2-point cushion is ~56% cover in the NBA, far below the old 60%.
    assert spread_cover_probability(2.0, "nba") < 57.0
    assert spread_cover_probability(0.0, "nba") == 50.0
    # Monotonic in point edge.
    assert (
        spread_cover_probability(1.0, "nba")
        < spread_cover_probability(4.0, "nba")
        < spread_cover_probability(10.0, "nba")
    )


def test_spread_negative_cushion_with_plus_juice_has_no_edge() -> None:
    """WNBA bug: -1.8 pt cushion at +100 must not produce +210 edge."""
    from web.bet_advisor import spread_odds_edge

    assert spread_odds_edge(-1.8, 100) == 0.0
    assert spread_odds_edge(-1.8, -110) == 0.0


def test_wnba_home_favored_line_without_cushion_not_recommended() -> None:
    """Model home -0.3 vs book home -1.5 should not recommend the home spread."""
    margin = model_home_margin(-47.5, "wnba")
    assert abs(margin - 0.3) < 0.05
    picks = evaluate_spread_picks(
        league="wnba",
        away_name="Atlanta Dream",
        home_name="Golden State Valkyries",
        away_slug="atlanta-dream",
        home_slug="golden-state-valkyries",
        total_score=-47.5,
        win_probability=47.5,
        consensus_spread=-1.5,
        away_spread_odds=-110,
        home_spread_odds=100,
    )
    assert all(p.side != "home" for p in picks)


def test_spread_pick_uses_cover_probability() -> None:
    picks = evaluate_spread_picks(
        league="nba",
        away_name="A",
        home_name="B",
        away_slug="a",
        home_slug="b",
        total_score=-99.0,
        win_probability=99.0,
        consensus_spread=-1.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
    )
    assert picks
    assert picks[0].win_probability > 50.0
    assert picks[0].win_probability != 99.0


def test_cross_sign_reason_does_not_claim_book_beats_model_line() -> None:
    picks = evaluate_picks(
        away_name="San Diego Padres",
        home_name="Cincinnati Reds",
        away_slug="san-diego-padres",
        home_slug="cincinnati-reds",
        total_score=54.79,
        win_probability=54.79,
        away_market=250,
        home_market=-300,
    )
    if picks:
        assert "beats the model line" not in picks[0].reason.lower()
        assert "underdog" in picks[0].reason.lower()


def test_wnba_spread_margin_aligns_with_unified_away_favorite() -> None:
    """Atlanta @ Seattle regression: away fav 61% → positive home margin, not home fav."""
    total = 61.43
    margin = model_home_margin(total, "wnba")
    assert margin > 0
    picks = evaluate_spread_picks(
        league="wnba",
        away_name="Atlanta Dream",
        home_name="Seattle Storm",
        away_slug="atlanta-dream",
        home_slug="seattle-storm",
        total_score=total,
        win_probability=61.43,
        consensus_spread=8.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
        model_margin_home=margin,
    )
    assert picks
    assert picks[0].side == "home"
    assert picks[0].model_margin is not None
    assert picks[0].model_margin > 0
    team_margin = (
        picks[0].model_margin
        if picks[0].side == "home"
        else -picks[0].model_margin
    )
    assert team_margin > 0


def test_nfl_layer_predicted_margin_uses_spread_convention() -> None:
    from web.blend_service import _layer_home_margin

    assert _layer_home_margin({"predicted_margin": 6.5}, "nfl") == -6.5


def test_official_spread_pick_requires_hubacek_decorrelation() -> None:
    from web.pick_strategy import evaluate_official_picks_for_game

    base = {
        "total_score": -72.0,
        "win_probability": 72.0,
        "favorite_side": "home",
        "blended_home_win_probability": 72.0,
        "home_spread_margin": -9.5,
    }
    without_decor = evaluate_official_picks_for_game(
        league="wnba",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-72.0,
        win_probability=72.0,
        blended=base,
        away_market=145,
        home_market=-165,
        consensus_spread=-3.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
    )
    assert without_decor == []

    with_decor = evaluate_official_picks_for_game(
        league="wnba",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-72.0,
        win_probability=72.0,
        blended={**base, "market_decorrelated": True},
        away_market=145,
        home_market=-165,
        consensus_spread=-3.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
    )
    assert with_decor
    for pick in with_decor:
        assert pick.strategy == "hubacek"
        assert (pick.extra.get("model_market_gap_pp") or 0) > 0


def test_ensemble_home_margin_sign_does_not_flip_spread_pick() -> None:
    """Storm @ Mercury: home favored by ~7.7 must not recommend away +3.5."""
    from web.pick_strategy import evaluate_official_picks_for_game

    blended = {
        "total_score": -63.01,
        "win_probability": 63.01,
        "favorite_side": "home",
        "blended_home_win_probability": 63.01,
        "market_decorrelated": True,
        # EnsembleML stores score-diff margin; blended_home_spread_margin negates it.
        "home_spread_margin": -7.7,
        "ensemble_ml": {
            "home_win_probability": 63.01,
            "predicted_home_margin": 7.66,
        },
    }
    picks = evaluate_official_picks_for_game(
        league="wnba",
        away_name="Seattle Storm",
        home_name="Phoenix Mercury",
        away_slug="seattle-storm",
        home_slug="phoenix-mercury",
        total_score=-63.01,
        win_probability=63.01,
        blended=blended,
        away_market=145,
        home_market=-175,
        consensus_spread=-3.5,
        away_spread_odds=-105,
        home_spread_odds=-115,
    )
    assert not picks or picks[0].side == "home"
    if picks:
        assert picks[0].side == "home"
        assert picks[0].spread_line == -3.5


def test_spread_picks_require_hubacek_decorrelation_gap() -> None:
    """Small decorrelation vs line fails Hubáček spread gate even with positive edge."""
    blended = {
        "blended_home_win_probability": 55.0,
        "market_decorrelated": True,
    }
    picks = evaluate_spread_picks(
        league="wnba",
        away_name="A",
        home_name="B",
        away_slug="a",
        home_slug="b",
        total_score=-55.0,
        win_probability=55.0,
        consensus_spread=-1.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
        model_margin_home=-1.6,
        hubacek_only=True,
        blended=blended,
    )
    assert picks == []


def test_best_pick_only_selects_highest_profit_score() -> None:
    weak = BetPick(
        side="away",
        team_name="Away",
        team_slug="away",
        strategy="value",
        confidence="medium",
        edge=3.0,
        model_projection=110,
        market_odds=120,
        win_probability=48.0,
        reason="weak",
        ev_pct=1.5,
        profit_score=1.0,
    )
    strong = BetPick(
        side="home",
        team_name="Home",
        team_slug="home",
        strategy="value",
        confidence="high",
        edge=5.0,
        model_projection=-130,
        market_odds=-110,
        win_probability=58.0,
        reason="strong",
        ev_pct=4.0,
        profit_score=9.0,
    )
    assert best_pick_only([]) == []
    chosen = best_pick_only([weak, strong])
    assert len(chosen) == 1
    assert chosen[0].side == "home"
    assert chosen[0].profit_score >= strong.profit_score


def test_blend_outputs_are_market_decorrelated_reads_football_pred() -> None:
    from web.bet_advisor import blend_outputs_are_market_decorrelated

    assert blend_outputs_are_market_decorrelated(
        {"football_pred": {"market_decorrelated": True}}
    )
    assert not blend_outputs_are_market_decorrelated(
        {"football_pred": {"home_win_probability": 55.0}}
    )


if __name__ == "__main__":
    test_spread_line_for_side()
    test_spread_point_edge_home_favorite()
    test_evaluate_spread_picks_meets_edge_threshold()
    test_evaluate_spread_picks_skips_without_consensus()
    test_model_home_margin_sign()
    test_spread_point_edge_away_favorite_small_margin()
    test_moneyline_edge_same_sign_underdog()
    test_moneyline_edge_same_sign_favorite()
    test_moneyline_edge_cross_sign_not_raw_subtraction()
    test_official_pick_threshold_is_hubacek_gap()
    test_evaluate_picks_cross_sign_positive_ev_qualifies()
    test_expected_value_pct_favorite()
    test_passes_moneyline_pick_gate_model_favorite()
    test_evaluate_picks_same_sign_meets_threshold()
    test_evaluate_picks_away_favorite_same_sign_soft_line()
    test_evaluate_spread_picks_favors_underdog_when_market_overlays()
    test_spread_edge_juice_adjustment()
    test_spread_pick_uses_cover_probability()
    test_official_spread_pick_uses_blended_home_spread_margin()
    test_ensemble_home_margin_sign_does_not_flip_spread_pick()
    test_spread_picks_require_hubacek_decorrelation_gap()
    test_cross_sign_reason_does_not_claim_book_beats_model_line()
    test_best_pick_only_selects_highest_profit_score()
    print("test_bet_advisor.py: all tests passed")
