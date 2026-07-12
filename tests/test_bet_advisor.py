"""Bet advisor unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
    market_home_prob_pct,
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
    """Model underdog priced as favorite: compare to breakeven, not raw subtraction.

    For a 45% side, favorite breakeven is ~-82. Any valid American favorite
    (≤ -100) is worse than that, so edge is 0 — garbage like -70 must not invent
    a positive edge either (see test_odds_edge_rejects_invalid_sub_100_american).
    """
    edge = _odds_edge(122, -110, 45.0)
    fair_underdog = _breakeven_american(45.0, as_underdog=True)
    breakeven_fav = _breakeven_american(45.0, as_underdog=False)
    assert fair_underdog > 0
    assert breakeven_fav > -100  # sub-50% "favorite" fair is softer than -100
    assert edge == 0.0
    # Higher-confidence dog can still clear a soft favorite price.
    edge_soft = _odds_edge(100, -105, 48.0)
    be_soft = _breakeven_american(48.0, as_underdog=False)
    # 48% fav breakeven ≈ -92; -105 is still worse → 0
    assert be_soft > -100
    assert edge_soft == 0.0
    # Model favorite priced softer as dog: classic +value cross-sign.
    padres_prob = 54.79
    edge_dog = _odds_edge(-121, 109, padres_prob)
    assert edge_dog > 0.0


def test_odds_edge_rejects_invalid_sub_100_american() -> None:
    """Garbage |odds| < 100 must not invent a positive edge."""
    assert _odds_edge(150, -50, 40.0) == 0.0
    assert _odds_edge(-150, 50, 60.0) == 0.0
    assert _odds_edge(122, -70, 45.0) == 0.0


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


def test_expected_value_pct_rejects_non_finite_and_bool_prob() -> None:
    """±inf used to clamp into 99.9%/0.1%; bool True→1%; NaN leaked."""
    import math

    import pytest

    from web.bet_advisor import kelly_fraction

    for bad in (float("nan"), float("inf"), float("-inf"), True, False):
        with pytest.raises(ValueError):
            expected_value_pct(bad, -110)
        assert kelly_fraction(bad, -110) == 0.0
    assert math.isfinite(expected_value_pct(55.0, -110))


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
    # Negative cushions are below 50% (symmetric around the CDF).
    assert spread_cover_probability(-3.0, "nba") < 50.0
    assert abs(
        spread_cover_probability(-3.0, "nba") + spread_cover_probability(3.0, "nba") - 100.0
    ) < 1e-9
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
        # EnsembleML already stores book-convention margin (negative = home favored).
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


def test_evaluate_picks_applies_thin_sample_ev_cap_via_games_proxy() -> None:
    """Early-season non-sparse leagues soft-cap absurd EV when proxy is tiny."""
    picks = evaluate_picks(
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=55.0,
        win_probability=55.0,
        away_market=400,
        home_market=-500,
        away_prob=55.0,
        home_prob=45.0,
        base_away_prob=55.0,
        base_home_prob=45.0,
        hubacek_only=True,
        min_market_gap_pp=2.0,
        min_win_confidence_pp=0.0,
        min_ev_pct=2.0,
        league="nba",
        games_played_proxy=3,
    )
    assert picks
    # Uncapped EV at +400 with 55% is +175%; thin-sample soft-cap is 55%.
    assert all(pick.ev_pct <= 55.0 for pick in picks)
    assert picks[0].extra.get("games_played_proxy") == 3
    uncapped = evaluate_picks(
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=55.0,
        win_probability=55.0,
        away_market=400,
        home_market=-500,
        away_prob=55.0,
        home_prob=45.0,
        base_away_prob=55.0,
        base_home_prob=45.0,
        hubacek_only=True,
        min_market_gap_pp=2.0,
        min_win_confidence_pp=0.0,
        min_ev_pct=2.0,
        league="nba",
        games_played_proxy=40,
    )
    assert uncapped
    assert uncapped[0].ev_pct > 55.0

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


def test_hubacek_spread_skips_missing_juice() -> None:
    """Official Hubáček must not invent −110 when spread juice is absent."""
    blended = {
        "blended_home_win_probability": 62.0,
        "home_spread_margin": -6.5,
        "market_decorrelated": True,
    }
    picks = evaluate_spread_picks(
        league="nba",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-62.0,
        win_probability=62.0,
        consensus_spread=-3.5,
        away_spread_odds=None,
        home_spread_odds=None,
        model_margin_home=-6.5,
        hubacek_only=True,
        blended=blended,
        min_ev_pct=0.0,
        min_cover_gap_pp=0.0,
        min_win_confidence_pp=0.0,
    )
    assert picks == []


def test_spread_picks_skip_missing_juice_even_without_hubacek() -> None:
    """Non-Hubáček ranking must not invent −110 when juice is absent."""
    picks = evaluate_spread_picks(
        league="nba",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-90.0,
        win_probability=90.0,
        consensus_spread=-1.5,
        away_spread_odds=None,
        home_spread_odds=None,
        model_margin_home=-12.0,
        min_edge=0.0,
    )
    assert picks == []


def test_enrich_spread_profit_metrics_skips_missing_juice() -> None:
    from web.bet_advisor import enrich_pick_profit_metrics

    pick = BetPick(
        side="home",
        team_name="Home",
        team_slug="home",
        strategy="value",
        confidence="medium",
        edge=5.0,
        model_projection=-150,
        market_odds=-110,
        win_probability=55.0,
        reason="test",
        bet_type="spread",
        consensus_spread=-3.5,
        spread_line=-3.5,
        spread_odds=None,
    )
    enriched = enrich_pick_profit_metrics(pick)
    assert enriched is pick
    assert enriched.ev_pct == 0.0
    assert enriched.profit_score == 0.0
    assert "kelly_pct" not in enriched.extra


def test_evaluate_spread_picks_applies_thin_sample_ev_cap() -> None:
    """Early-season NBA spreads must soft-cap absurd EV like moneylines do."""
    # Home has the point edge; give it plus-money juice so uncapped EV >> 55%.
    picks = evaluate_spread_picks(
        league="nba",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-70.0,
        win_probability=70.0,
        consensus_spread=-3.5,
        away_spread_odds=-500,
        home_spread_odds=400,
        model_margin_home=-10.0,
        min_edge=0.0,
        min_ev_pct=0.0,
        games_played_proxy=3,
    )
    assert picks
    assert all(pick.ev_pct <= 55.0 for pick in picks)
    assert picks[0].extra.get("games_played_proxy") == 3

    uncapped = evaluate_spread_picks(
        league="nba",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-70.0,
        win_probability=70.0,
        consensus_spread=-3.5,
        away_spread_odds=-500,
        home_spread_odds=400,
        model_margin_home=-10.0,
        min_edge=0.0,
        min_ev_pct=0.0,
        games_played_proxy=40,
    )
    assert uncapped
    assert uncapped[0].ev_pct > 55.0

    from web.bet_advisor import enrich_pick_profit_metrics

    enriched = enrich_pick_profit_metrics(picks[0])
    # Kelly / profit sizing must also respect the thin-sample haircut.
    assert enriched.ev_pct <= 55.0


def test_resolve_binary_win_probs_renormalizes_threeway() -> None:
    from web.bet_advisor import resolve_binary_win_probs

    away, home = resolve_binary_win_probs(
        {
            "threeway": True,
            "home_win_probability": 40.0,
            "draw_probability": 25.0,
            "away_win_probability": 35.0,
        },
        0.0,
    )
    assert away + home == pytest.approx(100.0)
    assert home == pytest.approx(40.0 / 75.0 * 100.0)
    assert away == pytest.approx(35.0 / 75.0 * 100.0)


def test_american_odds_zero_treated_as_even() -> None:
    from web.bet_advisor import american_implied_prob, american_to_decimal

    assert american_to_decimal(0) == pytest.approx(2.0)
    assert american_implied_prob(0) == pytest.approx(0.5)


def test_odds_edge_treats_zero_as_even() -> None:
    """ESPN EVEN (0) must match +100 in edge math, not invent a huge same-sign gap."""
    assert _odds_edge(-150, 0, 60.0) == _odds_edge(-150, 100, 60.0)


def test_normalize_american_odds_maps_even_and_rejects_garbage() -> None:
    from web.bet_advisor import normalize_american_odds

    assert normalize_american_odds(0) == 100
    assert normalize_american_odds(100) == 100
    assert normalize_american_odds(-110) == -110
    assert normalize_american_odds(50) is None
    assert normalize_american_odds(-50) is None
    assert normalize_american_odds(None) is None
    # JSON floats / numeric strings must round-trip, not drop as invalid.
    assert normalize_american_odds(-110.0) == -110
    assert normalize_american_odds("-110.0") == -110
    # Text EVEN/PK must match espn_client — raw book strings reach EV/Kelly paths.
    assert normalize_american_odds("EVEN") == 100
    assert normalize_american_odds("even") == 100
    assert normalize_american_odds("PK") == 100
    assert normalize_american_odds("OFF") is None
    assert normalize_american_odds(True) is None
    assert normalize_american_odds(False) is None
    # Non-finite floats must fail closed as None — never OverflowError.
    assert normalize_american_odds(float("inf")) is None
    assert normalize_american_odds(float("-inf")) is None
    assert normalize_american_odds(float("nan")) is None
    assert normalize_american_odds("inf") is None


def test_american_helpers_reject_sub_100_odds() -> None:
    from web.bet_advisor import american_implied_prob, american_to_decimal, expected_value_pct

    with pytest.raises(ValueError):
        american_implied_prob(50)
    with pytest.raises(ValueError):
        american_to_decimal(-50)
    with pytest.raises(ValueError):
        expected_value_pct(55.0, 50)


def test_evaluate_picks_even_market_uses_model_favorite_strategy() -> None:
    """ESPN EVEN (0) must gate like +100 underdog for cross-sign model favorites."""
    common = dict(
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-65.0,
        win_probability=65.0,
        away_market=-120,
        min_edge=0.0,
        min_ev_pct=0.0,
    )
    even_picks = evaluate_picks(**common, home_market=0)
    plus_picks = evaluate_picks(**common, home_market=100)
    assert even_picks and plus_picks
    assert even_picks[0].side == "home"
    assert plus_picks[0].side == "home"
    assert even_picks[0].strategy == plus_picks[0].strategy == "model_favorite"
    assert even_picks[0].market_odds == 100


def test_evaluate_picks_skips_garbage_american_odds() -> None:
    picks = evaluate_picks(
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-65.0,
        win_probability=65.0,
        away_market=50,
        home_market=-50,
        min_edge=0.0,
        min_ev_pct=0.0,
    )
    assert picks == []


def test_enrich_pick_profit_metrics_haircuts_kelly_when_sparse_ev_capped() -> None:
    """Sparse EV cap must also shrink Kelly / expected_units / profit_score."""
    from web.bet_advisor import BetPick, enrich_pick_profit_metrics, kelly_fraction

    pick = BetPick(
        side="home",
        team_name="A",
        team_slug="a",
        strategy="hubacek",
        confidence="high",
        edge=20.0,
        model_projection=-110,
        market_odds=250,
        win_probability=70.0,
        reason="thin sample",
        extra={"league": "worldcup", "games_played_proxy": 2},
    )
    enrich_pick_profit_metrics(pick)
    assert pick.ev_pct <= 18.0
    uncapped_kelly_pct = kelly_fraction(70.0, 250) * 100.0
    assert pick.extra["kelly_pct"] < uncapped_kelly_pct
    assert pick.extra["expected_units"] == pytest.approx(pick.ev_pct / 100.0)


def test_enrich_pick_profit_metrics_accepts_float_string_odds() -> None:
    """JSON float-string juice must not crash Kelly enrichment."""
    from web.bet_advisor import BetPick, enrich_pick_profit_metrics

    pick = BetPick(
        side="home",
        team_name="A",
        team_slug="a",
        strategy="hubacek",
        confidence="high",
        edge=5.0,
        model_projection=-110,
        market_odds="-110.0",  # type: ignore[arg-type]
        win_probability=58.0,
        reason="json odds",
        extra={"league": "mlb"},
    )
    enrich_pick_profit_metrics(pick)
    assert pick.extra.get("kelly_pct") is not None
    assert pick.profit_score > 0


def test_enrich_pick_profit_metrics_maps_even_zero_for_kelly() -> None:
    from web.bet_advisor import BetPick, enrich_pick_profit_metrics, kelly_fraction

    pick = BetPick(
        side="home",
        team_name="A",
        team_slug="a",
        strategy="hubacek",
        confidence="high",
        edge=5.0,
        model_projection=-110,
        market_odds=0,
        win_probability=58.0,
        reason="even",
        extra={"league": "mlb"},
    )
    enrich_pick_profit_metrics(pick)
    assert pick.extra["kelly_pct"] == pytest.approx(kelly_fraction(58.0, 100) * 100.0)


def test_enrich_pick_profit_metrics_skips_non_finite_or_bool_prob() -> None:
    """Corrupt probs must fail closed (not clamp inf→99.9% or True→1%)."""
    from web.bet_advisor import BetPick, enrich_pick_profit_metrics

    for bad in (float("nan"), float("inf"), True):
        pick = BetPick(
            side="home",
            team_name="A",
            team_slug="a",
            strategy="hubacek",
            confidence="high",
            edge=5.0,
            model_projection=-110,
            market_odds=-110,
            win_probability=bad,  # type: ignore[arg-type]
            reason="bad",
        )
        enrich_pick_profit_metrics(pick)
        assert pick.extra.get("kelly_pct") is None
        assert pick.ev_pct == 0.0


def test_market_home_prob_pct_rejects_bool_spread() -> None:
    """False consensus_spread must not become pick'em → 50% home win."""
    assert market_home_prob_pct(consensus_spread=False) is None  # type: ignore[arg-type]
    assert market_home_prob_pct(consensus_spread=True) is None  # type: ignore[arg-type]
    # Real pick'em still maps near 50%.
    assert market_home_prob_pct(consensus_spread=0.0) == pytest.approx(50.0, abs=0.5)


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
