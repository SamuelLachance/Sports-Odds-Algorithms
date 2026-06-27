"""Bet advisor unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import (  # noqa: E402
    _breakeven_american,
    _odds_edge,
    evaluate_picks,
    evaluate_spread_picks,
    model_home_margin,
    model_moneylines,
    spread_line_for_side,
    spread_point_edge,
)
from web.league_profiles import MIN_RECOMMENDED_EDGE  # noqa: E402


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
    assert edge < 40.0
    assert edge != 230.0


def test_edge_threshold_rejects_39_accepts_40() -> None:
    """MIN_RECOMMENDED_EDGE is 40 — reject at 39, accept at 40."""
    away_proj, _ = model_moneylines(28.33)
    assert _odds_edge(away_proj, 292, 28.33) == 39.0
    assert _odds_edge(away_proj, 293, 28.33) == 40.0

    below = evaluate_picks(
        away_name="Bosnia",
        home_name="Opponent",
        away_slug="bosnia",
        home_slug="opponent",
        total_score=28.33,
        win_probability=28.33,
        away_market=292,
        home_market=None,
    )
    assert below == []

    above = evaluate_picks(
        away_name="Bosnia",
        home_name="Opponent",
        away_slug="bosnia",
        home_slug="opponent",
        total_score=28.33,
        win_probability=28.33,
        away_market=293,
        home_market=None,
    )
    assert len(above) == 1
    assert above[0].edge == 40.0


def test_evaluate_picks_cross_sign_below_threshold() -> None:
    """Model favorite priced as underdog has real value but not +50 edge."""
    picks = evaluate_picks(
        away_name="San Diego Padres",
        home_name="Cincinnati Reds",
        away_slug="san-diego-padres",
        home_slug="cincinnati-reds",
        total_score=54.79,
        win_probability=54.79,
        away_market=109,
        home_market=-130,
    )
    assert picks == []


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
    assert base == 40.0
    assert worse_juice == 30.0
    assert better_juice == 45.0


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
    test_edge_threshold_rejects_39_accepts_40()
    test_evaluate_picks_cross_sign_below_threshold()
    test_evaluate_picks_same_sign_meets_threshold()
    test_evaluate_picks_away_favorite_same_sign_soft_line()
    test_evaluate_spread_picks_favors_underdog_when_market_overlays()
    test_spread_edge_juice_adjustment()
    test_spread_pick_uses_cover_probability()
    test_cross_sign_reason_does_not_claim_book_beats_model_line()
    print("test_bet_advisor.py: all tests passed")
