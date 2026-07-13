"""Official pick selection must use Hubáček decorrelated probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import (  # noqa: E402
    blend_outputs_are_market_decorrelated,
    ensure_hubacek_in_blend,
    expected_value_pct,
    official_pick_binary_probs,
    passes_hubacek_official_pick_gate,
)
from web.hubacek_picks import HUBACEK_MIN_WIN_CONFIDENCE_PP  # noqa: E402
from web.market_decorrelation import decorrelate_binary  # noqa: E402
from web.pick_strategy import evaluate_official_picks_for_game, get_pick_thresholds  # noqa: E402


def test_blend_outputs_are_market_decorrelated_flags() -> None:
    assert blend_outputs_are_market_decorrelated({"market_decorrelated": True})
    # Ensemble blend_mode alone is not enough — apply_ensemble_ml only sets
    # market_decorrelated on the moneyline path.
    assert not blend_outputs_are_market_decorrelated(
        {"blend_mode": "ensemble_ml", "ensemble_ml": {"home_win_probability": 60.0}}
    )
    assert blend_outputs_are_market_decorrelated(
        {
            "blend_mode": "ensemble_ml",
            "ensemble_ml": {"home_win_probability": 60.0},
            "market_decorrelated": True,
        }
    )
    assert blend_outputs_are_market_decorrelated(
        {"blend_mode": "algo_v1", "market_decorrelated": True}
    )
    assert not blend_outputs_are_market_decorrelated(
        {"blended_home_win_probability": 62.0, "total_score": -62.0}
    )


def test_ensure_hubacek_in_blend_pushes_away_from_market() -> None:
    blended = {
        "total_score": -60.0,
        "win_probability": 60.0,
        "favorite_side": "home",
        "blended_home_win_probability": 60.0,
    }
    updated = ensure_hubacek_in_blend(
        blended,
        league="nhl",
        away_market=130,
        home_market=-150,
    )
    assert updated["market_decorrelated"] is True
    assert updated["blended_home_win_probability"] > 60.0


def test_ensure_hubacek_in_blend_skips_threeway_soccer() -> None:
    """Path A / 1X2 blends must not be rewritten by binary Hubáček."""
    blended = {
        "threeway": True,
        "total_score": -42.0,
        "win_probability": 42.0,
        "favorite_side": "home",
        "blended_home_win_probability": 42.0,
        "home_win_probability": 42.0,
        "draw_probability": 28.0,
        "away_win_probability": 30.0,
    }
    updated = ensure_hubacek_in_blend(
        blended,
        league="epl",
        away_market=160,
        home_market=170,
    )
    assert updated is blended or updated == blended
    assert updated.get("market_decorrelated") is not True
    assert updated["blended_home_win_probability"] == 42.0
    assert updated["draw_probability"] == 28.0
    assert updated["away_win_probability"] == 30.0


def test_official_moneyline_ev_differs_from_raw_model_prob() -> None:
    """Official path must not grade +EV using pre-decorrelation win %."""
    raw_home = 62.0
    market_home = 50.0
    decor_home = decorrelate_binary(raw_home, market_home)
    assert decor_home != raw_home

    _, pick_home = official_pick_binary_probs(
        {
            "blended_home_win_probability": decor_home,
            "total_score": -decor_home,
            "blend_mode": "algo_v1",
            "market_decorrelated": True,
        },
        -decor_home,
        league="nhl",
        away_market=110,
        home_market=-130,
    )
    assert pick_home == decor_home
    assert pick_home != raw_home

    ev_decor = expected_value_pct(decor_home, -130)
    ev_raw = expected_value_pct(raw_home, -130)
    assert ev_decor != ev_raw


def test_official_pick_binary_probs_applies_hubacek_when_missing() -> None:
    away, home = official_pick_binary_probs(
        {"blended_home_win_probability": 58.0, "total_score": -58.0},
        -58.0,
        league="mlb",
        away_market=120,
        home_market=-140,
    )
    assert home > 58.0
    assert away == 100.0 - home


def test_hubacek_gate_rejects_market_agreement() -> None:
    assert not passes_hubacek_official_pick_gate(
        model_prob_pct=55.0,
        market_implied_pct=55.0,
        ev_pct=30.0,
    )
    assert not passes_hubacek_official_pick_gate(
        model_prob_pct=52.0,
        market_implied_pct=55.0,
        ev_pct=30.0,
    )
    assert passes_hubacek_official_pick_gate(
        model_prob_pct=72.0,
        market_implied_pct=55.0,
        ev_pct=5.0,
    )


def test_official_picks_use_hubacek_strategy_when_qualifying() -> None:
    decor_home = decorrelate_binary(62.0, 50.0)
    blended = {
        "total_score": -decor_home,
        "win_probability": decor_home,
        "favorite_side": "home",
        "blended_home_win_probability": decor_home,
        "market_decorrelated": True,
    }
    picks = evaluate_official_picks_for_game(
        league="nhl",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-decor_home,
        win_probability=decor_home,
        blended=blended,
        away_market=110,
        home_market=-130,
        consensus_spread=None,
        away_spread_odds=None,
        home_spread_odds=None,
    )
    if picks:
        assert picks[0].strategy == "hubacek"
        assert (picks[0].extra.get("model_market_gap_pp") or 0) > 0


def test_unavailable_blend_stub_never_emits_official_picks() -> None:
    """Coin-flip unavailable stubs must not qualify via market decorrelation."""
    blended = {
        "blend_mode": "mlb_runcast_unavailable",
        "blend_layers": 0,
        "total_score": 0.0,
        "win_probability": 50.0,
        "favorite_side": "neutral",
    }
    picks = evaluate_official_picks_for_game(
        league="mlb",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=0.0,
        win_probability=50.0,
        blended=blended,
        away_market=-150,
        home_market=130,
        consensus_spread=None,
        away_spread_odds=None,
        home_spread_odds=None,
    )
    assert picks == []


def test_mlb_pick_thresholds_use_backtested_policy() -> None:
    """MLB gates come from data/pick_strategy.json (walk-forward backtest)."""
    thresholds = get_pick_thresholds("mlb")
    assert thresholds["min_market_gap_pp"] >= 6.0
    assert thresholds["min_win_confidence_pp"] == 0.0
    assert thresholds["ml_lo"] == -200
    assert thresholds["ml_hi"] == 200
    thresholds_ncaabb = get_pick_thresholds("ncaabb")
    # College baseball uses baseball confidence default (no league override).
    assert thresholds_ncaabb["min_win_confidence_pp"] == 10.0


def test_mlb_official_picks_use_moneyline_decorrelation() -> None:
    from web.pick_strategy import get_pick_thresholds as _thresholds

    min_gap = _thresholds("mlb")["min_market_gap_pp"]
    pre_home = 68.0
    decor_home = decorrelate_binary(pre_home, 55.0)
    # devig(130/-150) home ≈ 58.0% -> ensure the scenario clears the MLB gap gate
    assert decor_home - 58.0 >= min_gap
    blended = {
        "total_score": -decor_home,
        "win_probability": decor_home,
        "favorite_side": "home",
        "blended_home_win_probability": decor_home,
        "baseball_pred": {
            "market_decorrelated": True,
            "market_decorrelation_source": "moneyline",
            "pre_decorrelation_home_win_probability": pre_home,
            "home_win_probability": decor_home,
        },
    }
    picks = evaluate_official_picks_for_game(
        league="mlb",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-decor_home,
        win_probability=decor_home,
        blended=blended,
        away_market=130,
        home_market=-150,
        consensus_spread=-1.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
    )
    assert picks
    assert picks[0].strategy == "hubacek"
    assert picks[0].bet_type == "moneyline"


def test_nhl_pick_thresholds_use_backtested_policy() -> None:
    """NHL open-line hybrid Hubáček policy from pick_strategy.json."""
    thresholds = get_pick_thresholds("nhl")
    assert thresholds["min_market_gap_pp"] >= 5.0
    assert thresholds["min_win_confidence_pp"] == 0.0
    assert thresholds["ml_lo"] == -250
    assert thresholds["ml_hi"] == 250
    assert thresholds["bet_type"] == "moneyline"
    assert thresholds["enabled"] is True
    assert thresholds.get("fav_mode", "any") == "any"


def test_nhl_official_picks_use_moneyline_decorrelation() -> None:
    """NHL official book is either-side ML in [-250, +250] at the open."""
    thresholds = get_pick_thresholds("nhl")
    min_gap = thresholds["min_market_gap_pp"]
    # Home -180 / away +160. Push model toward away so the dog clears gap+EV.
    pre_home = 40.0
    market_home = 64.3  # approx de-vig of -180 vs +160
    decor_home = decorrelate_binary(pre_home, market_home)
    decor_away = 100.0 - decor_home
    assert decor_away - (100.0 - market_home) >= min_gap
    blended = {
        "total_score": decor_away,  # away lean
        "win_probability": decor_away,
        "favorite_side": "away",
        "blended_home_win_probability": decor_home,
        "hockey_pred": {
            "algorithm": "NHLGradientBoost",
            "model_version": "v2",
            "market_decorrelated": True,
            "market_decorrelation_source": "moneyline",
            "pre_decorrelation_home_win_probability": pre_home,
            "home_win_probability": decor_home,
        },
    }
    picks = evaluate_official_picks_for_game(
        league="nhl",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=decor_away,
        win_probability=decor_away,
        blended=blended,
        away_market=160,
        home_market=-180,
        consensus_spread=None,
        away_spread_odds=None,
        home_spread_odds=None,
    )
    assert picks
    assert picks[0].strategy == "hubacek"
    assert picks[0].bet_type == "moneyline"
    assert picks[0].side == "away"
    assert picks[0].market_odds == 160


def test_nhl_official_picks_respect_ml_price_window() -> None:
    """NHL moneyline picks outside [-250, +250] are rejected."""
    pre_home = 85.0
    decor_home = decorrelate_binary(pre_home, 72.0)
    blended = {
        "total_score": -decor_home,
        "win_probability": decor_home,
        "favorite_side": "home",
        "blended_home_win_probability": decor_home,
        "hockey_pred": {
            "algorithm": "NHLGradientBoost",
            "model_version": "v2",
            "market_decorrelated": True,
            "market_decorrelation_source": "moneyline",
            "pre_decorrelation_home_win_probability": pre_home,
            "home_win_probability": decor_home,
        },
    }
    picks = evaluate_official_picks_for_game(
        league="nhl",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-decor_home,
        win_probability=decor_home,
        blended=blended,
        away_market=280,
        home_market=-400,
        consensus_spread=None,
        away_spread_odds=None,
        home_spread_odds=None,
    )
    assert picks == []


def test_mlb_official_picks_respect_ml_price_window() -> None:
    """MLB moneyline picks outside [-200, +200] are rejected."""
    pre_home = 80.0
    decor_home = decorrelate_binary(pre_home, 68.0)
    blended = {
        "total_score": -decor_home,
        "win_probability": decor_home,
        "favorite_side": "home",
        "blended_home_win_probability": decor_home,
        "baseball_pred": {
            "market_decorrelated": True,
            "market_decorrelation_source": "moneyline",
            "pre_decorrelation_home_win_probability": pre_home,
            "home_win_probability": decor_home,
        },
    }
    picks = evaluate_official_picks_for_game(
        league="mlb",
        away_name="Away",
        home_name="Home",
        away_slug="away",
        home_slug="home",
        total_score=-decor_home,
        win_probability=decor_home,
        blended=blended,
        away_market=210,
        home_market=-250,
        consensus_spread=-1.5,
        away_spread_odds=-110,
        home_spread_odds=-110,
    )
    assert not [p for p in picks if p.side == "home"]


def test_nfl_cfb_official_picks_hubacek_gates() -> None:
    """Hubáček OOS: NFL moneyline niche clears; CFB stays disabled."""
    from web.hubacek_picks import clear_strategy_cache
    from web.league_profiles import eligible_for_official_picks
    from web.pick_strategy import get_pick_thresholds, load_pick_strategy

    clear_strategy_cache()
    load_pick_strategy.cache_clear()

    nfl = get_pick_thresholds("nfl")
    assert nfl["bet_type"] == "moneyline"
    assert nfl["enabled"] is True
    assert nfl.get("fav_mode") == "favorite"
    assert eligible_for_official_picks("nfl") is True

    cfb = get_pick_thresholds("cfb")
    assert cfb["bet_type"] == "moneyline"
    assert cfb["enabled"] is False
    assert eligible_for_official_picks("cfb") is False

    cbb = get_pick_thresholds("cbb")
    assert cbb["bet_type"] == "moneyline"
    assert cbb["enabled"] is True
    assert cbb.get("fav_mode") == "favorite"
    assert eligible_for_official_picks("cbb") is True
