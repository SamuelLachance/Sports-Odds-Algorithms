"""Unit tests for shared NBA/WNBA v2 market-aware live helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.basketball_v2_market import (  # noqa: E402
    apply_market_features,
    devig_home_prob,
    resolve_market_heads,
)


def test_devig_home_prob_rejects_incomplete_and_invalid_prices() -> None:
    assert devig_home_prob(None, -110) is None
    assert devig_home_prob(-110, None) is None
    assert devig_home_prob(-50, 130) is None  # non-american juice band
    assert devig_home_prob("bad", -110) is None
    even = devig_home_prob(-110, -110)
    assert even is not None and abs(even - 0.5) < 0.02
    fav = devig_home_prob(-200, 170)
    assert fav is not None and fav > 0.6
    # string American odds are accepted
    assert abs(float(devig_home_prob("-110", "100") or 0) - float(devig_home_prob(-110, 100) or 0)) < 1e-9


def test_devig_home_prob_treats_espn_even_zero_as_plus_100() -> None:
    """ESPN EVEN (0) must enable market-aware heads, not return None."""
    from_zero = devig_home_prob(0, -110)
    from_plus = devig_home_prob(100, -110)
    assert from_zero is not None
    assert from_plus is not None
    assert abs(float(from_zero) - float(from_plus)) < 1e-9
    feats: dict[str, float] = {}
    has_market, _ = apply_market_features(feats, home_moneyline=0, away_moneyline=-110)
    assert has_market is True
    assert feats["has_market"] == 1.0


def test_devig_home_prob_rejects_bool_false_as_even() -> None:
    """False floats to 0.0; must not invent EVEN market-aware features."""
    assert devig_home_prob(False, -110) is None
    assert devig_home_prob(-110, False) is None
    assert devig_home_prob(False, False) is None
    assert devig_home_prob(True, -110) is None
    feats: dict[str, float] = {}
    has_market, has_spread = apply_market_features(
        feats, home_moneyline=False, away_moneyline=-110, home_spread=False
    )
    assert has_market is False
    assert has_spread is False
    assert feats["has_market"] == 0.0
    assert feats["has_spread"] == 0.0


def test_data_devig_two_way_delegates_and_accepts_strings() -> None:
    from web.nba_v2.data import devig_two_way as nba_devig
    from web.wnba_v2.data import devig_two_way as wnba_devig

    for fn in (nba_devig, wnba_devig):
        probs = fn("-150", "130")
        assert probs is not None
        assert abs(sum(probs) - 1.0) < 1e-9
        assert probs[0] > probs[1]
        assert fn("bad", 130) is None


def test_apply_market_features_defaults_when_odds_missing() -> None:
    features: dict[str, float] = {}
    has_market, has_spread = apply_market_features(features)
    assert has_market is False
    assert has_spread is False
    assert features["mkt_home_prob"] == 0.5
    assert features["has_market"] == 0.0
    assert features["mkt_home_spread"] == 0.0
    assert features["has_spread"] == 0.0
    assert features["spread_move"] == 0.0
    assert features["ml_steam_pp"] == 0.0
    assert features["has_steam"] == 0.0


def test_compute_steam_features_fail_closed_and_paired() -> None:
    from web.basketball_v2_market import compute_steam_features

    empty = compute_steam_features(home_spread_close=-3.5)
    assert empty["has_steam"] == 0.0
    assert empty["spread_move"] == 0.0

    half = compute_steam_features(home_spread_open=-3.0)  # close missing
    assert half["has_steam"] == 0.0

    spread = compute_steam_features(home_spread_open=-3.0, home_spread_close=-4.5)
    assert spread["has_steam"] == 1.0
    assert spread["spread_move"] == pytest.approx(-1.5)
    assert spread["ml_steam_pp"] == 0.0

    ml = compute_steam_features(
        home_ml_open=-110,
        away_ml_open=-110,
        home_ml_close=-150,
        away_ml_close=130,
    )
    assert ml["has_steam"] == 1.0
    assert ml["ml_steam_pp"] > 0.0  # close favours home more than open

    feats: dict[str, float] = {}
    apply_market_features(
        feats,
        home_moneyline=-150,
        away_moneyline=130,
        home_spread=-4.5,
        home_spread_open=-3.0,
        home_ml_open=-110,
        away_ml_open=-110,
    )
    assert feats["has_steam"] == 1.0
    assert feats["spread_move"] == pytest.approx(-1.5)
    assert feats["ml_steam_pp"] > 0.0


def test_apply_market_features_injects_live_odds() -> None:
    features: dict[str, float] = {}
    has_market, has_spread = apply_market_features(
        features,
        home_moneyline=-150,
        away_moneyline=130,
        home_spread=-4.5,
    )
    assert has_market is True
    assert has_spread is True
    assert features["has_market"] == 1.0
    assert features["has_spread"] == 1.0
    assert features["mkt_home_spread"] == -4.5
    assert 0.55 < features["mkt_home_prob"] < 0.70


def test_apply_market_features_failsoft_invalid_spreads() -> None:
    """Junk / PK / bool spreads must not raise or invent market_aware margin."""
    for junk in ("N/A", "OFF", "", False, "abc"):
        feats: dict[str, float] = {}
        has_market, has_spread = apply_market_features(feats, home_spread=junk)
        assert has_market is False
        assert has_spread is False
        assert feats["has_spread"] == 0.0
        assert feats["mkt_home_spread"] == 0.0

    pk: dict[str, float] = {}
    has_market, has_spread = apply_market_features(pk, home_spread="PK")
    assert has_market is False
    assert has_spread is True
    assert pk["mkt_home_spread"] == 0.0
    assert pk["has_spread"] == 1.0

    even: dict[str, float] = {}
    _, has_spread = apply_market_features(even, home_spread="EVEN")
    assert has_spread is True
    assert even["mkt_home_spread"] == 0.0

    # string numeric spreads still work (ESPN / enrich paths)
    numeric: dict[str, float] = {}
    _, has_spread = apply_market_features(numeric, home_spread="-3.5")
    assert has_spread is True
    assert numeric["mkt_home_spread"] == -3.5


def test_apply_market_features_rejects_juice_sized_spreads() -> None:
    """American juice/ML dumps must not enable market-aware margin heads."""
    from web.basketball_v2_market import _coerce_home_spread

    assert _coerce_home_spread(-110) is None
    assert _coerce_home_spread(150) is None
    assert _coerce_home_spread("-110") is None
    assert _coerce_home_spread(45) is None  # beyond NBA/WNBA max
    assert _coerce_home_spread(-4.5) == -4.5

    for junk in (-110, 110, 150, "-105", 50):
        feats: dict[str, float] = {}
        _, has_spread = apply_market_features(feats, home_spread=junk)
        assert has_spread is False
        assert feats["has_spread"] == 0.0
        assert feats["mkt_home_spread"] == 0.0


def test_resolve_market_heads_variants() -> None:
    art = {
        "feature_columns": ["elo_diff"],
        "clf_market_features": ["mkt_home_prob", "has_market"],
        "margin_market_features": ["mkt_home_spread", "has_spread"],
        "clf_market": object(),
        "lr_market": {"ok": True},
        "calibrator_market": {"x": [0, 1], "y": [0, 1]},
        "margin_market": object(),
    }

    use_clf, use_margin, variant, clf_cols, margin_cols = resolve_market_heads(
        art, has_market=False, has_spread=False
    )
    assert (use_clf, use_margin, variant) == (False, False, "pure")
    assert clf_cols == ["elo_diff"]
    assert margin_cols == ["elo_diff"]

    use_clf, use_margin, variant, clf_cols, margin_cols = resolve_market_heads(
        art, has_market=True, has_spread=False
    )
    assert use_clf is True
    assert use_margin is False
    assert variant == "market_aware"
    assert clf_cols == ["elo_diff", "mkt_home_prob", "has_market"]
    assert margin_cols == ["elo_diff"]

    use_clf, use_margin, variant, clf_cols, margin_cols = resolve_market_heads(
        art, has_market=False, has_spread=True
    )
    assert use_clf is False
    assert use_margin is True
    assert variant == "market_aware"
    assert clf_cols == ["elo_diff"]
    assert margin_cols == ["elo_diff", "mkt_home_spread", "has_spread"]


def test_resolve_market_heads_defaults_missing_feature_lists() -> None:
    art = {
        "feature_columns": ["elo_diff"],
        "clf_market": object(),
        "lr_market": {},
        "calibrator_market": {},
        "margin_market": object(),
    }
    use_clf, use_margin, variant, clf_cols, margin_cols = resolve_market_heads(
        art, has_market=True, has_spread=True
    )
    assert use_clf and use_margin and variant == "market_aware"
    assert clf_cols[-4:] == ["mkt_home_prob", "has_market", "ml_steam_pp", "has_steam"]
    assert margin_cols[-4:] == ["mkt_home_spread", "has_spread", "spread_move", "has_steam"]
