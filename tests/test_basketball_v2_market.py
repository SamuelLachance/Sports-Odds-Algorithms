"""Unit tests for shared NBA/WNBA v2 market-aware live helpers."""

from __future__ import annotations

import sys
from pathlib import Path

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
    assert clf_cols[-2:] == ["mkt_home_prob", "has_market"]
    assert margin_cols[-2:] == ["mkt_home_spread", "has_spread"]
