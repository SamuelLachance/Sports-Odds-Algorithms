"""Tests for the NBA ML pilot: features, odds parsing, backtest math, inference."""

from __future__ import annotations

import math
from datetime import date

import pytest

from web.nba_ml import nba_ml_artifacts_present, nba_ml_enabled
from web.nba_ml.backtest import _ats_result, _sane_american, american_profit
from web.nba_ml.features import (
    FeatureState,
    american_to_prob,
    devig_home_prob,
    spread_to_home_prob,
)


def test_american_to_prob_and_devig():
    assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-3)
    assert american_to_prob(+150) == pytest.approx(0.4, abs=1e-3)
    # ESPN EVEN (0) must match +100, not drop as None.
    assert american_to_prob(0) == pytest.approx(american_to_prob(100), abs=1e-9)
    # Invalid |odds| < 100 must not invent market probs.
    assert american_to_prob(50) is None
    assert american_to_prob(-50) is None
    assert devig_home_prob(-50, 130) is None
    # Devig removes the vig so probabilities sum to 1.
    p = devig_home_prob(-150, +130)
    assert 0.0 < p < 1.0
    assert devig_home_prob(None, +130) is None
    assert abs(devig_home_prob(0, -110) - devig_home_prob(100, -110)) < 1e-9


def test_spread_to_home_prob_monotonic():
    assert spread_to_home_prob(-10) > 0.5
    assert spread_to_home_prob(+10) < 0.5
    assert spread_to_home_prob(0) == pytest.approx(0.5, abs=1e-6)


def test_sane_american_rejects_corrupt_odds():
    # Decimal odds / garbage below magnitude 100 must not invent -110 juice.
    assert math.isnan(_sane_american(-1))
    assert math.isnan(_sane_american(1.87))
    assert _sane_american(-110) == -110.0
    assert _sane_american(+150) == 150.0
    # ESPN EVEN (0) maps to +100, not the -110 fallback.
    assert _sane_american(0) == 100.0
    from web.nba_ml.backtest import implied_prob

    assert implied_prob(0) == pytest.approx(0.5)
    # Corrupt -1 must not pay ~100u or fake -110 profit.
    assert math.isnan(american_profit(-1, True))
    assert math.isnan(implied_prob(-1))


def test_ats_grading():
    # Home -6, wins by 10 -> covers.
    assert _ats_result("home", 10, -6) == "win"
    assert _ats_result("away", 10, -6) == "loss"
    # Exact push.
    assert _ats_result("home", 6, -6) == "push"


def test_feature_state_is_leak_free_and_updates():
    state = FeatureState()
    day = date(2024, 1, 1)
    # First matchup: no history -> neutral-ish features, elo diff only from HFA.
    feats = state.features_for("bos", "lal", day)
    assert set(feats).issuperset({"elo_diff", "market_devig_home_prob"})
    assert feats["form10_margin_diff"] == 0.0
    state.update("bos", "lal", day, 110, 100)
    # After a home win, home Elo should exceed away Elo.
    assert state.teams["bos"].elo > state.teams["lal"].elo
    later = state.features_for("bos", "lal", date(2024, 1, 3))
    assert later["form5_margin_diff"] != 0.0
    # Both teams last played Jan 1, so equal rest (2 days) -> zero differential.
    assert later["rest_diff"] == pytest.approx(0.0)


def test_haversine_travel_present_for_cross_country():
    state = FeatureState()
    feats = state.features_for("bos", "lal", date(2024, 1, 1))
    # Away team traveling from LAL arena to BOS arena -> large distance.
    assert feats["travel_away_km"] > 3000


def test_flag_default_off_but_artifacts_present():
    # Live flag is opt-in; default off regardless of artifacts.
    assert nba_ml_enabled() is False
    # Artifacts should be committed with the pilot.
    assert nba_ml_artifacts_present() is True


def test_predict_nba_smoke():
    if not nba_ml_artifacts_present():
        pytest.skip("NBA ML artifacts not present")
    from web.nba_ml.predict import predict_nba

    out = predict_nba("bos", "lal", cutoff_date="2025-01-15", market_spread=-6.5,
                      home_ml=-260, away_ml=210)
    assert out is not None
    assert 0.0 <= out["home_win_probability"] <= 100.0
    assert out["model_market_edge_pp"] is not None
    assert predict_nba("zzz", "lal") is None
