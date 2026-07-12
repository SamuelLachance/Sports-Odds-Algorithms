"""Unit tests for web.portfolio_sizing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.portfolio_sizing import portfolio_stake_units  # noqa: E402


def test_quarter_kelly_baseline_with_zero_penalty() -> None:
    # 4% full Kelly → 1u quarter-Kelly when bankroll_units=100, no haircut.
    assert portfolio_stake_units(5.0, 0.04, correlation_penalty=0.0) == 1.0


def test_correlation_penalty_is_more_conservative() -> None:
    raw = portfolio_stake_units(5.0, 0.04, correlation_penalty=0.0)
    haircut = portfolio_stake_units(5.0, 0.04, correlation_penalty=0.15)
    assert haircut < raw
    assert haircut == pytest.approx(raw * 0.85, abs=0.01)


def test_clamps_to_min_and_max() -> None:
    assert portfolio_stake_units(2.0, 0.005, correlation_penalty=0.0) == 0.25
    assert portfolio_stake_units(10.0, 0.25, correlation_penalty=0.0) == 3.0


def test_non_positive_kelly_returns_min() -> None:
    assert portfolio_stake_units(5.0, 0.0) == 0.25
    assert portfolio_stake_units(5.0, -0.1) == 0.25


def test_extreme_ev_dampens_stake() -> None:
    normal = portfolio_stake_units(8.0, 0.08, correlation_penalty=0.0)
    extreme = portfolio_stake_units(50.0, 0.08, correlation_penalty=0.0)
    assert extreme < normal


def test_custom_bankroll_scales_units() -> None:
    # kelly 4%, bankroll 50 → quarter-Kelly units = 0.5
    assert portfolio_stake_units(5.0, 0.04, bankroll_units=50, correlation_penalty=0.0) == 0.5


def test_invalid_inputs_return_min() -> None:
    assert portfolio_stake_units("bad", 0.04) == 0.25  # type: ignore[arg-type]


def test_invalid_correlation_penalty_returns_min_not_crash() -> None:
    """Bad penalty used to raise inside float() after Kelly/EV already parsed."""
    assert portfolio_stake_units(5.0, 0.04, correlation_penalty="nope") == 0.25  # type: ignore[arg-type]


def test_negative_ev_returns_min_not_kelly_size() -> None:
    # Positive Kelly with negative EV must not size up off Kelly alone.
    assert portfolio_stake_units(-5.0, 0.08, correlation_penalty=0.0) == 0.25


def test_non_finite_ev_or_kelly_returns_min() -> None:
    """NaN EV used to fail open (nan < 0 is False) and size a full Kelly stake."""
    assert portfolio_stake_units(float("nan"), 0.08, correlation_penalty=0.0) == 0.25
    assert portfolio_stake_units(float("inf"), 0.08, correlation_penalty=0.0) == 0.25
    assert portfolio_stake_units(5.0, float("nan"), correlation_penalty=0.0) == 0.25
    assert portfolio_stake_units(5.0, float("inf"), correlation_penalty=0.0) == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
