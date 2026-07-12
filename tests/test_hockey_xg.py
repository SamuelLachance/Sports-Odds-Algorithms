"""Regression tests for college hockey SOS-adjusted xG rates."""

from __future__ import annotations

from dataclasses import dataclass

from web.hockey_xg import _college_sos_adjusted_rates


@dataclass
class _Rates:
    goals_for_pg: float
    goals_against_pg: float


def test_college_sos_credits_team_facing_elite_offenses() -> None:
    """Facing above-average offenses must inflate GF and deflate GA (not invert)."""
    team_metrics = {
        "yale": _Rates(3.0, 3.0),
        "elite1": _Rates(4.0, 2.0),
        "elite2": _Rates(4.0, 2.0),
        "avg": _Rates(2.5, 2.5),
    }
    games = [
        ("2024-01-01", ("yale", "elite1", "Yale", "E1", 2, 4)),
        ("2024-01-08", ("elite2", "yale", "E2", "Yale", 4, 1)),
    ]
    gf, ga = _college_sos_adjusted_rates("yale", team_metrics, games)
    assert gf > 3.0
    assert ga < 3.0
