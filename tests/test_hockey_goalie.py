"""Regression tests for NHL / college goalie xG multipliers."""

from __future__ import annotations

from web.hockey_goalie import _college_goalie_proxy, _sv_to_xg_multiplier, goalie_xg_multipliers


def test_better_goalie_suppresses_opponent_xg() -> None:
    """Elite SV% vs team baseline must shrink opponent xG (mult < 1)."""
    better = _sv_to_xg_multiplier(0.920, 0.905)
    worse = _sv_to_xg_multiplier(0.890, 0.905)
    assert better < 1.0
    assert worse > 1.0
    assert better < worse


def test_college_partial_ga_uses_opponent_defense() -> None:
    """Missing away GA must not invert home/away multipliers vs both-present path."""
    rates = {"yale": 2.0, "princeton": 3.0, "cornell": 4.0}
    home_mult, away_mult, meta = goalie_xg_multipliers(
        "ncaah",
        "yale",
        "harvard",
        team_ga_rates=rates,
    )
    assert meta.get("league_avg_ga") == 3.0
    # Unknown away GA → home offense uses league avg (mult 1.0).
    assert home_mult == 1.0
    # Known home GA 2.0 → suppresses away xG.
    assert away_mult == _college_goalie_proxy(2.0, 3.0)


def test_college_both_ga_matches_opponent_defense_convention() -> None:
    rates = {"yale": 2.0, "harvard": 4.0, "princeton": 3.0}
    home_mult, away_mult, _meta = goalie_xg_multipliers(
        "ncaah",
        "yale",
        "harvard",
        team_ga_rates=rates,
    )
    # home_xg scaled by away GA; away_xg scaled by home GA.
    assert home_mult == _college_goalie_proxy(4.0, 3.0)
    assert away_mult == _college_goalie_proxy(2.0, 3.0)
