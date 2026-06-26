"""Soccer context layer unit tests (pure logic, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_context import (  # noqa: E402
    MatchContextSignals,
    TeamStyleProfile,
    apply_threeway_context_shifts,
    compute_context_adjustments,
    parse_form_score,
)


def test_parse_form_score() -> None:
    assert parse_form_score("WWWWW") == 1.0
    assert parse_form_score("LLLLL") == 0.0
    assert parse_form_score("WDL") == round(1.5 / 3, 4)
    assert parse_form_score("") is None
    assert parse_form_score(None) is None


def test_apply_threeway_context_shifts_renormalizes() -> None:
    home, draw, away = apply_threeway_context_shifts(40.0, 30.0, 30.0, 2.0, -1.0, -1.0)
    assert abs(home + draw + away - 100.0) < 0.02
    assert home > 40.0


def test_form_adjustment_favors_better_recent_form() -> None:
    signals = MatchContextSignals(home_form="WWWWW", away_form="LLLLL")
    home_shift, draw_shift, away_shift, factors = compute_context_adjustments(signals)
    assert home_shift > 0
    assert away_shift < 0
    assert factors and factors[0]["key"] == "form"


def test_neutral_venue_reduces_home_shift() -> None:
    signals = MatchContextSignals(neutral_site=True, venue_name="Neutral")
    home_shift, _, away_shift, factors = compute_context_adjustments(signals)
    assert home_shift < 0
    assert away_shift > 0
    assert any(f["key"] == "venue" for f in factors)


def test_style_profiles_produce_style_factor() -> None:
    signals = MatchContextSignals(
        home_style=TeamStyleProfile(
            games=5,
            possession_pct=58.0,
            shots_per_game=14.0,
            press_index=24.0,
        ),
        away_style=TeamStyleProfile(
            games=5,
            possession_pct=42.0,
            shots_per_game=9.0,
            press_index=16.0,
        ),
    )
    home_shift, _, _, factors = compute_context_adjustments(signals)
    assert home_shift > 0
    assert any(f["key"] == "style" for f in factors)


def test_injury_count_favors_healthier_side() -> None:
    signals = MatchContextSignals(home_injuries=0, away_injuries=4)
    home_shift, _, _, factors = compute_context_adjustments(signals)
    assert home_shift > 0
    assert any(f["key"] == "injuries" for f in factors)


if __name__ == "__main__":
    test_parse_form_score()
    test_apply_threeway_context_shifts_renormalizes()
    test_form_adjustment_favors_better_recent_form()
    test_neutral_venue_reduces_home_shift()
    test_style_profiles_produce_style_factor()
    test_injury_count_favors_healthier_side()
    print("test_soccer_context.py: all tests passed")
