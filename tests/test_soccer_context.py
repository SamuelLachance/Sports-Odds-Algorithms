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


def test_path_a_context_layer_syncs_blended_home_and_locks_daily_hook() -> None:
    """ESPN Path A context must update blended_home and set context_adjustment_pp."""
    from unittest.mock import patch

    from web.soccer_blend import _attach_soccer_context_layer

    base = {
        "home_win_probability": 45.0,
        "draw_probability": 28.0,
        "away_win_probability": 27.0,
        "blended_home_win_probability": 45.0,
        "blended_threeway": {
            "home_win_probability": 45.0,
            "draw_probability": 28.0,
            "away_win_probability": 27.0,
        },
        "soccer_pred": {
            "home_win_probability": 45.0,
            "draw_probability": 28.0,
            "away_win_probability": 27.0,
        },
        "total_score": -45.0,
        "win_probability": 45.0,
        "favorite_side": "away",
    }
    fake_context = {
        "home_win_probability": 48.5,
        "draw_probability": 27.0,
        "away_win_probability": 24.5,
        "factors": [{"key": "form", "label": "Form"}],
    }
    with patch(
        "web.soccer_blend.build_soccer_context_payload", return_value=fake_context
    ):
        out = _attach_soccer_context_layer(
            base,
            league="epl",
            cutoff_date="7-10-2026",
            home_abbr="ars",
            away_abbr="liv",
        )
    assert out["home_win_probability"] == 48.5
    assert out["blended_home_win_probability"] == 48.5
    assert out["blended_threeway"]["home_win_probability"] == 48.5
    assert out["soccer_pred"]["home_win_probability"] == 48.5
    assert out["context_adjusted"] is True
    assert out["context_adjustment_pp"] == 0.0


def test_soccer_v2_display_context_locks_daily_hook_without_mutating_probs() -> None:
    from unittest.mock import patch

    from web.soccer_blend import attach_soccer_context_display

    base = {
        "home_win_probability": 51.2,
        "draw_probability": 26.4,
        "away_win_probability": 22.4,
    }
    with patch(
        "web.soccer_blend.build_soccer_context_payload",
        return_value={
            "home_win_probability": 40.0,
            "draw_probability": 30.0,
            "away_win_probability": 30.0,
            "factors": [{"key": "injuries"}],
        },
    ):
        out = attach_soccer_context_display(
            base,
            league="epl",
            cutoff_date="7-10-2026",
            home_abbr="ars",
            away_abbr="liv",
        )
    assert out["home_win_probability"] == 51.2
    assert out["context_adjustment_pp"] == 0.0
    assert out["soccer_context"]["factors"][0]["key"] == "injuries"

    with patch("web.soccer_blend.build_soccer_context_payload", return_value=None):
        locked = attach_soccer_context_display(
            base,
            league="epl",
            cutoff_date="7-10-2026",
            home_abbr="ars",
            away_abbr="liv",
        )
    assert locked["context_adjustment_pp"] == 0.0
    assert "soccer_context" not in locked


if __name__ == "__main__":
    test_parse_form_score()
    test_apply_threeway_context_shifts_renormalizes()
    test_form_adjustment_favors_better_recent_form()
    test_neutral_venue_reduces_home_shift()
    test_style_profiles_produce_style_factor()
    test_injury_count_favors_healthier_side()
    test_path_a_context_layer_syncs_blended_home_and_locks_daily_hook()
    test_soccer_v2_display_context_locks_daily_hook_without_mutating_probs()
    print("test_soccer_context.py: all tests passed")
