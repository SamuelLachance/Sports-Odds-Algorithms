"""SBR closing-line cell parsers must keep EVEN/pick'em (not falsy-or drop)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_ml.dataset import _spread_juice  # noqa: E402
from web.sbr_odds import (  # noqa: E402
    _american_ml_cell,
    _spread_juice_cell,
    _spread_line_cell,
)


def test_sbr_american_ml_keeps_even_zero_as_plus_100() -> None:
    assert _american_ml_cell("0") == 100
    assert _american_ml_cell("-110") == -110
    assert _american_ml_cell("") is None
    assert _american_ml_cell("NL") is None


def test_sbr_spread_line_keeps_pickem_zero() -> None:
    assert _spread_line_cell("0") == 0.0
    assert _spread_line_cell("pk") == 0.0
    assert _spread_line_cell("PK") == 0.0
    assert _spread_line_cell("-3.5") == -3.5
    assert _spread_line_cell("") is None


def test_sbr_spread_juice_maps_even_zero() -> None:
    assert _spread_juice_cell("0") == 100
    assert _spread_juice_cell("") == -110
    assert _spread_juice_cell("-105") == -105


def test_nba_ml_dataset_spread_juice_maps_even_zero() -> None:
    assert _spread_juice(0) == 100.0
    assert _spread_juice(None) == -110.0
    assert _spread_juice(-105) == -105.0
