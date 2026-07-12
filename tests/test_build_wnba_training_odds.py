"""WNBA training consensus must reject invalid even-book American medians."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_SCRIPT = PROJECT_ROOT / "scripts" / "build_wnba_training_table.py"
_SPEC = importlib.util.spec_from_file_location("build_wnba_training_table", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
consensus_odds = _MOD.consensus_odds


def test_consensus_rejects_even_book_median_dead_zone_for_juice_and_opens() -> None:
    """+100 vs −110 medians to −5 — must null spread juice and open MLs too."""
    rows = [
        {
            "home_ml": -110,
            "away_ml": -110,
            "home_spread": -3.5,
            "home_spread_odds": 100,
            "away_spread_odds": -110,
            "total": 160,
            "home_ml_open": 100,
            "away_ml_open": -110,
            "home_spread_open": -3,
        },
        {
            "home_ml": -105,
            "away_ml": -115,
            "home_spread": -3.5,
            "home_spread_odds": -110,
            "away_spread_odds": -110,
            "total": 161,
            "home_ml_open": -110,
            "away_ml_open": 100,
            "home_spread_open": -3.5,
        },
    ]
    c = consensus_odds(rows)
    assert c["home_ml"] == -107.5
    assert c["away_ml"] == -112.5
    assert c["spread_home_odds"] is None
    assert c["home_ml_open"] is None
    assert c["away_ml_open"] is None
    # Best price still picks a valid book, not the dead-zone median.
    assert c["best_spread_home_odds"] in (100, -110)
    assert abs(c["best_spread_home_odds"]) >= 100
