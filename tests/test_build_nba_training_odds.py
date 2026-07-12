"""NBA training-table odds join must refuse ambiguous ±1 day neighbors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_SCRIPT = PROJECT_ROOT / "scripts" / "build_nba_training_table.py"
_SPEC = importlib.util.spec_from_file_location("build_nba_training_table", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_odds_for_game = _MOD._odds_for_game


def test_odds_for_game_refuses_consecutive_same_matchup_series() -> None:
    index = {
        ("2025-06-30", "bos", "nyk"): {"home_ml": -110, "away_ml": -110},
        ("2025-07-02", "bos", "nyk"): {"home_ml": -145, "away_ml": 125},
    }
    assert _odds_for_game(index, {"date": "2025-07-01", "home": "bos", "away": "nyk"}) == {}


def test_odds_for_game_uses_single_neighbor() -> None:
    index = {
        ("2025-07-02", "bos", "nyk"): {
            "home_ml": -145,
            "away_ml": 125,
            "home_spread": -3.5,
            "home_spread_odds": -110,
            "away_spread_odds": -110,
            "total": 220.5,
            "n_books": 3,
            "home_spread_open": -3.0,
        },
    }
    row = _odds_for_game(index, {"date": "2025-07-01", "home": "bos", "away": "nyk"})
    assert row["home_ml"] == -145
    assert row["away_ml"] == 125
