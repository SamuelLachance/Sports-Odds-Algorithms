"""CBB training odds join must normalize EVEN and reject non-finite values."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_SCRIPT = PROJECT_ROOT / "scripts" / "build_cbb_training_table.py"
_SPEC = importlib.util.spec_from_file_location("build_cbb_training_table", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_odds_for_game = _MOD._odds_for_game


def test_cbb_odds_join_maps_even_text_and_zero() -> None:
    idx = {
        ("2024-01-01", "duke", "unc"): {
            "home_close_ml": "EVEN",
            "away_close_ml": "0",
            "home_close_spread": "-3.5",
            "home_spread_odds": "-110",
            "away_spread_odds": "PK",
            "close_total": "140",
            "n_books": "2",
        }
    }
    out = _odds_for_game(idx, {"date": "2024-01-01", "home_abbr": "duke", "away_abbr": "unc"})
    assert out["home_ml"] == 100.0
    assert out["away_ml"] == 100.0
    assert out["spread_away_odds"] == 100.0
    assert out["home_spread"] == -3.5


def test_cbb_odds_join_rejects_nan_moneyline_and_spread() -> None:
    idx = {
        ("2024-01-01", "duke", "unc"): {
            "home_close_ml": "nan",
            "away_close_ml": "-110",
            "home_close_spread": "nan",
            "home_spread_odds": "-110",
            "away_spread_odds": "-110",
            "close_total": "140",
            "n_books": "2",
        }
    }
    out = _odds_for_game(idx, {"date": "2024-01-01", "home_abbr": "duke", "away_abbr": "unc"})
    assert out["home_ml"] is None
    assert out["away_ml"] == -110.0
    assert out["home_spread"] is None
    assert out["home_spread"] is None or math.isfinite(out["home_spread"])
