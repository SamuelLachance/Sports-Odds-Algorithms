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


def test_cbb_odds_join_uses_canon_abbr_aliases() -> None:
    """Index keys are canon_abbr'd; join must map uconn→conn the same way."""
    idx = {
        ("2024-01-01", "conn", "unc"): {
            "home_close_ml": "-150",
            "away_close_ml": "130",
            "home_close_spread": "-4.5",
            "home_spread_odds": "-110",
            "away_spread_odds": "-110",
            "close_total": "145",
            "n_books": "2",
        }
    }
    out = _odds_for_game(
        idx, {"date": "2024-01-01", "home_abbr": "uconn", "away_abbr": "unc"}
    )
    assert out["home_ml"] == -150.0
    assert out["away_ml"] == 130.0
    assert out["home_spread"] == -4.5


def test_cbb_odds_join_maps_pk_spread_and_rejects_bool() -> None:
    """Handicap PK/EVEN is pick'em 0; bool False must not become EVEN/+100."""
    idx_pk = {
        ("2024-01-01", "duke", "unc"): {
            "home_close_ml": "-110",
            "away_close_ml": "100",
            "home_close_spread": "PK",
            "home_spread_odds": "-110",
            "away_spread_odds": "-110",
            "close_total": "140",
            "n_books": "2",
        }
    }
    out_pk = _odds_for_game(
        idx_pk, {"date": "2024-01-01", "home_abbr": "duke", "away_abbr": "unc"}
    )
    assert out_pk["home_spread"] == 0.0

    idx_bool = {
        ("2024-01-01", "duke", "unc"): {
            "home_close_ml": False,
            "away_close_ml": "-110",
            "home_close_spread": False,
            "home_spread_odds": "-110",
            "away_spread_odds": "-110",
            "close_total": "140",
            "n_books": "2",
        }
    }
    out_bool = _odds_for_game(
        idx_bool, {"date": "2024-01-01", "home_abbr": "duke", "away_abbr": "unc"}
    )
    assert out_bool["home_ml"] is None
    assert out_bool["home_spread"] is None
    assert out_bool["away_ml"] == -110.0


def test_cbb_odds_join_rejects_juice_as_spread_and_pk_total() -> None:
    """American juice in spread cells / PK totals must not poison training."""
    idx = {
        ("2024-01-01", "duke", "unc"): {
            "home_close_ml": "-150",
            "away_close_ml": "130",
            "home_close_spread": "-110",
            "home_spread_odds": "-110",
            "away_spread_odds": "-110",
            "close_total": "PK",
            "n_books": "2",
        }
    }
    out = _odds_for_game(
        idx, {"date": "2024-01-01", "home_abbr": "duke", "away_abbr": "unc"}
    )
    assert out["home_spread"] is None
    assert out["total_line"] is None
    assert out["home_ml"] == -150.0
