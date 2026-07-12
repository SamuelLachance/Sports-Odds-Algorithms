"""Football training tables must not label ties as home losses."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nfl_binary_home_win_skips_ties() -> None:
    mod = _load("build_nfl_training_table.py")
    assert mod.binary_home_win(24, 17) == 1
    assert mod.binary_home_win(10, 27) == 0
    assert mod.binary_home_win(20, 20) is None


def test_cfb_binary_home_win_skips_ties() -> None:
    mod = _load("build_cfb_training_table.py")
    assert mod.binary_home_win(31, 28) == 1
    assert mod.binary_home_win(14, 21) == 0
    assert mod.binary_home_win(17, 17) is None


def test_mlb_v2_binary_home_win_skips_ties() -> None:
    """MLB v2 must not label ties as home_win=0 (away win) like the feature engine skip."""
    mod = _load("build_mlb_training_table.py")
    assert mod.binary_home_win(5, 3) == 1
    assert mod.binary_home_win(2, 7) == 0
    assert mod.binary_home_win(3, 3) is None
