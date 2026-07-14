"""Smoke tests for scripts/backtest_cbb_bets.py import surface."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "backtest_cbb_bets.py"
POLICY_PATH = PROJECT_ROOT / "data" / "models" / "cbb_v2" / "bet_policy.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("backtest_cbb_bets", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_backtest_cbb_bets_imports() -> None:
    assert SCRIPT_PATH.is_file()
    module = _load_module()
    assert callable(module.american_to_decimal)
    assert callable(module.kelly_units)
    assert callable(module.load_odds)
    assert callable(module.build_side_table)
    assert module.LEAGUE == "cbb"
    assert module.KELLY_FRACTION == 0.25


def test_backtest_cbb_bets_helpers() -> None:
    module = _load_module()
    assert module.american_to_decimal(-110) == pytest.approx(1.9090909, rel=1e-5)
    assert module.american_to_decimal(150) == pytest.approx(2.5)
    assert module.kelly_units(0.55, -110) > 0
    assert module.kelly_units(0.40, -110) == 0.0


def test_build_side_table_rejects_garbage_juice_instead_of_inventing_minus_110() -> None:
    """Missing / |odds|<100 must drop the side — not coerce to −110 EV."""
    import pandas as pd

    module = _load_module()
    preds = pd.DataFrame(
        {
            "season": [2024, 2024],
            "model_margin": [3.0, -2.0],
            "margin": [7.0, -4.0],
            "home_spread": [-3.5, 2.5],
            "home_spread_open": [-3.0, 2.0],
            "spread_home_odds": [5.0, -110.0],  # garbage juice on row 0
            "spread_away_odds": [float("nan"), -105.0],
        }
    )
    sides = module.build_side_table(preds, sigma=12.0, exec_price="close")
    # Row 0: both sides invalid/missing → dropped. Row 1: both valid → 2 sides.
    assert len(sides) == 2
    assert set(sides["odds"].tolist()) == {-110.0, -105.0}
    assert bool((sides.odds.abs() >= 100).all())


def test_cbb_bet_policy_enabled_all_seasons_positive() -> None:
    import json

    assert POLICY_PATH.is_file()
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    # The enable flag flips with re-validation state (currently disabled pending
    # leak-free backtests) — require an explicit bool so gating never fails open.
    assert isinstance(policy.get("enabled"), bool)
    assert policy.get("bet_type") == "moneyline"
    bt = policy.get("backtest") or {}
    assert bt.get("seasons_total", 0) >= 3
    assert bt.get("worst_season_roi", -1) > 0

