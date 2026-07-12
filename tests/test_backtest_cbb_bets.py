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


def test_cbb_bet_policy_disabled_with_reason() -> None:
    import json

    assert POLICY_PATH.is_file()
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy.get("enabled") is False
    assert "reason" in policy
    close = policy.get("research_best_close") or {}
    assert (close.get("backtest") or {}).get("roi_pct", 0) < 0
