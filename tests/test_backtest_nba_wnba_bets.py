"""Regression: NBA/WNBA spread backtests must not invent −110 juice."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load(script_name: str):
    path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script_name",
    ["backtest_nba_bets.py", "backtest_wnba_bets.py"],
)
def test_valid_spread_juice_rejects_garbage(script_name: str) -> None:
    module = _load(script_name)
    assert module._valid_spread_juice(None) is None
    assert module._valid_spread_juice(float("nan")) is None
    assert module._valid_spread_juice(5.0) is None
    assert module._valid_spread_juice(-50) is None
    assert module._valid_spread_juice(0) == 100.0
    assert module._valid_spread_juice(-110) == -110.0
    assert module._valid_spread_juice(105) == 105.0


@pytest.mark.parametrize(
    "script_name",
    ["backtest_nba_bets.py", "backtest_wnba_bets.py"],
)
def test_simulate_spread_skips_missing_juice_instead_of_inventing_minus_110(
    script_name: str,
) -> None:
    """Missing / |odds|<100 must drop the game — not coerce to −110 EV."""
    module = _load(script_name)
    garbage_only = pd.DataFrame(
        {
            "season": [2024],
            "model_margin": [8.0],
            "margin": [12.0],
            "home_spread": [-3.5],
            "home_spread_open": [-3.0],
            "spread_home_odds": [5.0],
            "spread_away_odds": [float("nan")],
            "best_spread_home_odds": [float("nan")],
            "best_spread_away_odds": [float("nan")],
        }
    )
    empty = module.simulate_spread(
        garbage_only,
        margin_col="model_margin",
        sigma=12.0,
        min_cover_gap_pp=-999.0,
        min_point_edge=-999.0,
        exec_price="consensus",
    )
    # Pre-fix this row would invent −110/−110 and emit bets.
    assert empty.get("bets", 0) == 0

    valid = pd.DataFrame(
        {
            "season": [2024],
            "model_margin": [8.0],
            "margin": [12.0],
            "home_spread": [-3.5],
            "home_spread_open": [-3.0],
            "spread_home_odds": [-110.0],
            "spread_away_odds": [-110.0],
            "best_spread_home_odds": [float("nan")],
            "best_spread_away_odds": [float("nan")],
        }
    )
    result = module.simulate_spread(
        valid,
        margin_col="model_margin",
        sigma=12.0,
        min_cover_gap_pp=-999.0,
        min_point_edge=-999.0,
        exec_price="consensus",
    )
    assert result.get("bets", 0) >= 1
