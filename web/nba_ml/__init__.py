"""NBA machine-learning pilot package.

A leak-free, market-aware NBA prediction stack:

- ``features``  : point-in-time feature engineering (no lookahead).
- ``dataset``   : builds the versioned training matrix from games + odds.
- ``model``     : XGBoost margin + win-probability models.
- ``calibrate`` : isotonic probability calibration + market-aware ensemble.
- ``backtest``  : nested walk-forward harness, CLV/ROI vs real closing lines.
- ``predict``   : cheap inference from committed model artifacts.

Heavy training runs offline; the daily site build only loads artifacts.
"""

from __future__ import annotations

from web.nba_ml.config import (
    ARTIFACT_DIR,
    FEATURE_COLUMNS,
    MARGIN_MODEL_PATH,
    METADATA_PATH,
    WINPROB_MODEL_PATH,
    nba_ml_artifacts_present,
    nba_ml_enabled,
)

__all__ = [
    "ARTIFACT_DIR",
    "FEATURE_COLUMNS",
    "MARGIN_MODEL_PATH",
    "METADATA_PATH",
    "WINPROB_MODEL_PATH",
    "nba_ml_artifacts_present",
    "nba_ml_enabled",
]
