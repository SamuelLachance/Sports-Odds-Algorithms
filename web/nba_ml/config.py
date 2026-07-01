"""Shared paths, feature schema, and flags for the NBA ML pilot."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ARTIFACT_DIR = PROJECT_ROOT / "data" / "models" / "nba"
MARGIN_MODEL_PATH = ARTIFACT_DIR / "margin_model.joblib"
WINPROB_MODEL_PATH = ARTIFACT_DIR / "winprob_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

DATASET_DIR = PROJECT_ROOT / "data" / "supplemental" / "nba-ml"
FEATURE_MATRIX_PATH = DATASET_DIR / "feature_matrix.parquet"
FEATURE_MATRIX_CSV = DATASET_DIR / "feature_matrix.csv"
BACKTEST_REPORT_PATH = DATASET_DIR / "backtest_report.json"
BACKTEST_REPORT_MD = DATASET_DIR / "backtest_report.md"

ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "nba.csv"
OPENING_ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "nba_open.csv"

# Neutral home-court edge baseline (points) used before model correction.
HOME_COURT_POINTS = 2.5

# Rolling-window lengths for point-in-time form features.
FORM_WINDOWS = (5, 10, 20)

# Elo configuration for the leak-free Elo feature.
ELO_BASE = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_MOV_MULT = True
ELO_SEASON_REGRESS = 0.25  # regress 25% toward mean between seasons

# Ordered feature columns fed to the models. Kept explicit so the daily
# inference path builds vectors in exactly the same order as training.
#
# Note: current 2K ratings and live ESPN injuries are intentionally excluded
# from the trained features because point-in-time historical values are not
# available (using today's values on old games would leak/stale the backtest).
# Injuries remain a small live-only post-model nudge via availability_signals.
FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "elo_win_prob",
    "off_eff_diff",
    "def_eff_diff",
    "net_eff_diff",
    "pace_sum",
    "form5_margin_diff",
    "form10_margin_diff",
    "form20_margin_diff",
    "form10_winrate_diff",
    "rest_diff",
    "home_b2b",
    "away_b2b",
    "home_three_in_four",
    "away_three_in_four",
    "travel_home_km",
    "travel_away_km",
    "tz_shift_diff",
    "season_progress",
    # Market-aware features (the sharpest public signal).
    "market_spread",
    "market_devig_home_prob",
)

# Feature subset available without any market line (used to detect whether
# the model is adding value beyond the market).
NON_MARKET_FEATURES: tuple[str, ...] = tuple(
    c for c in FEATURE_COLUMNS if not c.startswith("market_")
)


def nba_ml_artifacts_present() -> bool:
    """Whether trained NBA ML artifacts exist on disk."""
    return MARGIN_MODEL_PATH.is_file() and WINPROB_MODEL_PATH.is_file()


def nba_ml_enabled() -> bool:
    """Whether the LIVE prediction path should use the NBA ML model.

    Opt-in and off by default: the model is well-calibrated but does not beat
    the closing line out-of-sample, so we do not silently change the public
    board. Enable explicitly with ``NBA_ML_ENABLED=1`` (still requires
    artifacts). ``NBA_ML_DISABLED=1`` always wins.
    """
    if os.environ.get("NBA_ML_DISABLED") == "1":
        return False
    if os.environ.get("NBA_ML_ENABLED") == "1":
        return nba_ml_artifacts_present()
    return False
