"""Configuration for per-league ensemble ML mega-models."""

from __future__ import annotations

import os
from pathlib import Path

from web.league_profiles import LEAGUE_PROFILES, is_soccer_league, uses_spread_bets

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ROOT = PROJECT_ROOT / "data" / "models" / "ensemble"

# Train / apply on all supported leagues (soccer uses 3-way heads).
TRAIN_LEAGUES: tuple[str, ...] = tuple(
    key for key in LEAGUE_PROFILES if key not in {"dwl", "pwl", "vwl", "lmp", "wbc"}
)

STACKING_FEATURES: tuple[str, ...] = (
    "legacy_home_prob",
    "power_home_prob",
    "sport_home_prob",
    "meta_stacked_home_prob",
    "legacy_margin",
    "power_margin",
    "sport_margin",
    "market_devig_home_prob",
    "market_spread",
    "market_home_ml",
    "market_away_ml",
)

SOCCER_STACKING_FEATURES: tuple[str, ...] = (
    "legacy_home_prob",
    "legacy_draw_prob",
    "legacy_away_prob",
    "power_home_prob",
    "power_draw_prob",
    "power_away_prob",
    "sport_home_prob",
    "sport_draw_prob",
    "sport_away_prob",
    "meta_home_prob",
    "meta_draw_prob",
    "meta_away_prob",
    "market_devig_home_prob",
    "market_devig_draw_prob",
    "market_devig_away_prob",
)

DEFAULT_MARGIN_SIGMA: dict[str, float] = {
    "nba": 13.0,
    "wnba": 11.0,
    "cbb": 12.0,
    "nfl": 14.0,
    "cfb": 15.0,
    "nhl": 2.5,
    "mlb": 3.5,
    "ncaabb": 4.0,
}

MIN_TRAIN_ROWS = 80
CALIBRATION_FRACTION = 0.2

# Per-league dataset build settings (defaults keep training fast for most sports).
DEFAULT_MAX_CALIBRATION_GAMES = 250
DEFAULT_TARGET_ROWS = 250
DEFAULT_POWER_TRAIN_WINDOW = 900

# NHL: use every walk-forward game with full prior history (max training rows).
LEAGUE_DATASET_OVERRIDES: dict[str, dict[str, int | None | str]] = {
    "nhl": {
        "max_calibration_games": None,
        "target_rows": None,
        "power_train_window": None,
        "dated_source": "espn",
    },
}


def get_dataset_profile(league: str) -> dict[str, int | None | str]:
    league = league.lower()
    base = {
        "max_calibration_games": DEFAULT_MAX_CALIBRATION_GAMES,
        "target_rows": DEFAULT_TARGET_ROWS,
        "power_train_window": DEFAULT_POWER_TRAIN_WINDOW,
        "dated_source": "espn",
    }
    override = LEAGUE_DATASET_OVERRIDES.get(league, {})
    return {**base, **override}


def model_dir(league: str) -> Path:
    return MODEL_ROOT / league.lower()


def model_artifact_path(league: str) -> Path:
    return model_dir(league) / "ensemble.joblib"


def metadata_path(league: str) -> Path:
    return model_dir(league) / "metadata.json"


def is_spread_league(league: str) -> bool:
    return uses_spread_bets(league.lower())


def ensemble_enabled() -> bool:
    """Global kill-switch (default on). Set ENSEMBLE_ML_ENABLED=0 to disable."""
    return os.environ.get("ENSEMBLE_ML_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def ensemble_model_available(league: str) -> bool:
    if not ensemble_enabled():
        return False
    league = league.lower()
    if is_soccer_league(league):
        return metadata_path(league).is_file() and model_artifact_path(league).is_file()
    return model_artifact_path(league).is_file()
