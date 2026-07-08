"""Configuration for per-league ensemble ML mega-models."""

from __future__ import annotations

import os
from pathlib import Path

from web.hockey_pred_model import is_hockey_league
from web.cbb_pred_model import is_cbb_league
from web.wnba_pred_model import is_wnba_league
from web.mlb_pred_model import is_mlb_league
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

# NHL: lean on Poisson xG + market (meta stack is 100% sport_pred for NHL).
NHL_STACKING_FEATURES: tuple[str, ...] = (
    "sport_home_prob",
    "meta_stacked_home_prob",
    "market_devig_home_prob",
    "sport_minus_market",
    "expected_home_goals",
    "expected_away_goals",
    "sport_xg_diff",
    "overtime_probability",
    "power_home_prob",
    "market_home_ml",
    "market_away_ml",
)

# Authoritative closing-line sportsbook log-loss benchmarks (devigged moneyline).
SPORTSBOOK_LOGLOSS_BENCHMARKS: dict[str, float] = {
    "nba": 0.60166,
    "cbb": 0.53720,
    "wnba": 0.58730,
    "nhl": 0.66040,
    "mlb": 0.67374,
    "epl": 0.9667,
    "bundesliga": 0.9244,
    "laliga": 0.9938,
    "seriea": 0.9449,
    "ligue1": 0.9740,
    "worldcup": 0.7500,
    "fifa_friendlies": 0.7608,
    "uefa_euro": 0.7500,
    "uefa_nations": 0.7500,
    "copa_america": 0.8027,
    "concacaf_wcq": 0.7861,
    "concacaf_gold": 0.7356,
    "concacaf_nations": 0.7761,
}

# Back-compat aliases used by market-blend tuning.
NBA_TARGET_LOGLOSS = SPORTSBOOK_LOGLOSS_BENCHMARKS["nba"]
NHL_TARGET_LOGLOSS = SPORTSBOOK_LOGLOSS_BENCHMARKS["nhl"]
CBB_TARGET_LOGLOSS = SPORTSBOOK_LOGLOSS_BENCHMARKS["cbb"]
CBB_MARKET_BLEND_MAX = 0.35
WNBA_MARKET_BLEND_MAX = 0.35
MLB_MARKET_BLEND_MAX = 0.35
SOCCER_MARKET_BLEND_MAX = 0.35
MAX_DECORRELATION_WEIGHT = 0.35

CBB_STACKING_FEATURES: tuple[str, ...] = (
    "sport_home_prob",
    "meta_stacked_home_prob",
    "market_devig_home_prob",
    "sport_minus_market",
    "sport_margin",
    "market_spread",
    "market_home_ml",
    "market_away_ml",
)

# MLB: lean on sport/meta + closing line (same pattern as NHL).
MLB_STACKING_FEATURES: tuple[str, ...] = (
    "sport_home_prob",
    "meta_stacked_home_prob",
    "market_devig_home_prob",
    "sport_minus_market",
    "power_home_prob",
    "legacy_home_prob",
    "market_home_ml",
    "market_away_ml",
)

MLB_TARGET_LOGLOSS = SPORTSBOOK_LOGLOSS_BENCHMARKS["mlb"]


def sportsbook_logloss_benchmark(league: str) -> float | None:
    """Return the authoritative closing-line sportsbook log-loss target, if known."""
    return SPORTSBOOK_LOGLOSS_BENCHMARKS.get(league.lower())


LEAGUE_STACKING_FEATURES: dict[str, tuple[str, ...]] = {
    "nhl": NHL_STACKING_FEATURES,
    "mlb": MLB_STACKING_FEATURES,
    "cbb": CBB_STACKING_FEATURES,
}

DATASET_CACHE_DIR = PROJECT_ROOT / "data" / "supplemental" / "ensemble-ml"

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

# Full walk-forward history by default (no 250-game subsample cap).
DEFAULT_MAX_CALIBRATION_GAMES: int | None = None
DEFAULT_TARGET_ROWS: int | None = None
DEFAULT_POWER_TRAIN_WINDOW: int | None = None

# Legacy fast-training caps (opt-in via get_dataset_profile(fast=True)).
FAST_MAX_CALIBRATION_GAMES = 250
FAST_TARGET_ROWS = 250
FAST_POWER_TRAIN_WINDOW = 900

# Leagues that merge supplemental public history (archive CSV, football-data.co.uk).
SUPPLEMENTAL_DATED_LEAGUES: frozenset[str] = frozenset(
    {
        "nba",
        "nhl",
        "mlb",
        "nfl",
        "epl",
        "bundesliga",
        "laliga",
        "seriea",
        "ligue1",
        "worldcup",
        "fifa_friendlies",
        "concacaf_wcq",
        "concacaf_gold",
        "concacaf_nations",
        "uefa_euro",
        "uefa_nations",
        "copa_america",
    }
)


def get_stacking_features(league: str) -> tuple[str, ...]:
    return LEAGUE_STACKING_FEATURES.get(league.lower(), STACKING_FEATURES)


def dataset_cache_path(league: str) -> Path:
    return DATASET_CACHE_DIR / f"{league.lower()}_dataset.csv"


def get_dataset_profile(league: str, *, fast: bool = False) -> dict[str, int | None | str]:
    league = league.lower()
    if fast:
        base = {
            "max_calibration_games": FAST_MAX_CALIBRATION_GAMES,
            "target_rows": FAST_TARGET_ROWS,
            "power_train_window": FAST_POWER_TRAIN_WINDOW,
            "dated_source": "espn",
        }
    else:
        base = {
            "max_calibration_games": DEFAULT_MAX_CALIBRATION_GAMES,
            "target_rows": DEFAULT_TARGET_ROWS,
            "power_train_window": DEFAULT_POWER_TRAIN_WINDOW,
            "dated_source": (
                "supplemental"
                if league in SUPPLEMENTAL_DATED_LEAGUES
                else "espn"
            ),
        }
    return base


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
    if (
        is_hockey_league(league)
        or is_cbb_league(league)
        or is_wnba_league(league)
        or is_mlb_league(league)
        or is_soccer_league(league)
    ):
        return False
    return model_artifact_path(league).is_file()
