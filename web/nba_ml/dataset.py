"""Build the versioned NBA training matrix from the real odds+scores table.

Iterates games chronologically, emitting point-in-time features and targets for
each game *before* updating state with its result (no lookahead).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from web.nba_ml.config import (
    DATASET_DIR,
    FEATURE_COLUMNS,
    FEATURE_MATRIX_CSV,
    FEATURE_MATRIX_PATH,
    ODDS_CSV,
)
from web.nba_ml.features import FeatureState, parse_game_date

META_COLUMNS = (
    "date",
    "season",
    "home_key",
    "away_key",
    "home_final",
    "away_final",
    "home_margin",
    "home_win",
    "n_prior_games",
    "home_close_spread",
    "home_open_spread",
    "home_close_ml",
    "away_close_ml",
    "home_spread_odds",
    "away_spread_odds",
    "n_books",
)


def _num(value):
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _spread_juice(value, default: float = -110.0) -> float:
    """Spread juice defaulting to -110; ESPN EVEN (0) maps to +100."""
    odds = _num(value)
    if odds is None:
        return default
    if odds == 0:
        return 100.0
    return odds


def build_dataset(odds_csv: Path = ODDS_CSV) -> pd.DataFrame:
    if not odds_csv.is_file():
        raise FileNotFoundError(f"Missing NBA odds table: {odds_csv}")

    raw = pd.read_csv(odds_csv)
    raw = raw.dropna(subset=["home_final", "away_final"])
    raw = raw.sort_values("date").reset_index(drop=True)

    state = FeatureState()
    records: list[dict] = []

    for _, row in raw.iterrows():
        game_day = parse_game_date(row["date"])
        home = str(row["home_key"]).lower()
        away = str(row["away_key"]).lower()
        home_score = int(row["home_final"])
        away_score = int(row["away_final"])

        market_spread = _num(row.get("home_close_spread"))
        home_ml = _num(row.get("home_close_ml"))
        away_ml = _num(row.get("away_close_ml"))

        n_prior = min(state.teams[home].games, state.teams[away].games)

        feats = state.features_for(
            home,
            away,
            game_day,
            market_spread=market_spread,
            market_home_ml=home_ml,
            market_away_ml=away_ml,
        )

        record = dict(feats)
        record.update(
            {
                "date": row["date"],
                "season": game_day.year if game_day.month >= 9 else game_day.year - 1,
                "home_key": home,
                "away_key": away,
                "home_final": home_score,
                "away_final": away_score,
                "home_margin": home_score - away_score,
                "home_win": 1 if home_score > away_score else 0,
                "n_prior_games": n_prior,
                "home_close_spread": market_spread,
                "home_open_spread": _num(row.get("home_open_spread")),
                "home_close_ml": home_ml,
                "away_close_ml": away_ml,
                "home_spread_odds": _spread_juice(row.get("home_spread_odds")),
                "away_spread_odds": _spread_juice(row.get("away_spread_odds")),
                "n_books": _num(row.get("n_books")) or 1,
            }
        )
        records.append(record)

        state.update(home, away, game_day, home_score, away_score)

    frame = pd.DataFrame.from_records(records)
    ordered = list(FEATURE_COLUMNS) + [c for c in META_COLUMNS if c not in FEATURE_COLUMNS]
    return frame[ordered]


def save_dataset(frame: pd.DataFrame) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(FEATURE_MATRIX_PATH, index=False)
        return FEATURE_MATRIX_PATH
    except Exception:  # noqa: BLE001 - parquet engine optional
        frame.to_csv(FEATURE_MATRIX_CSV, index=False)
        return FEATURE_MATRIX_CSV


def load_dataset() -> pd.DataFrame:
    if FEATURE_MATRIX_PATH.is_file():
        return pd.read_parquet(FEATURE_MATRIX_PATH)
    if FEATURE_MATRIX_CSV.is_file():
        return pd.read_csv(FEATURE_MATRIX_CSV)
    raise FileNotFoundError("No NBA feature matrix built yet. Run build_dataset + save_dataset.")


if __name__ == "__main__":
    df = build_dataset()
    path = save_dataset(df)
    print(f"Built {len(df)} rows -> {path}")
    print(df[list(FEATURE_COLUMNS)].describe().T[["mean", "std", "min", "max"]])
