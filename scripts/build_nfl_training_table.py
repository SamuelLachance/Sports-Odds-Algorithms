"""Build the NFL v2 training table from nflverse closing-odds history.

Replays data/supplemental/closing-odds/nflverse_games.csv chronologically
through NflFeatureEngine and writes one leak-free feature row per game to
data/nfl_history/training_table.csv (with market columns attached).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nfl_v2.feature_engine import FEATURE_COLUMNS, NflFeatureEngine  # noqa: E402
from web.nfl_v2.replay import nflverse_rows_to_games, replay_games  # noqa: E402

ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "nflverse_games.csv"
OUT_DIR = PROJECT_ROOT / "data" / "nfl_history"

META_COLUMNS = (
    "season",
    "date",
    "home",
    "away",
    "home_score",
    "away_score",
    "home_win",
    "margin",
    "total_points",
    "week",
    "game_type",
    "home_close_ml",
    "away_close_ml",
    "home_close_spread",
    "away_close_spread",
    "home_spread_odds",
    "away_spread_odds",
    "close_total",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build NFL v2 training table from nflverse closing-odds history.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Writes data/nfl_history/training_table.csv. "
            "Requires data/supplemental/closing-odds/nflverse_games.csv "
            "(or --odds-csv). Next: python scripts/train_nfl_model.py"
        ),
    )
    parser.add_argument(
        "--odds-csv",
        type=Path,
        default=ODDS_CSV,
        help="nflverse closing-odds CSV",
    )
    parser.add_argument(
        "--emit-from-season",
        type=int,
        default=1999,
        help="First season to write feature rows for",
    )
    args = parser.parse_args()

    if not args.odds_csv.is_file():
        print(
            f"ERROR: missing odds CSV: {args.odds_csv}\n"
            "  Expected nflverse closing-odds history at "
            "data/supplemental/closing-odds/nflverse_games.csv\n"
            "  Pass --odds-csv PATH if the file lives elsewhere.",
            file=sys.stderr,
        )
        return 1

    frame = pd.read_csv(args.odds_csv)
    frame = frame[frame.game_type.isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
    frame = frame.dropna(subset=["home_score", "away_score"])
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    rows_raw = frame.to_dict(orient="records")
    games = nflverse_rows_to_games(rows_raw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"

    engine = NflFeatureEngine()
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(META_COLUMNS) + list(FEATURE_COLUMNS))
        writer.writeheader()

        def emit(game: dict, features: dict) -> None:
            nonlocal written
            if int(game["season"]) < args.emit_from_season:
                return
            home_score = int(game["home_score"])
            away_score = int(game["away_score"])
            row = {
                "season": int(game["season"]),
                "date": game["date"],
                "home": game["home"],
                "away": game["away"],
                "home_score": home_score,
                "away_score": away_score,
                "home_win": 1 if home_score > away_score else 0,
                "margin": home_score - away_score,
                "total_points": home_score + away_score,
                "week": game.get("week"),
                "game_type": game.get("game_type"),
                "home_close_ml": game.get("home_close_ml"),
                "away_close_ml": game.get("away_close_ml"),
                "home_close_spread": game.get("home_close_spread"),
                "away_close_spread": game.get("away_close_spread"),
                "home_spread_odds": game.get("home_spread_odds"),
                "away_spread_odds": game.get("away_spread_odds"),
                "close_total": game.get("close_total"),
            }
            row.update({col: float(features[col]) for col in FEATURE_COLUMNS})
            writer.writerow(row)
            written += 1

        replay_games(engine, games, emit=emit)

    if written == 0:
        print(
            f"ERROR: wrote 0 rows to {out_path}\n"
            f"  Check --emit-from-season ({args.emit_from_season}) and odds CSV contents.",
            file=sys.stderr,
        )
        return 1
    print(f"wrote {written} rows -> {out_path}")
    print(f"n_features={len(FEATURE_COLUMNS)} teams_seen={len(engine.teams)} qbs={len(engine.qbs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
