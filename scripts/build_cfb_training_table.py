"""Build the CFB v2 training table from closing-odds FBS history.

Replays data/supplemental/closing-odds/cfb.csv chronologically through
CfbFeatureEngine and writes one leak-free feature row per game to
data/cfb_history/training_table.csv (with market columns attached).

``FIRST_SEASON`` (2019) is the earliest ESPN-fetch-safe season documented in
``scripts/fetch_cfb_odds.py``. ``--emit-from-season`` defaults to the minimum
season in the odds CSV that has scores + a closing line (currently 2022 in
the checked-in cfb.csv); it lowers automatically when earlier rows are
backfilled.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.cfb_v2.feature_engine import FEATURE_COLUMNS, CfbFeatureEngine, cfb_season_of  # noqa: E402
from web.cfb_v2.replay import csv_rows_to_games, replay_games  # noqa: E402

ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "cfb.csv"
OUT_DIR = PROJECT_ROOT / "data" / "cfb_history"

# Earliest season we allow for emit/fetch expansion (see fetch_cfb_odds.EARLIEST_SAFE_SEASON).
FIRST_SEASON = 2019


def binary_home_win(home_score: int, away_score: int) -> int | None:
    """1/0 for decisive games; None for ties (exclude from classifier training)."""
    if home_score == away_score:
        return None
    return 1 if home_score > away_score else 0


def min_season_with_scores_and_odds(frame: pd.DataFrame) -> int | None:
    """Earliest CFB season that has finals + at least one closing market line."""
    if frame.empty or "season" not in frame.columns:
        return None
    scored = frame["home_final"].notna() & frame["away_final"].notna()
    has_close = frame["home_close_spread"].notna() | frame["home_close_ml"].notna()
    usable = frame.loc[scored & has_close, "season"]
    if usable.empty:
        usable = frame.loc[scored, "season"]
    if usable.empty:
        return None
    return int(usable.min())


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
    "home_close_ml",
    "away_close_ml",
    "home_close_spread",
    "away_close_spread",
    "home_spread_odds",
    "away_spread_odds",
    "home_open_spread",
    "away_open_spread",
    "close_total",
    "n_books",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build CFB v2 training table from FBS closing-odds history.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Writes data/cfb_history/training_table.csv. "
            "Requires data/supplemental/closing-odds/cfb.csv (or --odds-csv). "
            f"FIRST_SEASON floor={FIRST_SEASON}. "
            "Next: python scripts/train_cfb_model.py"
        ),
    )
    parser.add_argument(
        "--odds-csv",
        type=Path,
        default=ODDS_CSV,
        help="CFB closing-odds CSV",
    )
    parser.add_argument(
        "--emit-from-season",
        type=int,
        default=None,
        help=(
            "First season to write feature rows for "
            f"(default: min season with scores+odds in CSV, floored at {FIRST_SEASON})"
        ),
    )
    args = parser.parse_args()

    if not args.odds_csv.is_file():
        print(
            f"ERROR: missing odds CSV: {args.odds_csv}\n"
            "  Expected CFB closing-odds history at "
            "data/supplemental/closing-odds/cfb.csv\n"
            "  Pass --odds-csv PATH if the file lives elsewhere.",
            file=sys.stderr,
        )
        return 1

    frame = pd.read_csv(args.odds_csv)
    frame["day"] = pd.to_datetime(frame.date).dt.date
    frame["season"] = frame.day.map(cfb_season_of)
    frame = frame.sort_values(["day", "home_key", "away_key"]).reset_index(drop=True)

    csv_min = min_season_with_scores_and_odds(frame)
    if args.emit_from_season is None:
        if csv_min is None:
            print(
                "ERROR: odds CSV has no rows with scores to derive --emit-from-season",
                file=sys.stderr,
            )
            return 1
        emit_from = max(FIRST_SEASON, csv_min)
    else:
        emit_from = int(args.emit_from_season)
        if emit_from < FIRST_SEASON:
            print(
                f"WARNING: --emit-from-season {emit_from} is below FIRST_SEASON "
                f"{FIRST_SEASON} (ESPN-safe floor).",
                flush=True,
            )

    print(
        f"cfb.csv seasons {int(frame.season.min())}-{int(frame.season.max())}; "
        f"min scores+odds={csv_min}; emit_from={emit_from} (FIRST_SEASON={FIRST_SEASON})",
        flush=True,
    )

    rows_raw = frame.to_dict(orient="records")
    games = csv_rows_to_games(rows_raw)

    from web.cfb_v2.epa import attach_epa_priors_to_games  # noqa: E402
    from web.cfb_v2.rankings import attach_rankings_to_games  # noqa: E402

    attach_rankings_to_games(games)
    attach_epa_priors_to_games(games)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"

    engine = CfbFeatureEngine()
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(META_COLUMNS) + list(FEATURE_COLUMNS))
        writer.writeheader()

        def emit(game: dict, features: dict) -> None:
            nonlocal written
            if int(game["season"]) < emit_from:
                return
            home_score = int(game["home_score"])
            away_score = int(game["away_score"])
            home_win = binary_home_win(home_score, away_score)
            if home_win is None:
                return
            row = {
                "season": int(game["season"]),
                "date": game["date"],
                "home": game["home"],
                "away": game["away"],
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "margin": home_score - away_score,
                "total_points": home_score + away_score,
                "home_close_ml": game.get("home_close_ml"),
                "away_close_ml": game.get("away_close_ml"),
                "home_close_spread": game.get("home_close_spread"),
                "away_close_spread": game.get("away_close_spread"),
                "home_spread_odds": game.get("home_spread_odds"),
                "away_spread_odds": game.get("away_spread_odds"),
                "home_open_spread": game.get("home_open_spread"),
                "away_open_spread": game.get("away_open_spread"),
                "close_total": game.get("close_total"),
                "n_books": game.get("n_books"),
            }
            row.update({col: float(features[col]) for col in FEATURE_COLUMNS})
            writer.writerow(row)
            written += 1

        replay_games(engine, games, emit=emit)

    if written == 0:
        print(
            f"ERROR: wrote 0 rows to {out_path}\n"
            f"  Check --emit-from-season ({emit_from}) and odds CSV contents.",
            file=sys.stderr,
        )
        return 1
    print(f"wrote {written} rows -> {out_path}")
    print(f"n_features={len(FEATURE_COLUMNS)} teams_seen={len(engine.teams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
