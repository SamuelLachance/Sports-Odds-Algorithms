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
OPENS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "nfl.csv"
OUT_DIR = PROJECT_ROOT / "data" / "nfl_history"


def binary_home_win(home_score: int, away_score: int) -> int | None:
    """1/0 for decisive games; None for ties (exclude from classifier training)."""
    if home_score == away_score:
        return None
    return 1 if home_score > away_score else 0


def _attach_opening_lines(games: list[dict]) -> list[dict]:
    """Join SBR / Odds-API opens from nfl.csv when present; never invent open=close."""
    if not OPENS_CSV.is_file():
        return games
    try:
        opens = pd.read_csv(OPENS_CSV)
    except (OSError, ValueError):
        return games
    needed = {"date", "home_key", "away_key"}
    if opens.empty or not needed.issubset(set(opens.columns)):
        return games
    opens = opens.copy()
    opens["day"] = opens["date"].astype(str).str[:10]
    opens["home"] = (
        opens["home_key"].astype(str).str.lower().str.strip().replace({"wsh": "was", "lar": "la"})
    )
    opens["away"] = (
        opens["away_key"].astype(str).str.lower().str.strip().replace({"wsh": "was", "lar": "la"})
    )
    has_open = any(c in opens.columns for c in ("home_open_ml", "home_open_spread", "open_total"))
    if not has_open:
        return games
    lookup: dict[tuple[str, str, str], dict] = {}
    for row in opens.to_dict(orient="records"):
        key = (str(row["day"]), str(row["home"]), str(row["away"]))
        # Prefer rows that actually carry an open line when duplicates collide.
        prev = lookup.get(key)
        if prev is None or (
            prev.get("home_open_spread") in (None, "")
            and row.get("home_open_spread") not in (None, "")
        ):
            lookup[key] = row

    def _lookup_row(day: str, home: str, away: str) -> dict | None:
        home = {"wsh": "was", "lar": "la"}.get(home, home)
        away = {"wsh": "was", "lar": "la"}.get(away, away)
        for delta in (0, -1, 1):
            if not day:
                break
            try:
                y, m, d = int(day[:4]), int(day[5:7]), int(day[8:10])
            except ValueError:
                return lookup.get((day, home, away))
            from datetime import date, timedelta

            cand = (date(y, m, d) + timedelta(days=delta)).isoformat()
            row = lookup.get((cand, home, away))
            if row is not None:
                return row
        return None

    for game in games:
        row = _lookup_row(
            str(game.get("date") or "")[:10],
            str(game.get("home") or "").lower(),
            str(game.get("away") or "").lower(),
        )
        if row is None:
            continue
        for field in (
            "home_open_ml",
            "away_open_ml",
            "home_open_spread",
            "away_open_spread",
            "open_total",
            "close_total",
        ):
            val = row.get(field)
            if val is None or (isinstance(val, float) and val != val):
                continue
            game[field] = val
    return games


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
    "home_open_ml",
    "away_open_ml",
    "home_open_spread",
    "away_open_spread",
    "open_total",
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
    games = _attach_opening_lines(games)

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
                "week": game.get("week"),
                "game_type": game.get("game_type"),
                "home_close_ml": game.get("home_close_ml"),
                "away_close_ml": game.get("away_close_ml"),
                "home_close_spread": game.get("home_close_spread"),
                "away_close_spread": game.get("away_close_spread"),
                "home_spread_odds": game.get("home_spread_odds"),
                "away_spread_odds": game.get("away_spread_odds"),
                "close_total": game.get("close_total"),
                "home_open_ml": game.get("home_open_ml"),
                "away_open_ml": game.get("away_open_ml"),
                "home_open_spread": game.get("home_open_spread"),
                "away_open_spread": game.get("away_open_spread"),
                "open_total": game.get("open_total"),
            }
            row.update({col: float(features[col]) for col in FEATURE_COLUMNS})
            # Hard gate: never write NaN feature cells.
            for col in FEATURE_COLUMNS:
                val = row[col]
                if val != val:  # NaN
                    raise ValueError(f"NaN in feature {col} for {game.get('date')} {game.get('home')}@{game.get('away')}")
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
