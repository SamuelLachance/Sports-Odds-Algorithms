"""Build the NBA v2 training table from cached history + closing-odds CSV.

Replays 1997-2026 chronologically through NbaFeatureEngine and writes one row
per game from --emit-from onward to data/nba_history/training_table.csv.
Market odds are joined from data/supplemental/closing-odds/nba.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_v2.data import CACHE_ROOT, load_closing_odds_index  # noqa: E402
from web.nba_v2.feature_engine import FEATURE_COLUMNS, NbaFeatureEngine  # noqa: E402
from web.nba_v2.replay import merge_season_games, replay_season  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "nba_history"

META_COLUMNS = (
    "season", "date", "event_id", "home", "away",
    "home_score", "away_score", "home_win", "margin", "total_points",
    "season_type", "has_box",
    "home_ml", "away_ml", "home_spread", "spread_home_odds", "spread_away_odds",
    "total_line", "books", "home_ml_open", "away_ml_open", "home_spread_open",
    "best_home_ml", "best_away_ml", "best_spread_home_odds", "best_spread_away_odds",
)


def load_season(season: int) -> list[dict[str, Any]]:
    season_dir = CACHE_ROOT / str(season)
    events_path = season_dir / "events.json"
    boxes_path = season_dir / "boxes.json"
    if not events_path.is_file():
        return []
    events = json.loads(events_path.read_text(encoding="utf-8")).get("events", [])
    boxes = {}
    if boxes_path.is_file():
        boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    return merge_season_games([], events, boxes, season=season)


def _odds_for_game(
    index: dict[tuple[str, str, str], dict[str, Any]],
    game: dict[str, Any],
) -> dict[str, Any]:
    date = str(game.get("date") or "")[:10]
    home = str(game.get("home") or "")
    away = str(game.get("away") or "")
    row = index.get((date, home, away))
    if row is None:
        # UTC/local date skew — only when exactly one ±1 neighbor exists
        # (refuse consecutive same-venue series; mirrors closing_odds_db).
        from datetime import date as date_cls, timedelta

        try:
            base = date_cls.fromisoformat(date)
        except ValueError:
            return {}
        neighbors: list[dict[str, Any]] = []
        for offset in (-1, 1):
            alt = (base + timedelta(days=offset)).isoformat()
            candidate = index.get((alt, home, away))
            if candidate is not None:
                neighbors.append(candidate)
        if len(neighbors) == 1:
            row = neighbors[0]
    if not row:
        return {}
    return {
        "home_ml": row.get("home_ml"),
        "away_ml": row.get("away_ml"),
        "home_spread": row.get("home_spread"),
        "spread_home_odds": row.get("home_spread_odds"),
        "spread_away_odds": row.get("away_spread_odds"),
        "total_line": row.get("total"),
        "close_total": row.get("total"),
        "open_total": row.get("open_total"),
        "books": row.get("n_books") or 0,
        "n_books": row.get("n_books") or 0,
        "home_ml_open": row.get("home_ml_open"),
        "away_ml_open": row.get("away_ml_open"),
        "home_open_ml": row.get("home_ml_open"),
        "away_open_ml": row.get("away_ml_open"),
        "home_spread_open": row.get("home_spread_open"),
        "home_open_spread": row.get("home_spread_open"),
        "home_close_ml": row.get("home_ml"),
        "away_close_ml": row.get("away_ml"),
        "home_close_spread": row.get("home_spread"),
        # CSV is already consensus median; best == consensus for this source
        "best_home_ml": row.get("home_ml"),
        "best_away_ml": row.get("away_ml"),
        "best_spread_home_odds": row.get("home_spread_odds"),
        "best_spread_away_odds": row.get("away_spread_odds"),
    }


def _attach_market_to_games(
    games: list[dict[str, Any]],
    odds_index: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """Copy closing-odds fields onto game dicts before feature emit (steam needs opens)."""
    for game in games:
        market = _odds_for_game(odds_index, game)
        if not market:
            continue
        for key, value in market.items():
            if value is not None:
                game[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NBA v2 training table")
    parser.add_argument("--start-season", type=int, default=1997)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--emit-from", type=int, default=2002)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"
    odds_index = load_closing_odds_index()
    print(f"closing-odds index: {len(odds_index)} games", flush=True)

    engine = NbaFeatureEngine()
    rows_written = 0

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(META_COLUMNS) + list(FEATURE_COLUMNS))

        for season in range(args.start_season, args.end_season + 1):
            games = load_season(season)
            if not games:
                print(f"season {season}: no games cached", flush=True)
                continue
            _attach_market_to_games(games, odds_index)
            emitted = 0

            def emit(game: dict[str, Any], features: dict[str, float]) -> None:
                nonlocal emitted, rows_written
                if int(game.get("season") or 0) < args.emit_from:
                    return
                market = _odds_for_game(odds_index, game)
                home_score = int(game["home_score"])
                away_score = int(game["away_score"])
                meta = [
                    game.get("season"), game.get("date"), game.get("event_id") or "",
                    game.get("home"), game.get("away"),
                    home_score, away_score, int(home_score > away_score),
                    home_score - away_score, home_score + away_score,
                    game.get("season_type"),
                    int(bool(game.get("home_box"))),
                    market.get("home_ml"), market.get("away_ml"),
                    market.get("home_spread"),
                    market.get("spread_home_odds"), market.get("spread_away_odds"),
                    market.get("total_line"), market.get("books", 0),
                    market.get("home_ml_open"), market.get("away_ml_open"),
                    market.get("home_spread_open"),
                    market.get("best_home_ml"), market.get("best_away_ml"),
                    market.get("best_spread_home_odds"), market.get("best_spread_away_odds"),
                ]
                writer.writerow(
                    [v if v is not None else "" for v in meta]
                    + [round(float(features[c]), 5) for c in FEATURE_COLUMNS]
                )
                emitted += 1
                rows_written += 1

            replay_season(engine, games, emit=emit)
            boxed = sum(1 for g in games if g.get("home_box"))
            with_odds = sum(
                1 for g in games
                if int(g.get("season") or 0) >= args.emit_from
                and _odds_for_game(odds_index, g).get("home_spread") is not None
            )
            print(
                f"season {season}: {len(games)} games ({boxed} with box, "
                f"~{with_odds} with odds), {emitted} rows emitted",
                flush=True,
            )

    print(f"training table: {rows_written} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
