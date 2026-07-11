"""Build the CBB v2 training table from ESPN completed-game history.

Replays seasons chronologically through CbbFeatureEngine (PIT-safe; no Torvik)
and writes one row per game from --emit-from onward to
data/cbb_history/training_table.csv. Joins closing odds when
data/supplemental/closing-odds/cbb.csv exists.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.cbb_v2.data import (  # noqa: E402
    FIRST_SEASON,
    CACHE_ROOT,
    fetch_season_events,
    load_closing_odds_index,
)
from web.cbb_v2.feature_engine import FEATURE_COLUMNS, CbbFeatureEngine  # noqa: E402
from web.cbb_v2.replay import merge_season_games, replay_season  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "cbb_history"

META_COLUMNS = (
    "season", "date", "event_id", "home", "away",
    "home_abbr", "away_abbr",
    "home_score", "away_score", "home_win", "margin", "total_points",
    "season_type", "conference_game", "neutral_site",
    "home_ml", "away_ml", "home_spread", "spread_home_odds", "spread_away_odds",
    "total_line", "books",
)


def load_season(season: int) -> list[dict[str, Any]]:
    season_dir = CACHE_ROOT / str(season)
    events_path = season_dir / "events.json"
    if events_path.is_file():
        import json

        events = json.loads(events_path.read_text(encoding="utf-8")).get("events", [])
    else:
        events = fetch_season_events(season, use_cache=True)
    return merge_season_games([], events, season=season)


def _odds_for_game(
    odds_index: dict[tuple[str, str, str], dict[str, Any]],
    game: dict[str, Any],
) -> dict[str, Any]:
    date = str(game.get("date") or "")[:10]
    home = str(game.get("home_abbr") or "").lower()
    away = str(game.get("away_abbr") or "").lower()
    row = odds_index.get((date, home, away))
    if not row:
        return {}

    def _f(key: str) -> float | None:
        raw = row.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    return {
        "home_ml": _f("home_close_ml"),
        "away_ml": _f("away_close_ml"),
        "home_spread": _f("home_close_spread"),
        "spread_home_odds": _f("home_spread_odds"),
        "spread_away_odds": _f("away_spread_odds"),
        "total_line": _f("close_total"),
        "books": int(float(row["n_books"])) if row.get("n_books") not in (None, "") else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CBB v2 training table")
    parser.add_argument("--start-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--emit-from", type=int, default=2020)
    parser.add_argument("--refresh", action="store_true", help="Refetch ESPN day caches")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"
    odds_index = load_closing_odds_index()

    engine = CbbFeatureEngine()
    rows_written = 0

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(META_COLUMNS) + list(FEATURE_COLUMNS))

        for season in range(args.start_season, args.end_season + 1):
            if args.refresh:
                events = fetch_season_events(season, use_cache=False)
                games = merge_season_games([], events, season=season)
            else:
                games = load_season(season)
            if not games:
                print(f"season {season}: no games", flush=True)
                continue
            emitted = 0

            def emit(game: dict[str, Any], features: dict[str, float]) -> None:
                nonlocal emitted, rows_written
                if int(game.get("season") or 0) < args.emit_from:
                    return
                market = _odds_for_game(odds_index, game)
                home_score = int(game["home_score"])
                away_score = int(game["away_score"])
                meta = [
                    game.get("season"),
                    game.get("date"),
                    game.get("event_id") or "",
                    game.get("home"),
                    game.get("away"),
                    game.get("home_abbr") or "",
                    game.get("away_abbr") or "",
                    home_score,
                    away_score,
                    int(home_score > away_score),
                    home_score - away_score,
                    home_score + away_score,
                    game.get("season_type"),
                    int(bool(game.get("conference_game"))),
                    int(bool(game.get("neutral_site"))),
                    market.get("home_ml"),
                    market.get("away_ml"),
                    market.get("home_spread"),
                    market.get("spread_home_odds"),
                    market.get("spread_away_odds"),
                    market.get("total_line"),
                    market.get("books", 0),
                ]
                writer.writerow(meta + [features[c] for c in FEATURE_COLUMNS])
                emitted += 1
                rows_written += 1

            replay_season(engine, games, emit=emit)
            print(
                f"season {season}: games={len(games)} emitted={emitted} "
                f"teams={len(engine.teams)}",
                flush=True,
            )

    print(f"wrote {rows_written} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
