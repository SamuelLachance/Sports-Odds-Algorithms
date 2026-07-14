"""Build the WNBA v2 training table from cached history.

Replays 1997-2026 chronologically through WnbaFeatureEngine and writes one
row per game from --emit-from onward (default 2002: five Elo warm-up years,
box-score features begin 2003) to data/wnba_history/training_table.csv,
including consensus market odds (median across pre-game books) when cached.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.wnba_v2.data import CACHE_ROOT  # noqa: E402
from web.wnba_v2.feature_engine import FEATURE_COLUMNS, WnbaFeatureEngine  # noqa: E402
from web.wnba_v2.replay import merge_season_games, replay_season  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "wnba_history"

META_COLUMNS = (
    "season", "date", "event_id", "home", "away",
    "home_score", "away_score", "home_win", "margin", "total_points",
    "season_type", "has_box",
    "home_ml", "away_ml", "home_spread", "spread_home_odds", "spread_away_odds",
    "total_line", "books", "home_ml_open", "away_ml_open", "home_spread_open",
    "best_home_ml", "best_away_ml", "best_spread_home_odds", "best_spread_away_odds",
)


def _median(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    return float(statistics.median(clean))


def _clean_american(values: list[Any]) -> list[float]:
    """Finite American prices only; ESPN EVEN 0 / text → +100."""
    out: list[float] = []
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, bool):
            continue
        if isinstance(raw, str) and raw.strip().upper() in {"EVEN", "PK"}:
            out.append(100.0)
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        if val == 0:
            out.append(100.0)
        elif abs(val) >= 100:
            out.append(val)
    return out


def _valid_american_median(value: float | None) -> float | None:
    """Drop even-book medians that land in the invalid |x| < 100 dead zone."""
    if value is None or not math.isfinite(value):
        return None
    if value == 0:
        return 100.0
    if abs(value) < 100:
        return None
    return value


def consensus_odds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median market across pre-game books for one event."""
    if not rows:
        return {}
    home_mls = _clean_american([r.get("home_ml") for r in rows])
    away_mls = _clean_american([r.get("away_ml") for r in rows])
    spreads = [
        float(v)
        for v in (r.get("home_spread") for r in rows)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    home_so = _clean_american([r.get("home_spread_odds") for r in rows])
    away_so = _clean_american([r.get("away_spread_odds") for r in rows])
    totals = [
        float(v)
        for v in (r.get("total") for r in rows)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    ml_open_h = _clean_american([r.get("home_ml_open") for r in rows])
    ml_open_a = _clean_american([r.get("away_ml_open") for r in rows])
    sp_open = [
        float(v)
        for v in (r.get("home_spread_open") for r in rows)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]

    def _best(values: list[float]) -> float | None:
        """Best price for the bettor: highest decimal payout."""
        priced = [v for v in values if math.isfinite(v) and abs(v) >= 100]
        if not priced:
            return None
        return max(priced, key=lambda ml: ml if ml > 0 else 10000.0 / abs(ml))

    return {
        "home_ml": _valid_american_median(_median(home_mls)),
        "away_ml": _valid_american_median(_median(away_mls)),
        "home_spread": _median(spreads),
        "spread_home_odds": _valid_american_median(_median(home_so)),
        "spread_away_odds": _valid_american_median(_median(away_so)),
        "total_line": _median(totals),
        "books": len(rows),
        "home_ml_open": _valid_american_median(_median(ml_open_h)),
        "away_ml_open": _valid_american_median(_median(ml_open_a)),
        "home_spread_open": _median(sp_open),
        "best_home_ml": _best(home_mls),
        "best_away_ml": _best(away_mls),
        "best_spread_home_odds": _best(home_so),
        "best_spread_away_odds": _best(away_so),
    }


def load_season(season: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    season_dir = CACHE_ROOT / str(season)

    def read(name: str) -> Any:
        path = season_dir / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    results = (read("results.json") or {}).get("results", [])
    events = (read("events.json") or {}).get("events", [])
    boxes = read("boxes.json") or {}
    odds = read("odds.json") or {}
    games = merge_season_games(results, events, boxes, season=season)
    return games, odds


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WNBA v2 training table")
    parser.add_argument("--start-season", type=int, default=1997)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--emit-from", type=int, default=2002)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"

    engine = WnbaFeatureEngine()
    rows_written = 0

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(META_COLUMNS) + list(FEATURE_COLUMNS))

        for season in range(args.start_season, args.end_season + 1):
            games, odds = load_season(season)
            if not games:
                print(f"season {season}: no games cached", flush=True)
                continue
            # Bake consensus open/close onto games before FE so steam columns populate.
            for game in games:
                market = consensus_odds(odds.get(str(game.get("event_id") or "")) or [])
                if not market:
                    continue
                for key, value in market.items():
                    if value is not None:
                        game[key] = value
                if market.get("home_ml") is not None:
                    game["home_close_ml"] = market["home_ml"]
                    game["away_close_ml"] = market["away_ml"]
                if market.get("home_ml_open") is not None:
                    game["home_open_ml"] = market["home_ml_open"]
                    game["away_open_ml"] = market["away_ml_open"]
                if market.get("home_spread") is not None:
                    game["home_close_spread"] = market["home_spread"]
                if market.get("home_spread_open") is not None:
                    game["home_open_spread"] = market["home_spread_open"]
                if market.get("total_line") is not None:
                    game["close_total"] = market["total_line"]
                    game["total_line"] = market["total_line"]
                if market.get("books") is not None:
                    game["n_books"] = market["books"]
            emitted = 0

            def emit(game: dict[str, Any], features: dict[str, float]) -> None:
                nonlocal emitted, rows_written
                if int(game.get("season") or 0) < args.emit_from:
                    return
                market = consensus_odds(odds.get(str(game.get("event_id") or "")) or [])
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
            print(
                f"season {season}: {len(games)} games ({boxed} with box), "
                f"{emitted} rows emitted",
                flush=True,
            )

    print(f"training table: {rows_written} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
