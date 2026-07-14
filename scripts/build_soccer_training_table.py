"""Build the soccer v2 training table from cached football-data history.

Replays every country pool (top flight + 2nd division) through the
walk-forward feature engine and writes one row per TOP-FLIGHT match with
features, result and open/close odds to data/soccer_history/training_table.csv.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_v2.data import (  # noqa: E402
    DIVISION_COUNTRY,
    HISTORY_DIR,
    all_division_codes,
    current_season,
    load_division_season,
    seasons_for_division,
)
from web.soccer_v2.feature_engine import FEATURE_COLUMNS, SoccerFeatureEngine  # noqa: E402
from web.soccer_v2.replay import merge_country_rows, replay_country  # noqa: E402
from web.soccer_v2.xg import attach_xg_to_rows  # noqa: E402

META_COLUMNS = (
    "league",
    "division",
    "season",
    "date",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "result",
    "open_home",
    "open_draw",
    "open_away",
    "close_home",
    "close_draw",
    "close_away",
    "b365_home",
    "b365_draw",
    "b365_away",
    "b365c_home",
    "b365c_draw",
    "b365c_away",
    "max_home",
    "max_draw",
    "max_away",
)


def main() -> None:
    through = current_season()
    by_country: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for division in all_division_codes():
        rows: list[dict[str, Any]] = []
        for season in seasons_for_division(division, through=through):
            rows.extend(load_division_season(division, season))
        by_country[DIVISION_COUNTRY[division]][division] = rows
        print(f"{division}: {len(rows)} matches", flush=True)

    out_path = HISTORY_DIR / "training_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(META_COLUMNS) + list(FEATURE_COLUMNS))

        for country, divisions in sorted(by_country.items()):
            engine = SoccerFeatureEngine(country)
            merged = merge_country_rows(divisions)
            attach_xg_to_rows(merged)

            def emit(row: dict[str, Any], features: dict[str, float]) -> None:
                nonlocal written
                from web.soccer_v2.data import DIVISION_TO_LEAGUE

                result = (
                    "H"
                    if row["home_goals"] > row["away_goals"]
                    else ("A" if row["away_goals"] > row["home_goals"] else "D")
                )
                writer.writerow(
                    [
                        DIVISION_TO_LEAGUE.get(row["division"], ""),
                        row["division"],
                        row["season"],
                        row["date"],
                        row["home"],
                        row["away"],
                        row["home_goals"],
                        row["away_goals"],
                        result,
                    ]
                    + [
                        row.get(k) if row.get(k) is not None else ""
                        for k in (
                            "open_home",
                            "open_draw",
                            "open_away",
                            "close_home",
                            "close_draw",
                            "close_away",
                            "b365_home",
                            "b365_draw",
                            "b365_away",
                            "b365c_home",
                            "b365c_draw",
                            "b365c_away",
                            "max_home",
                            "max_draw",
                            "max_away",
                        )
                    ]
                    + [
                        round(float(features.get(col, 0.0)), 5)
                        for col in FEATURE_COLUMNS
                    ]
                )
                written += 1

            replay_country(engine, merged, on_features=emit)
            print(f"pool {country}: replayed {len(merged)} matches", flush=True)

    print(f"training table: {written} top-flight rows -> {out_path}")


if __name__ == "__main__":
    main()
