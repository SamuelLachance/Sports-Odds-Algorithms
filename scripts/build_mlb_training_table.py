"""Build the MLB v2 training table from cached statsapi history + closing odds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import devig_two_way_probs  # noqa: E402
from web.closing_odds_db import closing_odds_lookup  # noqa: E402
from web.mlb_stats_api import ESPN_TO_MLB_TEAM_ID  # noqa: E402
from web.mlb_v2.feature_engine import FEATURE_COLUMNS, MlbFeatureEngine  # noqa: E402
from web.mlb_v2.replay import replay_season  # noqa: E402

CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "mlb-history"
OUT_DIR = PROJECT_ROOT / "data" / "mlb_history"

# teamId -> ESPN abbr used by the closing-odds index ('oak' canonical for 133).
TEAM_ID_TO_ABBR: dict[int, str] = {
    team_id: abbr for abbr, team_id in ESPN_TO_MLB_TEAM_ID.items() if abbr != "ath"
}


def binary_home_win(home_score: int, away_score: int) -> int | None:
    """1/0 for decisive games; None for ties (exclude from classifier training)."""
    if home_score == away_score:
        return None
    return 1 if home_score > away_score else 0


def load_season(season: int) -> dict[str, Any] | None:
    season_dir = CACHE_ROOT / str(season)
    paths = {
        "games": season_dir / "games.json",
        "pitchers": season_dir / "pitchers.json",
        "team_hitting": season_dir / "team_hitting.json",
        "team_pitching": season_dir / "team_pitching.json",
    }
    if not all(p.is_file() for p in paths.values()):
        return None
    return {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}


def build_rows(start_season: int, end_season: int) -> pd.DataFrame:
    engine = MlbFeatureEngine()
    rows: list[dict[str, Any]] = []

    for season in range(start_season, end_season + 1):
        data = load_season(season)
        if data is None:
            print(f"season {season}: cache missing, skipped", flush=True)
            continue

        def on_row(game: dict[str, Any], feats: dict[str, float]) -> None:
            home_win = binary_home_win(int(game["home_score"]), int(game["away_score"]))
            if home_win is None:
                # Feature engine also skips ties; do not label them as away wins.
                return
            home_id = int(game["home_id"])
            away_id = int(game["away_id"])
            record: dict[str, Any] = {
                "season": season,
                "date": game["date"],
                "game_pk": game.get("gamePk"),
                "home_id": home_id,
                "away_id": away_id,
                "home_key": TEAM_ID_TO_ABBR.get(home_id, str(home_id)),
                "away_key": TEAM_ID_TO_ABBR.get(away_id, str(away_id)),
                "home_pp_id": game.get("home_pp_id"),
                "away_pp_id": game.get("away_pp_id"),
                "home_score": game.get("home_score"),
                "away_score": game.get("away_score"),
                "home_win": home_win,
            }
            record.update(feats)
            rows.append(record)

        replay_season(
            engine,
            season,
            data["games"],
            data["pitchers"],
            data["team_hitting"],
            data["team_pitching"],
            on_row=on_row,
        )
        print(f"season {season}: {sum(1 for r in rows if r['season'] == season)} rows", flush=True)

    return pd.DataFrame(rows)


def merge_odds(frame: pd.DataFrame) -> pd.DataFrame:
    home_mls: list[float | None] = []
    away_mls: list[float | None] = []
    home_opens: list[float | None] = []
    away_opens: list[float | None] = []
    market_probs: list[float | None] = []
    for row in frame.itertuples(index=False):
        odds = closing_odds_lookup("mlb", row.date, row.home_key, row.away_key) or {}
        home_ml = odds.get("home_close_ml")
        away_ml = odds.get("away_close_ml")
        market_home: float | None = None
        if home_ml is not None and away_ml is not None:
            _, market_home = devig_two_way_probs(int(away_ml), int(home_ml))
            if market_home is not None:
                market_home = round(market_home / 100.0, 6)
        home_mls.append(home_ml)
        away_mls.append(away_ml)
        home_opens.append(odds.get("home_open_ml"))
        away_opens.append(odds.get("away_open_ml"))
        market_probs.append(market_home)
    frame["home_close_ml"] = home_mls
    frame["away_close_ml"] = away_mls
    frame["home_open_ml"] = home_opens
    frame["away_open_ml"] = away_opens
    frame["market_home_prob"] = market_probs
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MLB v2 training table")
    parser.add_argument("--start-season", type=int, default=2011)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    args = parser.parse_args()

    frame = build_rows(args.start_season, args.end_season)
    if frame.empty:
        print("no rows built; run scripts/fetch_mlb_history.py first")
        return 1
    frame = merge_odds(frame)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"
    frame.to_csv(out_path, index=False)

    total = len(frame)
    with_odds = int(frame["market_home_prob"].notna().sum())
    print(f"\nwrote {out_path} ({total} rows, {with_odds} with closing odds, "
          f"{with_odds / total * 100:.1f}% match)")
    per_season = frame.groupby("season").agg(
        games=("date", "count"),
        odds=("market_home_prob", lambda s: int(s.notna().sum())),
        home_win=("home_win", "mean"),
    )
    print(per_season.to_string())
    missing = frame.loc[frame["market_home_prob"].isna()]
    if not missing.empty:
        sample = missing.groupby(["season"]).size().tail(6)
        print("\nmissing odds by season (tail):")
        print(sample.to_string())
    assert list(FEATURE_COLUMNS) == [c for c in FEATURE_COLUMNS], "feature columns intact"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
