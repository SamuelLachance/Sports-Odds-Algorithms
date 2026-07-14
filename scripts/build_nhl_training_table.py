"""Build the NHL v2 training table from cached history + closing odds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import devig_two_way_probs  # noqa: E402
from web.closing_odds_db import closing_odds_lookup  # noqa: E402
from web.nhl_v2.data import CACHE_ROOT, build_game_index, espn_key, parse_moneypuck_teams  # noqa: E402
from web.nhl_v2.feature_engine import FEATURE_COLUMNS, NhlFeatureEngine  # noqa: E402
from web.nhl_v2.replay import build_goalie_index, load_goalie_xg_csv, replay_season  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "nhl_history"


def load_season(season: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    season_dir = CACHE_ROOT / str(season)
    team_path = season_dir / "team_games.json"
    goalie_path = season_dir / "goalies.json"
    if not team_path.is_file() or not goalie_path.is_file():
        return None
    return (
        json.loads(team_path.read_text(encoding="utf-8")),
        json.loads(goalie_path.read_text(encoding="utf-8")),
    )


def build_rows(start_season: int, end_season: int) -> pd.DataFrame:
    print("parsing MoneyPuck team file...", flush=True)
    mp_games = parse_moneypuck_teams()
    print(f"  {len(mp_games)} MoneyPuck games", flush=True)
    goalie_xg = load_goalie_xg_csv(CACHE_ROOT / "goalie_xg_games.csv")
    print(f"  {len(goalie_xg)} goalie xG game rows", flush=True)

    engine = NhlFeatureEngine()
    rows: list[dict[str, Any]] = []

    for season in range(start_season, end_season + 1):
        data = load_season(season)
        if data is None:
            print(f"season {season}: cache missing, skipped", flush=True)
            continue
        team_rows, goalie_rows = data
        games = list(build_game_index(team_rows).values())
        goalie_index = build_goalie_index(goalie_rows)

        def on_row(game: dict[str, Any], feats: dict[str, float]) -> None:
            record: dict[str, Any] = {
                "season": season,
                "date": game["date"],
                "game_id": game["gameId"],
                "game_type": game.get("game_type"),
                "home": game["home"],
                "away": game["away"],
                "home_key": espn_key(game["home"]),
                "away_key": espn_key(game["away"]),
                "home_score": game.get("home_goals"),
                "away_score": game.get("away_goals"),
                "home_win": int(game.get("home_win") or 0),
                "home_goalie_id": game.get("home_goalie_id"),
                "away_goalie_id": game.get("away_goalie_id"),
                "has_mp": 1 if game.get("mp") else 0,
            }
            record.update(feats)
            rows.append(record)

        replay_season(
            engine, season, games, mp_games, goalie_index,
            goalie_xg=goalie_xg, on_row=on_row,
        )
        n_season = sum(1 for r in rows if r["season"] == season)
        n_mp = sum(1 for r in rows if r["season"] == season and r["has_mp"])
        print(f"season {season}: {n_season} rows ({n_mp} with MoneyPuck)", flush=True)

    return pd.DataFrame(rows)


def _american_to_prob(ml: float) -> float:
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)


def _devig_pair(home_ml: float, away_ml: float) -> float:
    ph = _american_to_prob(home_ml)
    pa = _american_to_prob(away_ml)
    tot = ph + pa
    return ph / tot if tot > 0 else 0.5


def _apply_market_features(feats: dict[str, float], odds: dict[str, Any]) -> None:
    """Overwrite market FEATURE_COLUMNS in-place. Never leave NaNs.

    Missing markets keep defaults (has_*=0, mkt_home_prob=0.5, steam=0) —
    real open/close values only when present in nhl.csv.
    """
    h_ml = odds.get("home_close_ml")
    a_ml = odds.get("away_close_ml")
    if h_ml is None or a_ml is None:
        return
    mkt_p = _devig_pair(float(h_ml), float(a_ml))
    feats["has_market"] = 1.0
    feats["mkt_home_prob"] = float(mkt_p)

    spread = odds.get("home_close_spread")
    if spread is not None:
        feats["has_spread"] = 1.0
        feats["mkt_home_spread"] = float(spread)
        open_spread = odds.get("home_open_spread")
        if open_spread is not None:
            feats["has_open_line"] = 1.0
            feats["spread_move"] = float(spread) - float(open_spread)

    books = odds.get("n_books")
    if books is not None:
        try:
            feats["n_books"] = float(books)
        except (TypeError, ValueError):
            pass

    total = odds.get("close_total")
    if total is not None:
        feats["has_total"] = 1.0
        feats["total_line"] = float(total)
        feats["model_total_vs_line"] = float(feats.get("exp_total_env", 0.0)) - float(total)

    open_h = odds.get("home_open_ml")
    open_a = odds.get("away_open_ml")
    if open_h is not None and open_a is not None:
        open_p = _devig_pair(float(open_h), float(open_a))
        feats["ml_steam_pp"] = (mkt_p - open_p) * 100.0
        feats["has_steam"] = 1.0
        feats["has_open_line"] = 1.0


def merge_odds(frame: pd.DataFrame) -> pd.DataFrame:
    home_close: list[float | None] = []
    away_close: list[float | None] = []
    home_open: list[float | None] = []
    away_open: list[float | None] = []
    home_spread: list[float | None] = []
    close_total: list[float | None] = []
    market: list[float | None] = []

    feat_cols = list(FEATURE_COLUMNS)
    # Row-wise overwrite of market feature block (vectorized after).
    market_updates: list[dict[str, float]] = []

    for row in frame.itertuples(index=False):
        odds = closing_odds_lookup("nhl", row.date, row.home_key, row.away_key) or {}
        h_ml = odds.get("home_close_ml")
        a_ml = odds.get("away_close_ml")
        prob: float | None = None
        if h_ml is not None and a_ml is not None:
            _, prob = devig_two_way_probs(int(a_ml), int(h_ml))
            if prob is not None:
                prob = round(prob / 100.0, 6)
        home_close.append(h_ml)
        away_close.append(a_ml)
        home_open.append(odds.get("home_open_ml"))
        away_open.append(odds.get("away_open_ml"))
        home_spread.append(odds.get("home_close_spread"))
        close_total.append(odds.get("close_total"))
        market.append(prob)

        feats = {c: float(getattr(row, c)) for c in feat_cols}
        _apply_market_features(feats, odds)
        market_updates.append(feats)

    frame["home_close_ml"] = home_close
    frame["away_close_ml"] = away_close
    frame["home_open_ml"] = home_open
    frame["away_open_ml"] = away_open
    frame["home_close_spread"] = home_spread
    frame["close_total"] = close_total
    frame["market_home_prob"] = market

    mkt_frame = pd.DataFrame(market_updates)
    for col in feat_cols:
        frame[col] = mkt_frame[col].to_numpy()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NHL v2 training table")
    parser.add_argument("--start-season", type=int, default=2008)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument(
        "--require-odds",
        action="store_true",
        help="Drop rows without closing ML (no NA market fills)",
    )
    args = parser.parse_args()

    frame = build_rows(args.start_season, args.end_season)
    if frame.empty:
        print("no rows built; run scripts/fetch_nhl_history.py first")
        return 1

    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        print(f"ERROR: feature engine missing columns: {missing[:20]}")
        return 1

    frame = merge_odds(frame)
    if args.require_odds:
        before = len(frame)
        frame = frame[frame["market_home_prob"].notna()].copy()
        print(f"require-odds: kept {len(frame)}/{before} rows", flush=True)

    # Hard gate: no NaNs in model features.
    na_counts = frame[list(FEATURE_COLUMNS)].isna().sum()
    bad = na_counts[na_counts > 0]
    if len(bad):
        print("ERROR: NaNs in FEATURE_COLUMNS:")
        print(bad.head(30).to_string())
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"
    frame.to_csv(out_path, index=False)

    total = len(frame)
    with_odds = int(frame["market_home_prob"].notna().sum())
    with_steam = int((frame["has_steam"] > 0).sum()) if "has_steam" in frame.columns else 0
    print(
        f"\nwrote {out_path} ({total} rows, {with_odds} with closing odds, "
        f"{with_odds / max(total, 1) * 100:.1f}% match, {with_steam} with ML steam)"
    )
    per_season = frame.groupby("season").agg(
        games=("date", "count"),
        mp=("has_mp", "sum"),
        odds=("market_home_prob", lambda s: int(s.notna().sum())),
        opens=("home_open_ml", lambda s: int(s.notna().sum())),
        steam=("has_steam", "sum"),
        home_win=("home_win", "mean"),
        goalies=("home_goalie_id", lambda s: int(s.notna().sum())),
    )
    print(per_season.to_string())
    print(f"FEATURE_COLUMNS={len(FEATURE_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
