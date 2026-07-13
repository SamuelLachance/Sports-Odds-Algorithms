"""Rebuild backtest-ready oos_predictions.csv from hybrid OOS + training odds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def _load_hybrid_oos(league: str) -> pd.DataFrame | None:
    hybrid_dir = PROJECT_ROOT / "data" / "models" / f"{league}_hybrid"
    v2 = PROJECT_ROOT / "data" / "models" / f"{league}_v2"
    best_path = hybrid_dir / "best_config.json"
    candidates: list[Path] = []
    if best_path.is_file():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        cfg_name = (best.get("config") or {}).get("name") or best.get("name")
        if cfg_name:
            candidates.append(hybrid_dir / f"oos_{_safe_name(cfg_name)}.csv")
        # overnight WNBA winner may live under mut_* name in best
    candidates.append(v2 / "oos_predictions_hybrid.csv")
    candidates.extend(sorted(hybrid_dir.glob("oos_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True))
    for path in candidates:
        if path.is_file():
            frame = pd.read_csv(path)
            print(f"{league}: loaded hybrid OOS {path.name} n={len(frame)}", flush=True)
            return frame
    return None


def _merge_odds(league: str, hybrid: pd.DataFrame) -> pd.DataFrame:
    table_path = PROJECT_ROOT / "data" / f"{league}_history" / "training_table.csv"
    if not table_path.is_file():
        return hybrid
    table = pd.read_csv(table_path)

    # Normalize team keys
    if league == "mlb":
        if "home" not in hybrid.columns and "home_key" in hybrid.columns:
            hybrid = hybrid.copy()
            hybrid["home"] = hybrid["home_key"]
            hybrid["away"] = hybrid["away_key"]
        if "home" not in table.columns and "home_key" in table.columns:
            table = table.copy()
            table["home"] = table["home_key"]
            table["away"] = table["away_key"]

    if league == "cbb" and "home_abbr" in table.columns:
        # CBB backtest joins on abbr; keep both
        pass

    merge_keys = [c for c in ("season", "date", "home", "away") if c in hybrid.columns and c in table.columns]
    if "game_pk" in hybrid.columns and "game_pk" in table.columns:
        merge_keys = ["game_pk"]
    elif "event_id" in hybrid.columns and "event_id" in table.columns:
        merge_keys = ["event_id"]

    odds_cols = [
        c
        for c in table.columns
        if any(
            tok in c.lower()
            for tok in (
                "ml",
                "spread",
                "odds",
                "open",
                "close",
                "book",
                "max_",
                "b365",
                "score",
                "margin",
                "result",
                "label",
                "n_books",
                "total",
            )
        )
        or c in ("home_abbr", "away_abbr", "home_key", "away_key")
    ]
    # Drop odds already present without _tbl suffix conflicts carefully
    keep = list(dict.fromkeys(merge_keys + odds_cols))
    right = table[keep].drop_duplicates(subset=merge_keys)
    # Prefer hybrid values when both exist
    overlap = [c for c in odds_cols if c in hybrid.columns and c not in merge_keys]
    hybrid2 = hybrid.drop(columns=overlap, errors="ignore")
    merged = hybrid2.merge(right, on=merge_keys, how="left", suffixes=("", "_tbl"))
    print(f"{league}: merged odds via {merge_keys} -> cols={len(merged.columns)}", flush=True)
    return merged


def _add_aliases(league: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # Hybrid already market-blended → expose as both pure and mkt heads for grids
    if "model_prob" in frame.columns:
        frame["model_prob_mkt"] = frame["model_prob"]
    if "model_margin" in frame.columns:
        frame["model_margin_mkt"] = frame["model_margin"]
        frame["pred_margin"] = frame["model_margin"]

    # MLB/NHL aliases
    if "home_close_ml" not in frame.columns and "home_ml" in frame.columns:
        frame["home_close_ml"] = frame["home_ml"]
        frame["away_close_ml"] = frame["away_ml"]
    if "home_open_ml" not in frame.columns:
        if "home_ml_open" in frame.columns:
            frame["home_open_ml"] = frame["home_ml_open"]
            frame["away_open_ml"] = frame["away_ml_open"]
        elif "home_close_ml" in frame.columns:
            frame["home_open_ml"] = frame["home_close_ml"]
            frame["away_open_ml"] = frame["away_close_ml"]

    # NFL/CFB spread
    if "home_spread" not in frame.columns and "home_close_spread" in frame.columns:
        frame["home_spread"] = frame["home_close_spread"]

    # WNBA/NBA books
    if "books" not in frame.columns:
        if "n_books" in frame.columns:
            frame["books"] = frame["n_books"]
        elif "home_ml" in frame.columns:
            frame["books"] = np.where(frame["home_ml"].notna(), 1.0, 0.0)
        elif "home_close_ml" in frame.columns:
            frame["books"] = np.where(frame["home_close_ml"].notna(), 1.0, 0.0)

    # CBB abbrs
    if league == "cbb":
        if "home_abbr" not in frame.columns and "home" in frame.columns:
            frame["home_abbr"] = frame["home"]
            frame["away_abbr"] = frame["away"]

    # Soccer 1X2 calibrated aliases expected by backtest
    if league == "soccer":
        if "model_prob_h" in frame.columns:
            frame["cal_home"] = frame["model_prob_h"]
            frame["cal_draw"] = frame["model_prob_d"]
            frame["cal_away"] = frame["model_prob_a"]
            frame["calm_home"] = frame["model_prob_h"]
            frame["calm_draw"] = frame["model_prob_d"]
            frame["calm_away"] = frame["model_prob_a"]
        if "label" not in frame.columns and "result" in frame.columns:
            frame["label"] = frame["result"].map({"H": 0, "D": 1, "A": 2})

    # Deduplicate match keys
    keys = [c for c in ("season", "date", "home", "away", "game_pk", "event_id") if c in frame.columns]
    if {"season", "date", "home", "away"}.issubset(frame.columns):
        before = len(frame)
        frame = frame.drop_duplicates(subset=["season", "date", "home", "away"], keep="first")
        if len(frame) < before:
            print(f"{league}: dropped {before - len(frame)} duplicate match rows", flush=True)
    return frame


def rebuild(league: str) -> Path | None:
    hybrid = _load_hybrid_oos(league)
    if hybrid is None:
        print(f"{league}: SKIP no hybrid OOS", flush=True)
        return None
    frame = _merge_odds(league, hybrid)
    frame = _add_aliases(league, frame)
    out = PROJECT_ROOT / "data" / "models" / f"{league}_v2" / "oos_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    # also refresh hybrid copy
    frame.to_csv(out.parent / "oos_predictions_hybrid.csv", index=False)
    print(f"{league}: wrote {out} n={len(frame)}", flush=True)
    return out


def main() -> int:
    leagues = sys.argv[1:] or ["wnba", "nba", "mlb", "nhl", "nfl", "cfb", "cbb", "soccer"]
    for league in leagues:
        try:
            rebuild(league)
        except Exception as exc:  # noqa: BLE001
            print(f"{league}: ERROR {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
