"""Merge hybrid OOS probs with training-table odds into backtest-ready CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hybrid_v2.adapters import ALL_LEAGUES, get_adapter  # noqa: E402


def merge_league(league: str) -> Path | None:
    adapter = get_adapter(league)
    # Prefer shipped copy, else best oos from hybrid dir
    candidates = [
        adapter.v2_dir / "oos_predictions_hybrid.csv",
        *sorted(adapter.hybrid_dir.glob("oos_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True),
    ]
    hybrid_path = next((p for p in candidates if p.is_file()), None)
    if hybrid_path is None:
        print(f"{league}: no hybrid OOS", flush=True)
        return None
    hybrid = pd.read_csv(hybrid_path)
    if "model_prob" not in hybrid.columns and "model_prob_h" in hybrid.columns:
        # soccer multiclass — synthesize binary home prob for ML-style backtests
        hybrid["model_prob"] = hybrid["model_prob_h"]
    table = pd.read_csv(adapter.table_path) if adapter.table_path.is_file() else None

    keys = [c for c in ("season", "date", "home", "away", "event_id", "game_pk") if c in hybrid.columns]
    if table is not None:
        # Align team key columns for MLB
        if adapter.home_col != "home" and adapter.home_col in table.columns:
            if "home" not in hybrid.columns and adapter.home_col in hybrid.columns:
                pass
            elif "home" not in table.columns:
                table = table.rename(columns={adapter.home_col: "home", adapter.away_col: "away"})
        merge_keys = [c for c in ("season", "date", "home", "away") if c in hybrid.columns and c in table.columns]
        if not merge_keys and {"season", "date"}.issubset(hybrid.columns) and {"season", "date"}.issubset(table.columns):
            merge_keys = ["season", "date"]
            # MLB: join on home_id/away_id if present
            for a, b in (("home_key", "home"), ("away_key", "away")):
                if a in table.columns and b not in table.columns:
                    table[b] = table[a]
            if "home" in hybrid.columns:
                merge_keys = [c for c in ("season", "date", "home", "away") if c in hybrid.columns and c in table.columns]
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
                )
            )
        ]
        keep = list(dict.fromkeys(merge_keys + odds_cols))
        if merge_keys:
            merged = hybrid.merge(table[keep].drop_duplicates(merge_keys), on=merge_keys, how="left", suffixes=("", "_tbl"))
        else:
            merged = hybrid.copy()
    else:
        merged = hybrid.copy()

    # Backtest aliases
    if "model_prob_mkt" not in merged.columns and "model_prob" in merged.columns:
        merged["model_prob_mkt"] = merged["model_prob"]
    if "home_close_ml" not in merged.columns and "home_ml" in merged.columns:
        merged["home_close_ml"] = merged["home_ml"]
        merged["away_close_ml"] = merged["away_ml"]
    if "model_margin" in merged.columns and "pred_margin" not in merged.columns:
        merged["pred_margin"] = merged["model_margin"]
        merged["model_margin_mkt"] = merged["model_margin"]

    out = adapter.v2_dir / "oos_predictions.csv"
    adapter.v2_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"{league}: wrote {out} n={len(merged)} cols={len(merged.columns)}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", type=str, default=",".join(ALL_LEAGUES))
    args = parser.parse_args()
    for league in [x.strip() for x in args.leagues.split(",") if x.strip()]:
        try:
            merge_league(league)
        except Exception as exc:  # noqa: BLE001
            print(f"{league}: ERROR {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
