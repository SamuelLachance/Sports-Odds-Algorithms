"""Decorrelated Hubáček CFB niche search (ML + spread)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hubacek_binary_search import run_binary_league_search  # noqa: E402

V2 = PROJECT_ROOT / "data" / "models" / "cfb_v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-policy", action="store_true")
    parser.add_argument("--min-bets", type=int, default=40)
    parser.add_argument("--min-seasons", type=int, default=3)
    args = parser.parse_args()

    path = V2 / "oos_predictions_hybrid.csv"
    if not path.is_file():
        path = V2 / "oos_predictions.csv"
    if not path.is_file():
        raise SystemExit(f"missing OOS file under {V2}")

    frame = pd.read_csv(path)
    print(f"loaded cfb n={len(frame)} seasons={sorted(frame.season.unique().tolist())}", flush=True)

    run_binary_league_search(
        league="cfb",
        frame=frame,
        source="hybrid",
        project_root=PROJECT_ROOT,
        write_policy=args.write_policy,
        min_bets=args.min_bets,
        min_seasons=args.min_seasons,
        require_all_positive=True,
        ml_prices=(
            ("close", "home_close_ml", "away_close_ml"),
            ("open", "home_open_ml", "away_open_ml"),
        ),
        windows=(("all", None), ("from2023", 2023), ("from2024", 2024)),
        also_spread=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
