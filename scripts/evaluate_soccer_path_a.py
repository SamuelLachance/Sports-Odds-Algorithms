"""Grade Soccer Path A walk-forward calibration vs closing lines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_meta_model import META_WEIGHTS_PATH  # noqa: E402

from web.league_profiles import INTERNATIONAL_SOCCER_LEAGUES  # noqa: E402

TOP_LEAGUES = ("epl", "bundesliga", "laliga", "seriea", "ligue1")
INTERNATIONAL_LEAGUES = INTERNATIONAL_SOCCER_LEAGUES


def grade_entry(entry: dict) -> str:
    calibrated = entry.get("log_loss_calibrated")
    market = entry.get("log_loss_market_closing")
    path_a = entry.get("log_loss_tuned_path_a")
    if calibrated is None:
        return "C"
    if market is not None and calibrated <= market:
        return "A+"
    if market is not None and calibrated <= market + 0.005:
        return "A"
    if path_a is not None and calibrated < path_a:
        return "B+"
    if calibrated < 1.0:
        return "B"
    return "C"


def main() -> int:
    payload = json.loads(META_WEIGHTS_PATH.read_text(encoding="utf-8"))
    print("Soccer Path A calibration grades\n")
    print("Top domestic leagues:")
    for league in TOP_LEAGUES:
        entry = payload.get(league) or {}
        if not entry:
            print(f"  {league}: missing")
            continue
        grade = grade_entry(entry)
        print(
            f"  {league}: {grade} "
            f"(calibrated={entry.get('log_loss_calibrated')} "
            f"market={entry.get('log_loss_market_closing')} "
            f"path_a={entry.get('log_loss_tuned_path_a')} "
            f"beats_market={entry.get('beats_market_closing')})"
        )
    print("\nInternational:")
    for league in INTERNATIONAL_LEAGUES:
        entry = payload.get(league) or {}
        if not entry:
            print(f"  {league}: missing")
            continue
        grade = grade_entry(entry)
        print(
            f"  {league}: {grade} "
            f"(calibrated={entry.get('log_loss_calibrated')} "
            f"market={entry.get('log_loss_market_closing')} "
            f"path_a={entry.get('log_loss_tuned_path_a')} "
            f"beats_market={entry.get('beats_market_closing')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
