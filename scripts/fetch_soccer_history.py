"""Fetch full football-data.co.uk history for soccer v2 (top-5 + 2nd tiers).

Downloads every division-season CSV into data/soccer_history/. Roughly 250
files / ~30 MB; safe to re-run (cached files are skipped, current season can
be refreshed with --refresh-current).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.soccer_v2.data import (  # noqa: E402
    HISTORY_DIR,
    all_division_codes,
    current_season,
    fetch_csv_text,
    seasons_for_division,
)


def main() -> None:
    refresh_current = "--refresh-current" in sys.argv
    through = current_season()
    total = ok = 0
    for division in all_division_codes():
        for season in seasons_for_division(division, through=through):
            total += 1
            force = refresh_current and season == through
            try:
                text = fetch_csv_text(division, season, force=force)
            except OSError as exc:
                print(f"{division} {season}: ERROR {exc}", flush=True)
                continue
            if text is None:
                print(f"{division} {season}: no data", flush=True)
                continue
            ok += 1
            if force:
                time.sleep(0.2)
    print(f"done: {ok}/{total} division-seasons cached under {HISTORY_DIR}")


if __name__ == "__main__":
    main()
