"""Download and normalize public closing-odds archives into data/supplemental/closing-odds/."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.closing_odds_db import ODDS_DIR, clear_closing_odds_cache  # noqa: E402
from web.nflverse_odds import NFLVERSE_URL, write_nfl_cache  # noqa: E402
from web.sbr_odds import write_sbr_cache  # noqa: E402

DEFAULT_SBR_SEASONS = list(range(2015, 2026))


def _download_nflverse_source() -> Path:
    target = ODDS_DIR / "nflverse_games.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        NFLVERSE_URL,
        headers={"User-Agent": "Sports-Odds-Algorithms/2.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())
    return target


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch public closing odds archives.")
    parser.add_argument(
        "--sports",
        default="nba,nfl,nhl,mlb",
        help="Comma-separated leagues to refresh.",
    )
    parser.add_argument(
        "--seasons",
        default=",".join(str(year) for year in DEFAULT_SBR_SEASONS),
        help="Season start years for SBR archives.",
    )
    args = parser.parse_args()
    sports = [sport.strip().lower() for sport in args.sports.split(",") if sport.strip()]
    seasons = [int(year) for year in args.seasons.split(",") if year.strip()]

    if "nfl" in sports:
        print("Downloading nflverse games.csv...", flush=True)
        _download_nflverse_source()
        count = write_nfl_cache()
        print(f"Wrote nfl.csv ({count} rows)", flush=True)

    for sport in sports:
        if sport == "nfl":
            continue
        print(f"Fetching SBR archives for {sport}...", flush=True)
        count = write_sbr_cache(sport, seasons)
        print(f"Wrote {sport}.csv ({count} rows)", flush=True)

    clear_closing_odds_cache()
    print(f"Closing odds cache ready in {ODDS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
