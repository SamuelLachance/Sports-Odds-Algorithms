"""Snapshot multi-book opening lines into the early-lines store.

Polls The Odds API (needs ODDS_API_KEY) for each requested league and records
first-seen lines per (game, book) — the observed open. BetOnline/Pinnacle/
Bovada openers typically post well before ESPN's single-provider feed moves,
so running this early (or on a cron) builds a true opener history that the
daily slate then reads for "opened by X at Y" and EV-at-open.

Cost: len(markets) x len(regions) credits per league per run (free tier:
500/month). Defaults are h2h,spreads,totals x us,us2,eu = 9 credits/league.

Usage:
  python scripts/snapshot_early_lines.py --leagues wnba
  python scripts/snapshot_early_lines.py --leagues wnba,mlb,nba
  EARLY_LINES_REGIONS=us,us2 python scripts/snapshot_early_lines.py --leagues wnba
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.early_lines import ODDS_API_SPORT_KEYS, snapshot_league_odds_api  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Requires ODDS_API_KEY in the environment (the-odds-api.com).",
    )
    parser.add_argument(
        "--leagues",
        default="wnba",
        help=f"Comma-separated leagues ({', '.join(sorted(ODDS_API_SPORT_KEYS))})",
    )
    args = parser.parse_args()
    leagues = [s.strip().lower() for s in args.leagues.split(",") if s.strip()]
    worst = 0
    for league in leagues:
        status = snapshot_league_odds_api(league)
        print(json.dumps(status))
        if status.get("status") == "no_key":
            print(
                "ODDS_API_KEY is not set — get a free key at https://the-odds-api.com "
                "and export it before running.",
                file=sys.stderr,
            )
            return 2
        if status.get("status") not in ("ok",):
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
