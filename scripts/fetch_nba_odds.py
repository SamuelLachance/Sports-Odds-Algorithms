"""Backfill the real NBA closing/opening odds table from ESPN core odds API.

Writes data/supplemental/closing-odds/nba.csv with median-consensus opening and
closing spreads, moneylines, and totals. Resumable via per-day cache under
.build-cache/nba-odds/.

Usage:
    python scripts/fetch_nba_odds.py --start 2016-10-01 --end 2026-07-01
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.closing_odds_db import clear_closing_odds_cache  # noqa: E402
from web.nba_odds_espn import OUTPUT_CSV, OUTPUT_FIELDS, fetch_nba_odds_rows  # noqa: E402


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _round(value, ndigits=1):
    if value is None or value == "":
        return ""
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return ""


def _int(value):
    if value is None or value == "":
        return ""
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return ""


def write_csv(rows: list[dict], path: Path = OUTPUT_CSV) -> int:
    deduped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (row["date"], row["home_key"], row["away_key"])
        deduped[key] = row
    ordered = sorted(deduped.values(), key=lambda r: (r["date"], r["home_key"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "date": row["date"],
                    "home_key": row["home_key"],
                    "away_key": row["away_key"],
                    "home_final": _int(row.get("home_final")),
                    "away_final": _int(row.get("away_final")),
                    "home_close_ml": _int(row.get("home_close_ml")),
                    "away_close_ml": _int(row.get("away_close_ml")),
                    "home_close_spread": _round(row.get("home_close_spread")),
                    "away_close_spread": _round(row.get("away_close_spread")),
                    "home_spread_odds": _int(row.get("home_spread_odds")),
                    "away_spread_odds": _int(row.get("away_spread_odds")),
                    "home_open_spread": _round(row.get("home_open_spread")),
                    "away_open_spread": _round(row.get("away_open_spread")),
                    "close_total": _round(row.get("close_total")),
                    "open_total": _round(row.get("open_total")),
                    "n_books": _int(row.get("n_books")),
                    "source": row.get("source") or "espn-core",
                }
            )
    return len(ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill NBA odds from ESPN.")
    parser.add_argument("--start", default="2016-10-01", type=_parse_date)
    parser.add_argument("--end", default=date.today().isoformat(), type=_parse_date)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print(f"Fetching NBA odds {args.start} -> {args.end}...", flush=True)
    rows = fetch_nba_odds_rows(args.start, args.end, use_cache=not args.no_cache)
    count = write_csv(rows)
    clear_closing_odds_cache()
    print(f"Wrote {OUTPUT_CSV} ({count} games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
