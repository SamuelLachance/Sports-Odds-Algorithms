"""Backfill WNBA closing odds from ESPN into data/supplemental/closing-odds/wnba.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.closing_odds_db import clear_closing_odds_cache  # noqa: E402
from web.wnba_odds_espn import CLOSING_FIELDS, OUTPUT_CSV, fetch_wnba_odds_rows  # noqa: E402
from web.tracking_service import toronto_today  # noqa: E402


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_existing(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _merge_rows(existing: list[dict[str, str]], fresh: list[dict]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in existing:
        key = (row["date"], row["home_key"], row["away_key"])
        merged[key] = {field: row.get(field, "") for field in CLOSING_FIELDS}

    for row in fresh:
        key = (row["date"], row["home_key"], row["away_key"])
        merged[key] = {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in CLOSING_FIELDS
        }

    return sorted(merged.values(), key=lambda item: (item["date"], item["home_key"]))


def write_merged_csv(rows: list[dict[str, str]], path: Path = OUTPUT_CSV) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLOSING_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill WNBA odds from ESPN.")
    parser.add_argument("--start", default="2017-05-01", type=_parse_date)
    parser.add_argument("--end", default=toronto_today().isoformat(), type=_parse_date)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print(f"Fetching WNBA odds {args.start} -> {args.end}...", flush=True)
    fresh = fetch_wnba_odds_rows(args.start, args.end, use_cache=not args.no_cache)
    existing = _load_existing(OUTPUT_CSV)
    merged = _merge_rows(existing, fresh)
    count = write_merged_csv(merged)
    clear_closing_odds_cache()
    print(f"Merged {OUTPUT_CSV} ({count} rows, +{len(fresh)} ESPN games)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
