"""Backfill MLB closing odds from ESPN and merge with existing SBR archive rows."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.closing_odds_db import clear_closing_odds_cache  # noqa: E402
from web.mlb_odds_espn import CLOSING_FIELDS, OUTPUT_CSV, fetch_mlb_odds_rows  # noqa: E402
from web.tracking_service import toronto_today  # noqa: E402


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_existing(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _closing_signature(row: dict[str, str]) -> tuple[str, ...]:
    """Odds fields only — identical refreshes collapse; DH conflicts stay."""
    return tuple(
        row.get(field, "")
        for field in (
            "home_close_ml",
            "away_close_ml",
            "home_open_ml",
            "away_open_ml",
            "home_close_spread",
            "away_close_spread",
            "home_spread_odds",
            "away_spread_odds",
            "close_total",
            "open_total",
        )
    )


def _normalize_closing_row(row: dict, *, from_fresh: bool) -> dict[str, str]:
    if from_fresh:
        return {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in CLOSING_FIELDS
        }
    return {field: row.get(field, "") for field in CLOSING_FIELDS}


def _row_has_odds(row: dict[str, str]) -> bool:
    """True when a closing row has usable ML closes (or n_books>0 when present)."""
    if (row.get("home_close_ml") or "").strip() or (row.get("away_close_ml") or "").strip():
        return True
    try:
        return int(float(row.get("n_books") or 0)) > 0
    except (TypeError, ValueError):
        return False


def _home_ml_distance(left: dict[str, str], right: dict[str, str]) -> float:
    """Absolute home-ML gap for matching a fresh DH refresh to a prior leg."""
    try:
        return abs(float(left.get("home_close_ml") or 0) - float(right.get("home_close_ml") or 0))
    except (TypeError, ValueError):
        return float("inf")


def _merge_rows(existing: list[dict[str, str]], fresh: list[dict]) -> list[dict[str, str]]:
    """Merge ESPN refreshes into the CSV without collapsing MLB doubleheaders.

    Fresh rows for a key replace prior rows for that key. Multiple fresh rows
    with distinct odds under the same (date, home, away) are all kept so
    ``closing_odds_db`` can mark the key ambiguous instead of last-row-wins.

    Empty ESPN stubs (n_books=0) must not wipe prior SBR/ESPN closes, and an
    empty stub alongside a real DH sibling must not poison the key as ambiguous.

    A partial refresh that returns fewer priced legs than an existing DH must
    replace the nearest prior leg(s) and keep unmatched siblings so
    ``closing_odds_db`` can still fail closed on ambiguity.
    """
    fresh_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in fresh:
        key = (str(row["date"]), str(row["home_key"]), str(row["away_key"]))
        norm = _normalize_closing_row(row, from_fresh=True)
        bucket = fresh_by_key.setdefault(key, [])
        sig = _closing_signature(norm)
        if any(_closing_signature(prev) == sig for prev in bucket):
            continue
        bucket.append(norm)

    existing_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in existing:
        key = (row["date"], row["home_key"], row["away_key"])
        norm = _normalize_closing_row(row, from_fresh=False)
        bucket = existing_by_key.setdefault(key, [])
        sig = _closing_signature(norm)
        if any(_closing_signature(prev) == sig for prev in bucket):
            continue
        bucket.append(norm)

    merged: list[dict[str, str]] = []
    for key in sorted(set(existing_by_key) | set(fresh_by_key)):
        if key in fresh_by_key:
            fresh_bucket = fresh_by_key[key]
            with_odds = [row for row in fresh_bucket if _row_has_odds(row)]
            existing_odds = [
                row for row in existing_by_key.get(key, []) if _row_has_odds(row)
            ]
            if with_odds:
                if len(existing_odds) > len(with_odds):
                    remaining = list(existing_odds)
                    out: list[dict[str, str]] = []
                    for fresh_row in with_odds:
                        best_i = min(
                            range(len(remaining)),
                            key=lambda i: _home_ml_distance(remaining[i], fresh_row),
                        )
                        remaining.pop(best_i)
                        out.append(fresh_row)
                    out.extend(remaining)
                    merged.extend(out)
                else:
                    merged.extend(with_odds)
            elif existing_odds:
                merged.extend(existing_by_key[key])
            else:
                merged.extend(fresh_bucket)
        else:
            merged.extend(existing_by_key[key])
    return sorted(merged, key=lambda item: (item["date"], item["home_key"]))


def write_merged_csv(rows: list[dict[str, str]], path: Path = OUTPUT_CSV) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLOSING_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill MLB odds from ESPN.")
    parser.add_argument("--start", default="2022-04-01", type=_parse_date)
    parser.add_argument("--end", default=toronto_today().isoformat(), type=_parse_date)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print(f"Fetching MLB odds {args.start} -> {args.end}...", flush=True)
    fresh = fetch_mlb_odds_rows(args.start, args.end, use_cache=not args.no_cache)
    existing = _load_existing(OUTPUT_CSV)
    merged = _merge_rows(existing, fresh)
    count = write_merged_csv(merged)
    clear_closing_odds_cache()
    print(f"Merged {OUTPUT_CSV} ({count} rows, +{len(fresh)} ESPN games)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
