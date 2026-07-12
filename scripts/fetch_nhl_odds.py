"""Backfill NHL closing odds from ESPN and merge with existing SBR archive rows."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.closing_odds_db import ODDS_DIR, clear_closing_odds_cache  # noqa: E402
from web.nhl_odds_espn import CLOSING_FIELDS, OUTPUT_CSV, fetch_nhl_odds_rows  # noqa: E402
from web.tracking_service import toronto_today  # noqa: E402


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_existing(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_has_odds(row: dict[str, str]) -> bool:
    """True when a closing row has usable ML closes.

    ``n_books>0`` alone is not enough — ESPN can emit spread/total-only rows that
    must not wipe prior SBR moneylines.
    """
    return bool(
        (row.get("home_close_ml") or "").strip()
        or (row.get("away_close_ml") or "").strip()
    )


_MARKET_VALUE_FIELDS = (
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


def _row_has_market_values(row: dict[str, str]) -> bool:
    return any((row.get(field) or "").strip() for field in _MARKET_VALUE_FIELDS)


def _fill_empty_from_prior(prior: dict[str, str], fresh: dict[str, str]) -> dict[str, str]:
    """Keep prior odds fields when the fresh refresh leaves them blank."""
    out = dict(fresh)
    for field in CLOSING_FIELDS:
        if field in {"date", "home_key", "away_key", "source"}:
            continue
        if not (out.get(field) or "").strip() and (prior.get(field) or "").strip():
            out[field] = prior[field]
    if not (out.get("source") or "").strip():
        out["source"] = prior.get("source", "")
    return out


def _merge_rows(existing: list[dict[str, str]], fresh: list[dict]) -> list[dict[str, str]]:
    """Merge by (date, home, away). Keep prior closes when fresh odds are empty."""
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in existing:
        key = (row["date"], row["home_key"], row["away_key"])
        merged[key] = {field: row.get(field, "") for field in CLOSING_FIELDS}

    for row in fresh:
        key = (row["date"], row["home_key"], row["away_key"])
        normalized = {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in CLOSING_FIELDS
        }
        prior = merged.get(key)
        if prior is not None and _row_has_odds(prior) and not _row_has_odds(normalized):
            if not _row_has_market_values(normalized):
                continue  # empty stub — keep prior wholly
            # Spread/total-only refresh: keep prior MLs, still accept fresh fields.
            merged[key] = _fill_empty_from_prior(prior, normalized)
            continue
        if prior is not None:
            normalized = _fill_empty_from_prior(prior, normalized)
        merged[key] = normalized

    return sorted(merged.values(), key=lambda item: (item["date"], item["home_key"]))


def write_merged_csv(rows: list[dict[str, str]], path: Path = OUTPUT_CSV) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLOSING_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill NHL odds from ESPN.")
    parser.add_argument("--start", default="2022-10-01", type=_parse_date)
    parser.add_argument("--end", default=toronto_today().isoformat(), type=_parse_date)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print(f"Fetching NHL odds {args.start} -> {args.end}...", flush=True)
    fresh = fetch_nhl_odds_rows(args.start, args.end, use_cache=not args.no_cache)
    existing = _load_existing(OUTPUT_CSV)
    merged = _merge_rows(existing, fresh)
    count = write_merged_csv(merged)
    clear_closing_odds_cache()
    print(f"Merged {OUTPUT_CSV} ({count} rows, +{len(fresh)} ESPN games)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
