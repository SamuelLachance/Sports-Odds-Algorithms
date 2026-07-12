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
    """True when a closing row has usable ML closes.

    ``n_books>0`` alone is not enough — ESPN can emit run-line-only rows that
    must not wipe prior SBR moneylines or poison DH matching.
    """
    return bool(
        (row.get("home_close_ml") or "").strip()
        or (row.get("away_close_ml") or "").strip()
    )


def _parse_home_ml(row: dict[str, str]) -> float | None:
    raw = (row.get("home_close_ml") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _home_ml_distance(left: dict[str, str], right: dict[str, str]) -> float:
    """Absolute home-ML gap for matching a fresh DH refresh to a prior leg."""
    left_ml = _parse_home_ml(left)
    right_ml = _parse_home_ml(right)
    if left_ml is None or right_ml is None:
        return float("inf")
    return abs(left_ml - right_ml)


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


def _pair_fresh_to_existing(
    existing_odds: list[dict[str, str]],
    with_odds: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Match fresh priced legs to nearest prior legs; keep unmatched priors."""
    remaining = list(existing_odds)
    out: list[dict[str, str]] = []
    unmatched_fresh: list[dict[str, str]] = []
    for fresh_row in with_odds:
        if not remaining:
            unmatched_fresh.append(fresh_row)
            continue
        best_i = min(
            range(len(remaining)),
            key=lambda i: _home_ml_distance(remaining[i], fresh_row),
        )
        if _home_ml_distance(remaining[best_i], fresh_row) == float("inf"):
            unmatched_fresh.append(fresh_row)
            continue
        prior = remaining.pop(best_i)
        out.append(_fill_empty_from_prior(prior, fresh_row))
    out.extend(remaining)
    out.extend(unmatched_fresh)
    return out


def _merge_rows(existing: list[dict[str, str]], fresh: list[dict]) -> list[dict[str, str]]:
    """Merge ESPN refreshes into the CSV without collapsing MLB doubleheaders.

    Fresh rows for a key replace prior rows for that key. Multiple fresh rows
    with distinct odds under the same (date, home, away) are all kept so
    ``closing_odds_db`` can mark the key ambiguous instead of last-row-wins.

    Empty ESPN stubs (no ML closes) must not wipe prior SBR/ESPN closes, and an
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
                if existing_odds:
                    merged.extend(_pair_fresh_to_existing(existing_odds, with_odds))
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
