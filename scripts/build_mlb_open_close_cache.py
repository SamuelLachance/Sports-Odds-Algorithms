"""Merge SBR MLB open/close archive into data/supplemental/closing-odds/mlb.csv.

Keeps richer online/ESPN rows when present; fills missing opens from the
public flancast90 SBR archive. Never invents open=close.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.sbr_odds import fetch_sbr_archive_rows  # noqa: E402

OUT = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "mlb.csv"

FIELDS = [
    "date",
    "home_key",
    "away_key",
    "home_close_ml",
    "away_close_ml",
    "home_open_ml",
    "away_open_ml",
    "home_close_spread",
    "away_close_spread",
    "home_open_spread",
    "away_open_spread",
    "home_spread_odds",
    "away_spread_odds",
    "open_total",
    "close_total",
    "n_books",
    "source",
]


def _key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("date") or "")[:10],
        str(row.get("home_key") or "").lower().strip(),
        str(row.get("away_key") or "").lower().strip(),
    )


def _has(val) -> bool:
    if val is None or val == "":
        return False
    try:
        return val == val  # NaN check
    except Exception:
        return True


def main() -> int:
    existing: dict[tuple[str, str, str], dict] = {}
    if OUT.is_file():
        with OUT.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                existing[_key(row)] = dict(row)

    archive = fetch_sbr_archive_rows("mlb")
    print(f"existing={len(existing)} archive={len(archive)}", flush=True)
    for row in archive:
        key = _key(row)
        cur = existing.get(key)
        if cur is None:
            existing[key] = {k: row.get(k) for k in FIELDS}
            existing[key]["source"] = row.get("source") or "sbr-archive"
            continue
        # Fill blanks only — never overwrite a real open/close with empty.
        for field in FIELDS:
            if field in {"date", "home_key", "away_key", "source", "n_books"}:
                continue
            if not _has(cur.get(field)) and _has(row.get(field)):
                cur[field] = row.get(field)
        if "sbr" not in str(cur.get("source") or ""):
            src = str(cur.get("source") or "")
            cur["source"] = f"{src}+sbr-archive" if src else "sbr-archive"

    ordered = sorted(existing.values(), key=lambda r: (_key(r)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k) for k in FIELDS})
    open_n = sum(1 for r in ordered if _has(r.get("home_open_ml")))
    print(f"wrote {OUT} rows={len(ordered)} open_ml={open_n} ({open_n/max(len(ordered),1):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
