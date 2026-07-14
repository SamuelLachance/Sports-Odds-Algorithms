"""Download Torvik pregame win probs (public toRvik-data) for CBB features.

Source: https://github.com/andreweatherman/toRvik-data/tree/main/pregame_prob
Coverage: seasons ending 2008–2023.

Usage:
  python scripts/fetch_torvik_pregame.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "supplemental" / "torvik-pregame"
USER_AGENT = "Sports-Odds-Algorithms/2.0"
BASE = (
    "https://raw.githubusercontent.com/andreweatherman/toRvik-data/"
    "main/pregame_prob"
)


def fetch_year(year: int, *, force: bool = False) -> Path | None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"pregame_{year}.csv"
    if dest.is_file() and dest.stat().st_size > 50_000 and not force:
        print(f"skip existing {dest.name} ({dest.stat().st_size:,} bytes)", flush=True)
        return dest
    url = f"{BASE}/pregame_{year}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {year}: {exc}", flush=True)
        return None
    if len(data) < 5_000:
        print(f"FAIL {year}: tiny payload {len(data)}", flush=True)
        return None
    dest.write_bytes(data)
    print(f"wrote {dest} ({len(data):,} bytes)", flush=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ok = 0
    for year in range(args.start_year, args.end_year + 1):
        if fetch_year(year, force=args.force):
            ok += 1
    print(f"done: {ok} year files in {OUT_DIR}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
