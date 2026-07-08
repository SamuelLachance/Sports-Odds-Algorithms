"""Fetch football-data.co.uk international closing-odds CSV archives."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.international_soccer_odds import (  # noqa: E402
    CACHE_DIR,
    SOURCE_FILES,
    _fetch_source_file,
    international_odds_metadata,
    load_international_odds_index,
)


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        print(f"Fetching {filename}...", flush=True)
        text = _fetch_source_file(filename)
        if not text:
            print(f"  failed", flush=True)
            continue
        print(f"  cached {len(text)} bytes", flush=True)
    load_international_odds_index.cache_clear()
    meta = international_odds_metadata()
    print(f"International odds index: {meta['rows']} keyed rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
