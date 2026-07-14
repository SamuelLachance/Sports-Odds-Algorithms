"""Build season-to-date rolling Statcast team xwOBA dumps.

Public Baseball Savant expected-stats leaderboard. Training join uses the latest
snapshot with ``asof_date < game_date`` (never same-day).

Outputs data/supplemental/mlb-statcast/team_xwoba_rolling.csv

Usage:
  python scripts/build_mlb_statcast_rolling.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_mlb_statcast_priors import fetch_team_batter_year  # noqa: E402
from web.mlb_v2.statsapi_data import fetch_team_ids  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "supplemental" / "mlb-statcast"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for year in range(args.start_year, args.end_year + 1):
        asof = date(year, 10, 1).isoformat()
        try:
            team_ids = fetch_team_ids(year)
        except Exception as exc:  # noqa: BLE001
            print(f"team_ids {year}: {exc}", flush=True)
            continue
        n = 0
        # fetch_team_ids may return list[int] or dict
        if isinstance(team_ids, dict):
            items = list(team_ids.items())
        else:
            items = [(tid, "") for tid in team_ids]
        for team_id, abbr in items:
            try:
                rec = fetch_team_batter_year(year, int(team_id))
            except Exception as exc:  # noqa: BLE001
                print(f"  {year} {abbr or team_id}: {exc}", flush=True)
                time.sleep(args.sleep)
                continue
            if not rec:
                time.sleep(args.sleep)
                continue
            rows.append(
                {
                    "team_id": int(team_id),
                    "team_abbr": str(abbr or rec.get("team_abbr") or "").lower(),
                    "season": year,
                    "asof_date": asof,
                    "xwoba": rec.get("xwoba"),
                    "pa": rec.get("pa"),
                    "for_season": year + 1,
                }
            )
            n += 1
            time.sleep(args.sleep)
        print(f"season {year}: {n} teams", flush=True)

    if not rows:
        print("ERROR: no rolling rows", file=sys.stderr)
        return 1
    out = pd.DataFrame(rows)
    dest = OUT_DIR / "team_xwoba_rolling.csv"
    out.to_csv(dest, index=False)
    (OUT_DIR / "ROLLING_SOURCES.md").write_text(
        "# MLB rolling Statcast\n\n"
        "`team_xwoba_rolling.csv` from `scripts/build_mlb_statcast_rolling.py` "
        "(Baseball Savant expected statistics). Join with `asof_date < game_date`.\n",
        encoding="utf-8",
    )
    print(f"wrote {dest} rows={len(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
