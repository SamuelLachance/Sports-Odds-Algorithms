"""Orchestrate focused hybrid grids (+ optional ship) for all v2 leagues."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hybrid_v2.adapters import ALL_LEAGUES, get_adapter  # noqa: E402
from web.hybrid_v2.ship import ship_league  # noqa: E402
from web.hybrid_v2.train import run_focused_grid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train focused hybrid grids for v2 leagues")
    parser.add_argument(
        "--leagues",
        type=str,
        default=",".join(ALL_LEAGUES),
        help="Comma-separated leagues",
    )
    parser.add_argument("--max-configs", type=int, default=8)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--ship", action="store_true", help="Ship winners that beat baseline")
    parser.add_argument("--force-ship", action="store_true")
    parser.add_argument("--skip-missing-tables", action="store_true", default=True)
    args = parser.parse_args()

    leagues = [x.strip().lower() for x in args.leagues.split(",") if x.strip()]
    results = []
    for league in leagues:
        adapter = get_adapter(league)
        if not adapter.table_path.is_file():
            msg = f"SKIP {league}: missing {adapter.table_path}"
            print(msg, flush=True)
            results.append({"league": league, "skipped": True, "reason": "missing_table"})
            continue
        try:
            final = run_focused_grid(
                league,
                end_season=args.end_season,
                max_configs=args.max_configs,
            )
            ship_info = None
            if args.ship or args.force_ship:
                ship_info = ship_league(league, force=args.force_ship)
            results.append({"league": league, "final": final, "ship": ship_info})
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {league}: {exc}", flush=True)
            results.append({"league": league, "error": str(exc)})

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out_path = PROJECT_ROOT / "data" / "models" / "hybrid_all_leagues_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
