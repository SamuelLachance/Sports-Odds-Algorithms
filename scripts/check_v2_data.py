"""Print OOS metrics from data/models/*_v2/metadata.json for all v2 leagues.

Usage:
  python scripts/check_v2_data.py
  python scripts/check_v2_data.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "data" / "models"

LEAGUES = ("nba", "wnba", "cbb", "nfl", "cfb", "nhl", "mlb", "soccer")

METRIC_KEYS = (
    "oos_model_logloss",
    "oos_model_brier",
    "oos_model_acc",
    "oos_margin_mae",
    "oos_elo_logloss",
    "oos_elo_margin_mae",
    "oos_market_logloss",
    "oos_close_spread_mae",
    "oos_with_odds",
    "ship_models",
    "train_rows",
    "created_at",
)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def load_league(league: str) -> dict:
    meta_path = MODELS_ROOT / f"{league}_v2" / "metadata.json"
    row: dict = {"league": league, "path": str(meta_path), "ok": False}
    if not meta_path.is_file():
        row["error"] = "missing metadata.json"
        return row
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row["error"] = f"unreadable: {exc}"
        return row

    row["ok"] = True
    row["algorithm"] = meta.get("algorithm")
    for key in METRIC_KEYS:
        if key in meta:
            row[key] = meta[key]
    eval_seasons = meta.get("eval_seasons")
    if isinstance(eval_seasons, list) and eval_seasons:
        row["eval_seasons"] = f"{eval_seasons[0]}-{eval_seasons[-1]}"
    return row


def print_table(rows: list[dict]) -> None:
    cols = (
        "league",
        "oos_model_logloss",
        "oos_model_acc",
        "oos_margin_mae",
        "oos_market_logloss",
        "ship_models",
        "train_rows",
        "created_at",
    )
    widths = {c: len(c) for c in cols}
    display: list[dict[str, str]] = []
    for row in rows:
        if not row.get("ok"):
            line = {c: "-" for c in cols}
            line["league"] = str(row["league"])
            line["oos_model_logloss"] = row.get("error", "missing")
            display.append(line)
            continue
        line = {c: _fmt(row.get(c)) for c in cols}
        # shorten ISO timestamps
        created = line.get("created_at") or "-"
        if created != "-" and "T" in created:
            line["created_at"] = created.split("T", 1)[0]
        display.append(line)
        for c in cols:
            widths[c] = max(widths[c], len(line[c]))

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for line in display:
        print("  ".join(line[c].ljust(widths[c]) for c in cols))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print walk-forward OOS metrics from all *_v2 metadata.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table",
    )
    parser.add_argument(
        "--league",
        action="append",
        choices=list(LEAGUES),
        help="Limit to one or more leagues (repeatable)",
    )
    args = parser.parse_args()

    leagues = args.league or list(LEAGUES)
    rows = [load_league(league) for league in leagues]

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
        missing = [r["league"] for r in rows if not r.get("ok")]
        if missing:
            print(
                f"\nmissing/unreadable: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
