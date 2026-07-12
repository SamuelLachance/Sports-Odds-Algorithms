"""Print OOS metrics from data/models/*_v2/metadata.json for all v2 leagues.

Usage:
  python scripts/check_v2_data.py
  python scripts/check_v2_data.py --json
  python scripts/check_v2_data.py --features
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_ROOT = PROJECT_ROOT / "data" / "models"

LEAGUES = ("nba", "wnba", "cbb", "nfl", "cfb", "nhl", "mlb", "soccer")

# league -> module path that defines FEATURE_COLUMNS (and optional market lists)
_ENGINE_MODULES: dict[str, str] = {
    "nba": "web.nba_v2.feature_engine",
    "wnba": "web.wnba_v2.feature_engine",
    "cbb": "web.cbb_v2.feature_engine",
    "nfl": "web.nfl_v2.feature_engine",
    "cfb": "web.cfb_v2.feature_engine",
    "nhl": "web.nhl_v2.feature_engine",
    "mlb": "web.mlb_v2.feature_engine",
    "soccer": "web.soccer_v2.feature_engine",
}

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

# Hollow metadata.json that only parses must not count as ok for --with-v2 gates.
REQUIRED_KEYS = ("oos_model_logloss", "train_rows", "created_at")


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

    if not isinstance(meta, dict):
        row["error"] = "incomplete metadata (not an object)"
        return row

    missing = [key for key in REQUIRED_KEYS if key not in meta]
    if missing:
        row["error"] = f"incomplete metadata (missing: {', '.join(missing)})"
        return row

    row["ok"] = True
    row["algorithm"] = meta.get("algorithm")
    for key in METRIC_KEYS:
        if key in meta:
            row[key] = meta[key]
    eval_seasons = meta.get("eval_seasons")
    if isinstance(eval_seasons, list) and eval_seasons:
        row["eval_seasons"] = f"{eval_seasons[0]}-{eval_seasons[-1]}"

    # Feature counts from shipped metadata (pure + market extras).
    pure = meta.get("feature_columns")
    if isinstance(pure, list):
        row["n_pure_features"] = len(pure)
    clf_mkt = meta.get("clf_market_features")
    if isinstance(clf_mkt, list):
        row["n_clf_market_features"] = len(clf_mkt)
    margin_mkt = meta.get("margin_market_features")
    if isinstance(margin_mkt, list):
        row["n_margin_market_features"] = len(margin_mkt)
    return row


def load_engine_feature_counts(league: str) -> dict:
    """Import FEATURE_COLUMNS from the league feature engine (code truth)."""
    row: dict = {"league": league, "ok": False}
    mod_name = _ENGINE_MODULES.get(league)
    if not mod_name:
        row["error"] = "no engine module mapping"
        return row
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001 — report per-league import failures
        row["error"] = f"import failed: {exc}"
        return row
    cols = getattr(mod, "FEATURE_COLUMNS", None)
    if cols is None:
        row["error"] = "FEATURE_COLUMNS missing"
        return row
    row["ok"] = True
    row["n_pure_features"] = len(tuple(cols))
    # Optional engine-level market tuples (rare; usually live in train scripts).
    for attr, key in (
        ("CLF_MARKET_FEATURES", "n_clf_market_features"),
        ("MARGIN_MARKET_FEATURES", "n_margin_market_features"),
        ("MARKET_FEATURES", "n_market_features"),
    ):
        extra = getattr(mod, attr, None)
        if extra is not None:
            row[key] = len(tuple(extra))
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


def print_feature_table(meta_rows: list[dict], engine_rows: list[dict]) -> None:
    cols = (
        "league",
        "engine_pure",
        "meta_pure",
        "clf_mkt",
        "margin_mkt",
        "meta_total",
    )
    widths = {c: len(c) for c in cols}
    engine_by = {r["league"]: r for r in engine_rows}
    display: list[dict[str, str]] = []
    for meta in meta_rows:
        league = str(meta["league"])
        eng = engine_by.get(league) or {}
        pure_meta = meta.get("n_pure_features")
        clf = meta.get("n_clf_market_features") or 0
        margin = meta.get("n_margin_market_features") or 0
        total = None
        if pure_meta is not None:
            total = int(pure_meta) + int(clf) + int(margin)
        line = {
            "league": league,
            "engine_pure": _fmt(eng.get("n_pure_features") if eng.get("ok") else eng.get("error")),
            "meta_pure": _fmt(pure_meta if meta.get("ok") else meta.get("error")),
            "clf_mkt": _fmt(clf if meta.get("ok") and "n_clf_market_features" in meta else "-"),
            "margin_mkt": _fmt(
                margin if meta.get("ok") and "n_margin_market_features" in meta else "-"
            ),
            "meta_total": _fmt(total),
        }
        display.append(line)
        for c in cols:
            widths[c] = max(widths[c], len(line[c]))

    print("Feature counts (engine FEATURE_COLUMNS vs shipped metadata)")
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
        "--features",
        action="store_true",
        help="Also report pure/market feature counts per league",
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
    engine_rows = (
        [load_engine_feature_counts(league) for league in leagues] if args.features else []
    )

    if args.features:
        for row, eng in zip(rows, engine_rows):
            row["engine_n_pure_features"] = eng.get("n_pure_features")
            if eng.get("ok"):
                row["engine_ok"] = True
            else:
                row["engine_error"] = eng.get("error")

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
        if args.features:
            print()
            print_feature_table(rows, engine_rows)
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
