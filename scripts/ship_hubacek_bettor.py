"""Ship the meta-selected Hubáček β model for live telemetry.

Reads data/models/{league}_hybrid/hubacek_meta.json (written by
hubacek_meta_select.py), refits the latest-season-selected config on full
history, and writes data/models/{league}_v2/model_clf_bettor.cbm +
bettor_meta.json (incl. an isotonic calibrator from that config's OOS and
the meta-selection record). Does NOT touch the π bundle or pick gating.

Usage:
  python scripts/ship_hubacek_bettor.py --leagues nba
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hybrid_v2.adapters import get_adapter  # noqa: E402
from web.hybrid_v2.config import focused_search_space  # noqa: E402
from web.hybrid_v2.train import (  # noqa: E402
    _fit_catboost_decorrelated,
    prepare_frame,
    select_feature_columns,
)


def ship_league(league: str) -> dict | None:
    adapter = get_adapter(league)
    meta_path = adapter.hybrid_dir / "hubacek_meta.json"
    if not meta_path.is_file():
        print(f"[{league}] no hubacek_meta.json — run hubacek_meta_select first")
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seasons = meta.get("seasons") or []
    if not seasons:
        print(f"[{league}] meta-selection has no seasons")
        return None
    latest = seasons[-1]
    cfg = next(
        (c for c in focused_search_space() if c.name == latest["config"] and c.target_mode == "decorrelated"),
        None,
    )
    if cfg is None:
        print(f"[{league}] config {latest['config']} not in grid")
        return None

    frame = prepare_frame(adapter, end_season=adapter.end_season)
    cols = select_feature_columns(adapter, cfg)
    model, feature_cols, cat_cols = _fit_catboost_decorrelated(frame, cols, cfg)

    v2 = adapter.v2_dir
    v2.mkdir(parents=True, exist_ok=True)
    model_path = v2 / "model_clf_bettor.cbm"
    model.save_model(str(model_path))

    calibrator = None
    oos_path = adapter.hybrid_dir / f"oos_{latest['config']}.csv"
    if oos_path.is_file():
        try:
            from sklearn.isotonic import IsotonicRegression

            oos = pd.read_csv(oos_path, low_memory=False)
            if {"model_raw", "home_win"}.issubset(oos.columns) and len(oos) >= 500:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
                iso.fit(oos["model_raw"].astype(float), oos["home_win"].astype(float))
                calibrator = {
                    "x": [float(v) for v in iso.X_thresholds_],
                    "y": [float(v) for v in iso.y_thresholds_],
                    "kind": "isotonic_oos",
                }
        except Exception:  # noqa: BLE001 — calibrator optional
            calibrator = None

    bettor_meta = {
        "config": cfg.to_dict(),
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "decorrelation_c": cfg.decorrelation_c,
        "phi": latest.get("phi"),
        "calibrator": calibrator,
        "meta_selection": {
            k: meta.get(k)
            for k in (
                "mean_profit_per_round_pct",
                "worst_season_profit_pct",
                "positive_seasons",
                "total_seasons",
                "protocol",
            )
        },
        "shipped_at": datetime.now(timezone.utc).isoformat(),
        "note": "Hubacek beta (decorrelated) model — display/EV telemetry; official pick gating unchanged.",
    }
    (v2 / "bettor_meta.json").write_text(json.dumps(bettor_meta, indent=2), encoding="utf-8")
    print(
        f"[{league}] shipped beta {cfg.name} (c={cfg.decorrelation_c}, phi={latest.get('phi')}) "
        f"meta mean {meta.get('mean_profit_per_round_pct')}%/round worst {meta.get('worst_season_profit_pct')}%"
    )
    return bettor_meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leagues", default="nba")
    args = parser.parse_args()
    for league in [s.strip().lower() for s in args.leagues.split(",") if s.strip()]:
        try:
            ship_league(league)
        except Exception as exc:  # noqa: BLE001
            print(f"[{league}] FAILED: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
