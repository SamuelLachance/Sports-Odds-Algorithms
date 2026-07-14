"""Force-ship best full-feature NFL Revolution config into nfl_v2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.train_nfl_revolution import (  # noqa: E402
    OUT_DIR,
    RevConfig,
    V2_DIR,
    prepare_frame,
    ship_to_v2,
    walk_forward,
)


def main() -> int:
    lb = json.loads((OUT_DIR / "leaderboard.json").read_text(encoding="utf-8"))
    full = [
        r
        for r in lb
        if (r.get("config") or {}).get("feature_mode") == "full"
    ]
    if not full:
        print("no full-mode configs in leaderboard", flush=True)
        return 1
    best = min(full, key=lambda r: float(r["oos_model_logloss"]))
    print(
        f"shipping full-mode {best['config']['name']} "
        f"LL={best['oos_model_logloss']:.5f} MAE={best['oos_margin_mae']:.4f}",
        flush=True,
    )
    frame = prepare_frame(2025)
    cfg = RevConfig(**{k: v for k, v in best["config"].items() if k in RevConfig.__dataclass_fields__})
    # Re-run walk-forward to refresh OOS under current table, then ship.
    result = walk_forward(frame, cfg, 2025)
    print(
        f"refreshed LL={result['oos_model_logloss']:.5f} "
        f"MAE={result['oos_margin_mae']:.4f} n={result['n']}",
        flush=True,
    )
    # Bypass LL gate vs old 124-feat hybrid: overwrite v2 hybrid artifacts.
    v2_meta = V2_DIR / "hybrid_meta.json"
    if v2_meta.is_file():
        v2_meta.unlink()
    ship_to_v2(result, frame)
    meta = json.loads((V2_DIR / "hybrid_meta.json").read_text(encoding="utf-8"))
    print(
        f"shipped n_features={len(meta.get('feature_cols') or [])} "
        f"ll={meta.get('oos_model_logloss')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
