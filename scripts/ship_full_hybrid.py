"""Force-ship best full-feature hybrid config for a league into {league}_v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from web.hybrid_v2.adapters import get_adapter  # noqa: E402
from web.hybrid_v2.config import HybridConfig  # noqa: E402
from web.hybrid_v2.ship import ship_league  # noqa: E402
from web.hybrid_v2.train import prepare_frame, run_config  # noqa: E402


def ship_full(league: str) -> int:
    adapter = get_adapter(league)
    focused = adapter.hybrid_dir / "focused_final.json"
    if not focused.is_file():
        print(f"missing {focused} — run train_all_hybrids first", flush=True)
        return 1
    payload = json.loads(focused.read_text(encoding="utf-8"))
    board = payload.get("board") or []
    full = [
        r
        for r in board
        if (r.get("config") or {}).get("feature_mode") == "full"
        and r.get("oos_model_logloss") is not None
    ]
    if not full:
        print(f"[{league}] no full-mode configs on board", flush=True)
        return 1
    best = min(full, key=lambda r: float(r["oos_model_logloss"]))
    print(
        f"[{league}] shipping full-mode {best['config']['name']} "
        f"LL={best['oos_model_logloss']:.5f}",
        flush=True,
    )
    frame = prepare_frame(adapter)
    cfg = HybridConfig(**best["config"])
    summary = run_config(adapter, cfg, frame, int(frame.season.max()))
    (adapter.hybrid_dir / "best_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    info = ship_league(league, force=True)
    meta_path = adapter.v2_dir / "hybrid_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    print(
        f"[{league}] shipped feature_mode={meta.get('feature_mode')} "
        f"n_features={len(meta.get('feature_cols') or meta.get('clf_feature_cols') or [])} "
        f"info={info.get('shipped')}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", required=True)
    args = parser.parse_args()
    return ship_full(args.league.strip().lower())


if __name__ == "__main__":
    raise SystemExit(main())
