"""Overnight WNBA hybrid search — iterate configs until local 6:00 AM.

Keeps the best OOS logloss vs baseline (XGB+LR 0.62284). Writes a running
leaderboard under data/models/wnba_hybrid/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_wnba_hybrid import (  # noqa: E402
    BASELINE_LL,
    OUT_DIR,
    HybridConfig,
    default_search_space,
    prepare_frame,
    run_config,
)

ET = ZoneInfo("America/New_York")


def _now() -> datetime:
    return datetime.now(tz=ET)


def _stop_at(hour: int = 6, minute: int = 0) -> datetime:
    now = _now()
    stop = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= stop:
        # already past today's stop — run until tomorrow 6am
        from datetime import timedelta

        stop = stop + timedelta(days=1)
    return stop


def _mutate(cfg: HybridConfig, round_idx: int) -> HybridConfig:
    """Perturb a strong config for later overnight rounds."""
    depths = (4, 5, 6, 7, 8)
    lrs = (0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06)
    modes = ("full", "curves_plus_core", "core")
    backends = ("catboost", "lightgbm", "both")
    blends = (0.0, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6)
    return HybridConfig(
        name=f"mut_r{round_idx}_{cfg.backend}_d{depths[round_idx % len(depths)]}_"
        f"lr{str(lrs[round_idx % len(lrs)]).replace('.', '')}_"
        f"{modes[round_idx % len(modes)]}_w{int(blends[round_idx % len(blends)]*100)}",
        backend=backends[round_idx % len(backends)],
        depth=depths[round_idx % len(depths)],
        learning_rate=lrs[(round_idx * 3) % len(lrs)],
        iterations=400 + 50 * (round_idx % 8),
        l2=2.0 + (round_idx % 6),
        min_data=20 + 10 * (round_idx % 5),
        subsample=0.8 + 0.05 * (round_idx % 4),
        use_curves=(round_idx % 5) != 4,
        use_categoricals=True,
        use_market=True,
        market_blend_w=blends[round_idx % len(blends)],
        feature_mode=modes[(round_idx * 2) % len(modes)],
        early_stopping=30 + 5 * (round_idx % 4),
        seed=42 + round_idx,
    )


def _load_all_summaries() -> list[dict]:
    """Merge every summary_*.json so restarts never drop best runs."""
    by_name: dict[str, dict] = {}
    for path in OUT_DIR.glob("summary_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = (payload.get("config") or {}).get("name") or path.stem
        ll = payload.get("oos_model_logloss", 9.0)
        prev = by_name.get(name)
        if prev is None or ll < prev.get("oos_model_logloss", 9.0):
            by_name[name] = payload
    return sorted(by_name.values(), key=lambda r: r.get("oos_model_logloss", 9.0))


def _persist_board(results: list[dict], board_path: Path) -> list[dict]:
    merged = { (r.get("config") or {}).get("name"): r for r in _load_all_summaries() }
    for r in results:
        name = (r.get("config") or {}).get("name")
        if not name:
            continue
        prev = merged.get(name)
        if prev is None or r.get("oos_model_logloss", 9) <= prev.get("oos_model_logloss", 9):
            merged[name] = r
    ordered = sorted(merged.values(), key=lambda r: r.get("oos_model_logloss", 9.0))
    board_path.write_text(json.dumps(ordered[:80], indent=2), encoding="utf-8")
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--stop-hour", type=int, default=6)
    parser.add_argument("--max-runs", type=int, default=0, help="0 = until stop time")
    args = parser.parse_args()

    stop = _stop_at(args.stop_hour)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    board_path = OUT_DIR / "overnight_leaderboard.json"
    print(
        f"WNBA hybrid overnight search\n"
        f"  baseline LL={BASELINE_LL}\n"
        f"  now={_now().isoformat()}\n"
        f"  stop={stop.isoformat()}\n",
        flush=True,
    )

    print("Preparing frame (curves + market)...", flush=True)
    frame = prepare_frame(end_season=args.end_season)
    print(f"frame ready: {len(frame)} rows", flush=True)

    queue = default_search_space()
    results = _load_all_summaries()
    known = {(r.get("config") or {}).get("name") for r in results}

    run_idx = 0
    mut_round = 0
    best_ll = min((r.get("oos_model_logloss", 9) for r in results), default=9.0)
    print(f"recovered {len(results)} prior runs; best_ll={best_ll}", flush=True)
    _persist_board(results, board_path)

    while _now() < stop:
        if args.max_runs and run_idx >= args.max_runs:
            break
        if queue:
            cfg = queue.pop(0)
        else:
            parent = HybridConfig(name="seed")
            if results:
                top = min(results, key=lambda r: r.get("oos_model_logloss", 9))
                parent = HybridConfig(**{
                    k: v for k, v in top.get("config", {}).items()
                    if k in HybridConfig.__dataclass_fields__
                })
            cfg = _mutate(parent, mut_round)
            mut_round += 1

        if cfg.name in known:
            continue

        print(f"\n=== run {run_idx + 1}: {cfg.name} @ {_now().strftime('%H:%M:%S')} ===", flush=True)
        try:
            summary = run_config(cfg, frame, args.end_season)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {cfg.name}: {exc}", flush=True)
            summary = {
                "config": cfg.to_dict(),
                "error": str(exc),
                "oos_model_logloss": 9.0,
                "beat_baseline": False,
            }
        results.append(summary)
        known.add(cfg.name)
        results = _persist_board(results, board_path)
        best_ll = results[0].get("oos_model_logloss", best_ll)
        print(
            f"LEADERBOARD top3: "
            + ", ".join(
                f"{r.get('config', {}).get('name', '?')}={r.get('oos_model_logloss')}"
                for r in results[:3]
            ),
            flush=True,
        )
        if summary.get("beat_baseline") and summary.get("config"):
            winner = HybridConfig(**{
                k: v for k, v in summary["config"].items()
                if k in HybridConfig.__dataclass_fields__
            })
            for j in range(3):
                queue.append(_mutate(winner, mut_round + j + 100))
            mut_round += 3
        run_idx += 1
        time.sleep(1)

    print(
        f"\nStopped at {_now().isoformat()}. runs={run_idx} best_ll={best_ll} "
        f"baseline={BASELINE_LL} delta={BASELINE_LL - best_ll:.5f}",
        flush=True,
    )
    (OUT_DIR / "overnight_final.json").write_text(
        json.dumps(
            {
                "stopped_at": _now().isoformat(),
                "runs": run_idx,
                "baseline_ll": BASELINE_LL,
                "best": results[0] if results else None,
                "top5": results[:5],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
