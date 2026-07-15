"""Train the Hubáček decorrelated (β) arms for one or more leagues.

Runs only the ``target_mode="decorrelated"`` configs from the focused grid
(c ∈ {0.2..0.8} × with/without odds features), writes their OOS/summaries as
usual, and assembles ``data/models/{league}_hybrid/hubacek_board.json`` ranked
by opt-strategy Sharpe at the close. Deliberately does NOT touch
best_config.json / focused_final.json — the shipped π (prediction) models stay
untouched; β model shipping is a separate, meta-selected step.

Usage:
  python scripts/train_hubacek_arms.py --leagues nba
  python scripts/train_hubacek_arms.py --leagues nba,wnba,nhl,mlb,nfl,cfb,cbb
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hybrid_v2.adapters import get_adapter  # noqa: E402
from web.hybrid_v2.config import focused_search_space  # noqa: E402
from web.hybrid_v2.train import prepare_frame, run_config  # noqa: E402


def run_league(league: str) -> dict:
    adapter = get_adapter(league)
    if adapter.multiclass:
        return {"league": league, "skipped": "multiclass (3-way extension pending)"}
    end_season = adapter.end_season
    frame = prepare_frame(adapter, end_season=end_season)
    print(f"=== {league} Hubacek beta arms (rows={len(frame)}) ===", flush=True)
    board: list[dict] = []
    for cfg in focused_search_space():
        if cfg.target_mode != "decorrelated":
            continue
        summary = run_config(adapter, cfg, frame, end_season)
        board.append(summary)

    def _sharpe(entry: dict) -> float:
        best = (entry.get("hubacek") or {}).get("best_opt") or {}
        val = best.get("sharpe_per_round")
        return val if isinstance(val, (int, float)) else -9e9

    board.sort(key=_sharpe, reverse=True)
    payload = {
        "league": league,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "board": board,
    }
    out = adapter.hybrid_dir / "hubacek_board.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    top = board[0] if board else {}
    best = (top.get("hubacek") or {}).get("best_opt") or {}
    print(
        f"[{league}] best beta: {top.get('config', {}).get('name')} "
        f"rho={best.get('rho_model_book')} profit/round={best.get('mean_profit_per_round_pct')}% "
        f"sharpe={best.get('sharpe_per_round')} phi={best.get('phi')} (wrote {out.name})",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leagues", default="nba")
    args = parser.parse_args()
    for league in [s.strip().lower() for s in args.leagues.split(",") if s.strip()]:
        try:
            run_league(league)
        except Exception as exc:  # noqa: BLE001 — one league must not kill the batch
            print(f"[{league}] FAILED: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
