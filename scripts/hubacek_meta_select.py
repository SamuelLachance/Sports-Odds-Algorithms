"""Season-by-season max-Sharpe meta-selection over Hubáček β configs (§7.3.1).

For each league: for every evaluated season s, pick the (config, φ, price)
setting with the best Sharpe on all PRIOR seasons' rounds, then realize that
setting's profit on season s. The aggregate is the paper's "actual betting"
protocol — the number a bettor could really have achieved, with the setting
choice itself out-of-sample.

Writes data/models/{league}_hybrid/hubacek_meta.json.

Usage:
  python scripts/hubacek_meta_select.py --leagues nba
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hubacek_v2.eval import evaluate_rounds  # noqa: E402
from web.hybrid_v2.train import hubacek_round_key  # noqa: E402

PHIS = (0.0, 0.1, 0.15, 0.2)


def _load_oos(league: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in glob.glob(str(PROJECT_ROOT / "data" / "models" / f"{league}_hybrid" / "oos_cb_dec*.csv")):
        name = Path(path).stem.replace("oos_", "")
        frame = pd.read_csv(path, low_memory=False)
        frame["hub_round"] = hubacek_round_key(league, frame)
        out[name] = frame
    return out


def meta_select(league: str) -> dict | None:
    oos_by_cfg = _load_oos(league)
    if not oos_by_cfg:
        print(f"[{league}] no decorrelated OOS files — run train_hubacek_arms first")
        return None
    seasons = sorted(
        {int(s) for frame in oos_by_cfg.values() for s in frame["season"].dropna().unique()}
    )
    picks: list[dict] = []
    realized: list[dict] = []
    for i, season in enumerate(seasons):
        prior = seasons[:i]
        if not prior:
            continue
        best = None
        for cfg_name, frame in oos_by_cfg.items():
            hist = frame[frame["season"].isin(prior)]
            if hist.empty:
                continue
            for phi in PHIS:
                res = evaluate_rounds(hist, price="close", phi=phi, strategy="opt", round_key="hub_round")
                if not res or res.get("sharpe_per_round") is None:
                    continue
                if best is None or res["sharpe_per_round"] > best["sharpe"]:
                    best = {"config": cfg_name, "phi": phi, "sharpe": res["sharpe_per_round"]}
        if best is None:
            continue
        cur = oos_by_cfg[best["config"]]
        season_frame = cur[cur["season"] == season]
        res = evaluate_rounds(season_frame, price="close", phi=best["phi"], strategy="opt", round_key="hub_round")
        picks.append({"season": int(season), **best})
        if res:
            realized.append(
                {
                    "season": int(season),
                    "config": best["config"],
                    "phi": best["phi"],
                    "mean_profit_per_round_pct": res["mean_profit_per_round_pct"],
                    "sharpe_per_round": res["sharpe_per_round"],
                    "n_rounds": res["n_rounds"],
                    "n_bets": res["n_bets"],
                    "rho_model_book": res["rho_model_book"],
                    "cumulative_profit_units": res["cumulative_profit_units"],
                }
            )
    if not realized:
        print(f"[{league}] not enough seasons for meta-selection")
        return None
    profits = [r["mean_profit_per_round_pct"] for r in realized if r["mean_profit_per_round_pct"] is not None]
    payload = {
        "league": league,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "per-season max-Sharpe selection on prior rounds (Hubacek 2019 §7.3.1), realized next season at close prices",
        "seasons": realized,
        "mean_profit_per_round_pct": round(float(np.mean(profits)), 3) if profits else None,
        "worst_season_profit_pct": round(float(np.min(profits)), 3) if profits else None,
        "positive_seasons": int(sum(1 for p in profits if p > 0)),
        "total_seasons": len(profits),
    }
    out = PROJECT_ROOT / "data" / "models" / f"{league}_hybrid" / "hubacek_meta.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"[{league}] meta-selected: mean {payload['mean_profit_per_round_pct']}%/round, "
        f"worst season {payload['worst_season_profit_pct']}%, "
        f"{payload['positive_seasons']}/{payload['total_seasons']} seasons positive -> {out.name}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leagues", default="nba")
    args = parser.parse_args()
    for league in [s.strip().lower() for s in args.leagues.split(",") if s.strip()]:
        try:
            meta_select(league)
        except Exception as exc:  # noqa: BLE001
            print(f"[{league}] FAILED: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
