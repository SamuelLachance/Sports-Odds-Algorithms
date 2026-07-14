"""Pick a robust NHL Hubáček niche on expanded hybrid OOS and write policy."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from hubacek_binary_search import (  # noqa: E402
    build_ml_candidates,
    summarize,
    update_pick_strategy_ml,
)

V2 = PROJECT / "data" / "models" / "nhl_v2"


def eval_pol(frame, decor, gap, ev, lo, hi, sides, window, min_bets=80):
    ff = frame if window is None else frame[frame.season >= window]
    data = build_ml_candidates(
        ff, "model_raw", decor, home_ml_col="home_open_ml", away_ml_col="away_open_ml"
    )
    if data is None:
        return None
    ml = data["ml"]
    mask = (data["edge"] >= gap) & (data["ev"] >= ev) & (ml >= lo) & (ml <= hi)
    if sides == "favorite":
        mask = mask & data["fav"]
    elif sides == "away":
        mask = mask & (data["side"] == 1)
    elif sides == "home":
        mask = mask & (data["side"] == 0)
    elif sides == "dog":
        mask = mask & data["dog"]
    return summarize(
        data["season"],
        data["units"],
        data["pnl"],
        data["won"],
        mask,
        min_bets=min_bets,
        min_seasons=3,
        require_all_positive=True,
    )


def main() -> int:
    frame = pd.read_csv(V2 / "oos_predictions_hybrid.csv")
    print("open%", float(frame["home_open_ml"].notna().mean()), flush=True)

    old = eval_pol(frame, 0.12, 5, 4, -250, 250, "either", 2023)
    print("OLD gap5 ev4 either from2023", old, flush=True)

    cands = []
    for gap in (4, 5, 6, 8, 10):
        for ev in (0, 2, 4, 5, 6):
            for lo, hi in ((-250, 250), (-200, 200), (-150, 150), (-110, 110)):
                for sides in ("either", "home", "away", "favorite", "dog"):
                    for decor in (0.08, 0.12, 0.25, 0.35):
                        for win in (2023, 2022, None):
                            res = eval_pol(frame, decor, gap, ev, lo, hi, sides, win)
                            if not res:
                                continue
                            min_s = min(v["bets"] for v in res["per_season"].values())
                            if min_s < 20 or res["bets"] < 150:
                                continue
                            cands.append(
                                (
                                    res["worst_season_roi"],
                                    res["roi_pct"],
                                    res["bets"],
                                    min_s,
                                    decor,
                                    gap,
                                    ev,
                                    lo,
                                    hi,
                                    sides,
                                    win,
                                    res,
                                )
                            )
    cands.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    print("TOP robust n>=150 minS>=20:", flush=True)
    for c in cands[:15]:
        print(
            f"worst={c[0]:6.2f} roi={c[1]:6.2f} n={c[2]:4d} minS={c[3]:3d} "
            f"decor={c[4]} gap={c[5]} ev={c[6]} ML[{c[7]},{c[8]}] sides={c[9]} win={c[10]}",
            flush=True,
        )
    if not cands:
        print("ERROR: no robust niche", flush=True)
        return 1

    # Prefer LIVE decor 0.12 among strong floors when available.
    live = [c for c in cands if c[4] == 0.12]
    best = live[0] if live and live[0][0] >= 3.0 else cands[0]
    decor, gap, evmin, lo, hi, sides, win, res = (
        best[4],
        best[5],
        best[6],
        best[7],
        best[8],
        best[9],
        best[10],
        best[11],
    )
    params = {
        "bet_type": "moneyline",
        "exec_price": "open",
        "source_oos": "hybrid",
        "prob_col": "model_raw",
        "decor_w": float(decor),
        "min_edge_pp": float(gap),
        "min_ev_pct": float(evmin),
        "ml_lo": int(lo),
        "ml_hi": int(hi),
        "sides": sides,
        "season_window": "all" if win is None else f"from{win}",
        "hubacek": True,
    }
    policy = {
        "stake_mode": "kelly",
        "kelly_fraction": 0.25,
        "enabled": True,
        "enabled_note": "positive worst-season ROI (Hubáček expanded hybrid; robust volume floor)",
        "source": "hubacek_binary_search",
        "hubacek_decorrelated": True,
        **params,
        "backtest": {k: res[k] for k in res if k != "per_season"},
        "backtest_per_season": res["per_season"],
    }
    (V2 / "bet_policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    update_pick_strategy_ml(
        PROJECT / "data" / "pick_strategy.json",
        "nhl",
        params,
        res,
        enabled=True,
    )
    pick_path = PROJECT / "data" / "pick_strategy.json"
    pick = json.loads(pick_path.read_text(encoding="utf-8"))
    pick["nhl"]["note"] = (
        f"Hubáček NHL {date.today().isoformat()}: expanded-features hybrid model_raw "
        f"decor_w={decor} gap>={gap} EV>={evmin}% ML[{lo},{hi}] sides={sides} "
        f"exec=open window={params['season_window']} ROI {res['roi_pct']}% "
        f"worst {res['worst_season_roi']}% n={res['bets']}. ENABLED."
    )
    pick["nhl"]["backtest_roi_pct"] = res["roi_pct"]
    pick["nhl"]["backtest_bets"] = res["bets"]
    pick_path.write_text(json.dumps(pick, indent=2) + "\n", encoding="utf-8")
    print("wrote bet_policy + pick_strategy", flush=True)
    print(json.dumps(policy, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
