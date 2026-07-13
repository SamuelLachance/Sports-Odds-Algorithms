"""Verify and write the chosen Hubáček NFL niche (decor=0.12, gap>=2, fav, [-180,180])."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from search_nfl_hubacek import V2, build_candidates, load_oos, search_grid, summarize

PROJECT = Path(__file__).resolve().parent.parent


def main() -> None:
    hybrid = load_oos(V2 / "oos_predictions_hybrid.csv", "hybrid")
    frame = hybrid[hybrid.season >= 2022]
    data = build_candidates(frame, "model_prob", 0.12)
    assert data is not None
    mask = (
        (data["edge"] >= 2.0)
        & (data["ev"] >= 0.0)
        & (data["ml"] >= -180)
        & (data["ml"] <= 180)
        & data["fav"]
    )
    res = summarize(
        data["season"],
        data["units"],
        data["pnl"],
        data["won"],
        mask,
        min_bets=40,
        min_seasons=3,
        require_all_positive=True,
    )
    assert res is not None and res["worst_season_roi"] > 0
    print(json.dumps(res, indent=2))

    params = {
        "bet_type": "moneyline",
        "exec_price": "close",
        "source_oos": "hybrid",
        "prob_col": "model_prob",
        "decor_w": 0.12,
        "min_edge_pp": 2.0,
        "min_ev_pct": 0.0,
        "ml_lo": -180,
        "ml_hi": 180,
        "sides": "favorite",
        "season_window": "from2022",
        "hubacek": True,
    }
    policy = {
        "stake_mode": "kelly",
        "kelly_fraction": 0.25,
        "enabled": True,
        "enabled_note": "positive worst-season ROI (Hubáček decorrelated, live weight 0.12)",
        "source": "search_nfl_hubacek.py",
        "hubacek_decorrelated": True,
        **params,
        "backtest": {k: v for k, v in res.items() if k != "per_season"},
        "backtest_per_season": res["per_season"],
    }
    (V2 / "bet_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    pick_path = PROJECT / "data" / "pick_strategy.json"
    pick = json.loads(pick_path.read_text(encoding="utf-8"))
    entry = pick.setdefault("nfl", {})
    entry.update(
        {
            "bet_type": "moneyline",
            "enabled": True,
            "min_market_gap_pp": 2.0,
            "min_ev_pct": 0.0,
            "ml_lo": -180,
            "ml_hi": 180,
            "min_win_confidence_pp": 0.0,
            "fav_mode": "favorite",
            "backtest_roi_pct": res["roi_pct"],
            "backtest_bets": res["bets"],
            "note": (
                f"Hubáček NFL enable {date.today().isoformat()}: hybrid model_prob "
                f"decor_w=0.12 gap>=2.0 EV>=0% ML[-180,180] sides=favorite "
                f"window=from2022 ROI {res['roi_pct']}% worst {res['worst_season_roi']}% "
                f"n={res['bets']} all-season+. ENABLED."
            ),
        }
    )
    entry.pop("allowed_sides", None)
    pick["generated_at"] = date.today().isoformat()
    pick_path.write_text(json.dumps(pick, indent=2), encoding="utf-8")
    print("wrote policy + pick_strategy")


if __name__ == "__main__":
    main()
