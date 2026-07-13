"""Pick a robust Hubáček NFL niche (volume + EV floor + all-positive)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from search_nfl_hubacek import V2, build_candidates, load_oos, search_grid

PROJECT = Path(__file__).resolve().parent.parent


def main() -> None:
    hybrid = load_oos(V2 / "oos_predictions_hybrid.csv", "hybrid")
    positives: list[tuple] = []
    for wname, lo in (("from2022", 2022), ("from2023", 2023), ("from2020", 2020), ("from2018", 2018)):
        frame = hybrid[hybrid.season >= lo]
        for prob_col in ("model_raw", "model_prob"):
            for decor_w in (0.08, 0.12, 0.18, 0.25, 0.35, 0.50):
                data = build_candidates(frame, prob_col, decor_w)
                if data is None:
                    continue
                hits = search_grid(
                    data,
                    source="hybrid",
                    prob_col=prob_col,
                    decor_w=decor_w,
                    window=wname,
                    min_bets=40,
                    min_seasons=3,
                    require_all_positive=True,
                )
                for params, res in hits:
                    if res["worst_season_roi"] <= 0:
                        continue
                    # Live Hubáček uses DEFAULT_BINARY_DECORRELATION_WEIGHT=0.12
                    live_match = 1 if abs(float(params["decor_w"]) - 0.12) < 1e-9 else 0
                    side_bonus = 1 if params["sides"] == "favorite" else 0
                    raw_bonus = 1 if params["prob_col"] == "model_raw" else 0
                    # Prefer live weight + sample size (cap 80) + worst-season floor
                    score = (
                        live_match,
                        min(int(res["bets"]), 80),
                        params["min_edge_pp"],  # prefer real Hubáček disagreement
                        res["worst_season_roi"],
                        params["min_ev_pct"],
                        raw_bonus,
                        side_bonus,
                        res["roi_pct"],
                    )
                    positives.append((score, params, res))

    positives.sort(key=lambda x: x[0], reverse=True)
    print(f"robust positives: {len(positives)}")
    for score, params, res in positives[:20]:
        print(
            f"  bets={res['bets']:4d} worst={res['worst_season_roi']:6.2f} "
            f"roi={res['roi_pct']:6.2f} {params['season_window']} {params['prob_col']} "
            f"decor={params['decor_w']} gap>={params['min_edge_pp']} "
            f"ev>={params['min_ev_pct']} ml[{params['ml_lo']},{params['ml_hi']}] {params['sides']}"
        )
    if not positives:
        raise SystemExit("no robust niche")

    _, best_params, best_res = positives[0]
    print("\nCHOSEN ROBUST")
    print(json.dumps(best_params, indent=2))
    print(json.dumps({k: v for k, v in best_res.items() if k != "per_season"}, indent=2))
    print("per_season", json.dumps(best_res["per_season"], indent=2))

    policy = {
        "stake_mode": "kelly",
        "kelly_fraction": 0.25,
        "enabled": True,
        "enabled_note": "positive worst-season ROI (Hubáček decorrelated, volume-preferred)",
        "source": "search_nfl_hubacek.py + pick_robust_nfl_hubacek.py",
        "hubacek_decorrelated": True,
        **best_params,
        "backtest": {k: v for k, v in best_res.items() if k != "per_season"},
        "backtest_per_season": best_res["per_season"],
    }
    path = V2 / "bet_policy.json"
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"wrote {path}")

    pick_path = PROJECT / "data" / "pick_strategy.json"
    pick = json.loads(pick_path.read_text(encoding="utf-8"))
    entry = pick.setdefault("nfl", {})
    entry["bet_type"] = "moneyline"
    entry["enabled"] = True
    entry["min_market_gap_pp"] = float(best_params["min_edge_pp"])
    entry["min_ev_pct"] = float(best_params["min_ev_pct"])
    entry["ml_lo"] = int(best_params["ml_lo"])
    entry["ml_hi"] = int(best_params["ml_hi"])
    entry["min_win_confidence_pp"] = 0.0
    sides = best_params["sides"]
    if sides in ("home", "away"):
        entry["allowed_sides"] = [sides]
        entry["fav_mode"] = "any"
    elif sides == "favorite":
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "favorite"
    elif sides == "dog":
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "dog"
    else:
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "any"
    entry["backtest_roi_pct"] = best_res["roi_pct"]
    entry["backtest_bets"] = best_res["bets"]
    entry["note"] = (
        f"Hubáček NFL enable {date.today().isoformat()}: hybrid {best_params['prob_col']} "
        f"decor_w={best_params['decor_w']} gap>={best_params['min_edge_pp']} "
        f"EV>={best_params['min_ev_pct']}% ML[{best_params['ml_lo']},{best_params['ml_hi']}] "
        f"sides={sides} window={best_params['season_window']} "
        f"ROI {best_res['roi_pct']}% worst {best_res['worst_season_roi']}% "
        f"n={best_res['bets']} all-season+. ENABLED."
    )
    pick["generated_at"] = date.today().isoformat()
    pick_path.write_text(json.dumps(pick, indent=2), encoding="utf-8")
    print(f"updated {pick_path}")

    summary = {
        "selection": "volume_ev_all_positive",
        "n_robust": len(positives),
        "best": {"params": best_params, "backtest": best_res},
        "top10": [
            {"params": p, "bets": r["bets"], "worst": r["worst_season_roi"], "roi": r["roi_pct"]}
            for _, p, r in positives[:10]
        ],
    }
    out = V2 / "hubacek_robust_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
