"""Backtest the soccer v2 1X2 bet policy on walk-forward OOS predictions.

Fair reference = devigged Pinnacle/first-available OPENING line. A bet
qualifies when the calibrated model probability beats the reference by a
gap with positive EV at the EXECUTION price; quarter-Kelly staking.

Execution prices tested: "open" (Pinnacle open — sharpest, worst case),
"max" (best price across ~25 books at open era — line shopping), "b365"
(single soft book). Closing odds grade CLV.

Writes the best policy to data/models/soccer_v2/bet_policy.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "soccer_v2"

KELLY_FRACTION = 0.25
MAX_STAKE_UNITS = 3.0
COL = {"home": 0, "draw": 1, "away": 2}


def devig(oh: float, od: float, oa: float) -> tuple[float, float, float] | None:
    if any(pd.isna(o) or o is None or o <= 1.0 for o in (oh, od, oa)):
        return None
    inv = (1.0 / oh, 1.0 / od, 1.0 / oa)
    total = sum(inv)
    return inv[0] / total, inv[1] / total, inv[2] / total


def kelly_units(p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    edge = p * (b + 1.0) - 1.0
    if edge <= 0:
        return 0.0
    return float(min(edge / b * KELLY_FRACTION * 10.0, MAX_STAKE_UNITS))


def _exec_odds(row, exec_price: str) -> tuple:
    if exec_price == "max":
        odds = (row.max_home, row.max_draw, row.max_away)
        if not any(pd.isna(o) or o is None or o <= 1.0 for o in odds):
            return odds
        return (row.open_home, row.open_draw, row.open_away)
    if exec_price == "b365":
        odds = (row.b365_home, row.b365_draw, row.b365_away)
        if not any(pd.isna(o) or o is None or o <= 1.0 for o in odds):
            return odds
        return (row.open_home, row.open_draw, row.open_away)
    if exec_price == "close":
        return (row.close_home, row.close_draw, row.close_away)
    return (row.open_home, row.open_draw, row.open_away)


def simulate(
    df: pd.DataFrame,
    *,
    prob_prefix: str,
    min_gap: float,
    min_ev: float,
    max_dec: float,
    outcomes: tuple[str, ...],
    exec_price: str = "open",
) -> dict:
    total_staked = total_profit = 0.0
    bets = wins = 0
    staked_by: dict[int, float] = defaultdict(float)
    profit_by: dict[int, float] = defaultdict(float)
    count_by: dict[int, int] = defaultdict(int)
    clv_list: list[float] = []

    for row in df.itertuples():
        reference = devig(row.open_home, row.open_draw, row.open_away)
        if reference is None:
            continue
        odds = _exec_odds(row, exec_price)
        if any(pd.isna(o) or o is None or o <= 1.0 for o in odds):
            continue
        probs = (
            getattr(row, f"{prob_prefix}_home"),
            getattr(row, f"{prob_prefix}_draw"),
            getattr(row, f"{prob_prefix}_away"),
        )
        label = int(row.label)
        season = int(row.season)

        for outcome in outcomes:
            i = COL[outcome]
            dec = float(odds[i])
            if dec > max_dec:
                continue
            p = float(probs[i])
            if (p - reference[i]) * 100.0 < min_gap:
                continue
            if (p * dec - 1.0) * 100.0 < min_ev:
                continue
            stake = kelly_units(p, dec)
            if stake <= 0:
                continue
            bets += 1
            total_staked += stake
            count_by[season] += 1
            staked_by[season] += stake
            delta = stake * (dec - 1.0) if label == i else -stake
            total_profit += delta
            profit_by[season] += delta
            if label == i:
                wins += 1
            close_dec = (row.close_home, row.close_draw, row.close_away)[i]
            if close_dec is not None and not pd.isna(close_dec) and close_dec > 1.0:
                clv_list.append((dec / float(close_dec) - 1.0) * 100.0)

    if bets == 0 or total_staked <= 0:
        return {"bets": 0}
    per_season = {}
    rois = []
    for season in sorted(staked_by):
        roi = profit_by[season] / staked_by[season] * 100.0
        rois.append(roi)
        per_season[str(season)] = {
            "bets": count_by[season],
            "roi_pct": round(roi, 2),
            "profit_units": round(profit_by[season], 1),
        }
    return {
        "bets": bets,
        "staked_units": round(total_staked, 1),
        "profit_units": round(total_profit, 1),
        "roi_pct": round(total_profit / total_staked * 100.0, 2),
        "win_rate": round(wins / bets, 4),
        "avg_clv_pct": round(float(np.mean(clv_list)), 3) if clv_list else None,
        "clv_positive_rate": round(float(np.mean([c > 0 for c in clv_list])), 4)
        if clv_list
        else None,
        "per_season": per_season,
        "seasons_positive": sum(1 for r in rois if r > 0),
        "seasons_total": len(rois),
        "worst_season_roi": round(min(rois), 2) if rois else None,
        "median_season_roi": round(float(np.median(rois)), 2) if rois else None,
    }


def main() -> None:
    df = pd.read_csv(MODEL_DIR / "oos_predictions.csv")
    recent = df[df["season"] >= 2015].reset_index(drop=True)
    print(f"OOS rows: {len(df)} (recent era: {len(recent)})", flush=True)

    outcome_sets: list[tuple[str, ...]] = [
        ("home", "draw", "away"),
        ("home", "away"),
        ("away",),
        ("home",),
        ("draw",),
    ]
    grids: list[dict] = []
    for exec_price in ("max", "open"):
        for prob_prefix in ("calm", "cal"):
            for min_gap in (2.0, 3.0, 4.0, 5.0, 6.0):
                for min_ev in (0.0, 2.0, 5.0):
                    for max_dec in (4.0, 6.0, 11.0):
                        for outcomes in outcome_sets:
                            grids.append(
                                {
                                    "exec_price": exec_price,
                                    "prob_prefix": prob_prefix,
                                    "min_gap": min_gap,
                                    "min_ev": min_ev,
                                    "max_dec": max_dec,
                                    "outcomes": outcomes,
                                }
                            )

    results = []
    for params in grids:
        res = simulate(recent, **params)
        if res.get("bets", 0) < 1200:
            continue
        # prioritize season consistency, then ROI, then drawdown control
        score = (
            3.0 * res["roi_pct"]
            + 0.5 * res["worst_season_roi"]
            + 12.0 * res["seasons_positive"] / max(res["seasons_total"], 1)
            + (res["avg_clv_pct"] or 0.0)
        )
        results.append((score, params, res))

    results.sort(key=lambda item: -item[0])
    print("\nTOP POLICIES (ranked by worst-season + overall ROI):")
    for score, params, res in results[:15]:
        print(
            f"  [{params['exec_price']:>4}] {params['prob_prefix']:>4} "
            f"gap>={params['min_gap']} ev>={params['min_ev']} dec<={params['max_dec']} "
            f"outcomes={'/'.join(params['outcomes'])}: bets={res['bets']} "
            f"roi={res['roi_pct']}% worst={res['worst_season_roi']}% "
            f"pos={res['seasons_positive']}/{res['seasons_total']} "
            f"clv={res['avg_clv_pct']}% clv+={res['clv_positive_rate']}",
            flush=True,
        )

    if not results:
        print("no qualifying policy")
        return

    _score, best_params, best_res = results[0]
    sharp_check = simulate(recent, **{**best_params, "exec_price": "open"})

    policy = {
        "bet_type": "soccer_1x2",
        "stake_mode": "kelly",
        "kelly_fraction": KELLY_FRACTION,
        "model_variant": "market_aware"
        if best_params["prob_prefix"] == "calm"
        else "pure",
        "exec_price": best_params["exec_price"],
        "min_gap_pp": best_params["min_gap"],
        "min_ev_pct": best_params["min_ev"],
        "max_decimal_odds": best_params["max_dec"],
        "outcomes": list(best_params["outcomes"]),
        "backtest": best_res,
        "backtest_sharp_open": {
            k: sharp_check.get(k)
            for k in (
                "bets",
                "roi_pct",
                "worst_season_roi",
                "seasons_positive",
                "seasons_total",
                "avg_clv_pct",
                "clv_positive_rate",
            )
        },
    }
    (MODEL_DIR / "bet_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    print(f"\nbest policy saved -> {MODEL_DIR / 'bet_policy.json'}")
    print(json.dumps({k: v for k, v in policy.items() if k != "backtest"}, indent=2))
    print(
        "best backtest summary:",
        json.dumps({k: v for k, v in best_res.items() if k != "per_season"}, indent=2),
    )
    print("per-season:", json.dumps(best_res.get("per_season"), indent=1))


if __name__ == "__main__":
    main()
