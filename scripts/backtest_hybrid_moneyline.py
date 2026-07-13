"""Moneyline Hubáček grid for leagues whose official book is ML (NFL/CBB)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KELLY_FRACTION = 0.25


def american_to_decimal(ml: float) -> float:
    if ml == 0:
        ml = 100.0
    if abs(ml) < 100:
        return float("nan")
    return 1.0 + (ml / 100.0 if ml > 0 else 100.0 / abs(ml))


def devig(home_ml: float, away_ml: float) -> tuple[float, float]:
    ph = 1.0 / american_to_decimal(home_ml)
    pa = 1.0 / american_to_decimal(away_ml)
    total = ph + pa
    return ph / total, pa / total


def kelly_units(prob: float, ml: float) -> float:
    dec = american_to_decimal(ml)
    b = dec - 1.0
    edge = prob * dec - 1.0
    if edge <= 0 or b <= 0 or not math.isfinite(dec):
        return 0.0
    return float(min(max(edge / b * KELLY_FRACTION * 100.0, 0.25), 3.0))


def simulate(frame: pd.DataFrame, *, min_edge_pp: float, min_ev_pct: float, ml_lo: int, ml_hi: int, sides: str, home_col: str, away_col: str) -> dict:
    picks = []
    for row in frame.itertuples(index=False):
        hml = getattr(row, home_col)
        aml = getattr(row, away_col)
        if pd.isna(hml) or pd.isna(aml):
            continue
        hml, aml = float(hml), float(aml)
        if abs(hml) < 100 or abs(aml) < 100:
            continue
        mkt_h, mkt_a = devig(hml, aml)
        p = float(row.model_prob)
        cands = []
        if p - mkt_h >= min_edge_pp / 100.0:
            cands.append(("home", p, hml, float(row.home_win) > 0.5))
        if (1.0 - p) - mkt_a >= min_edge_pp / 100.0:
            cands.append(("away", 1.0 - p, aml, float(row.home_win) < 0.5))
        for side, prob, ml, won in cands:
            if sides == "home" and side != "home":
                continue
            if sides == "away" and side != "away":
                continue
            if sides == "favorite" and ml > 0:
                continue
            if sides == "dog" and ml < 0:
                continue
            if not (ml_lo <= ml <= ml_hi):
                continue
            dec = american_to_decimal(ml)
            ev = (prob * dec - 1.0) * 100.0
            if ev < min_ev_pct:
                continue
            units = kelly_units(prob, ml)
            picks.append(
                {
                    "season": int(row.season),
                    "won": won,
                    "pnl": units * (dec - 1.0) if won else -units,
                    "units": units,
                }
            )
    if not picks:
        return {"bets": 0}
    res = pd.DataFrame(picks)
    staked = float(res.units.sum())
    profit = float(res.pnl.sum())
    by = res.groupby("season").agg(bets=("pnl", "size"), profit=("pnl", "sum"), staked=("units", "sum"))
    by["roi"] = 100.0 * by.profit / by.staked.replace(0, np.nan)
    return {
        "bets": int(len(res)),
        "staked_units": round(staked, 1),
        "profit_units": round(profit, 1),
        "roi_pct": round(100.0 * profit / staked, 2) if staked else 0.0,
        "win_rate": round(float(res.won.mean()), 4),
        "seasons_positive": int((by.roi > 0).sum()),
        "seasons_total": int(len(by)),
        "worst_season_roi": round(float(by.roi.min()), 2),
        "median_season_roi": round(float(by.roi.median()), 2),
        "per_season": {
            str(int(idx)): {
                "bets": int(r.bets),
                "roi_pct": round(float(r.roi), 2),
                "profit_units": round(float(r.profit), 1),
            }
            for idx, r in by.iterrows()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True, choices=["nfl", "cbb", "nba", "wnba", "cfb"])
    parser.add_argument("--write-policy", action="store_true")
    parser.add_argument("--min-bets", type=int, default=40)
    parser.add_argument("--min-seasons", type=int, default=2)
    parser.add_argument("--min-season", type=int, default=None, help="Inclusive OOS season floor")
    parser.add_argument("--max-season", type=int, default=None, help="Inclusive OOS season ceiling")
    args = parser.parse_args()

    v2 = PROJECT_ROOT / "data" / "models" / f"{args.league}_v2"
    frame = pd.read_csv(v2 / "oos_predictions.csv")
    if "model_prob" not in frame.columns:
        raise SystemExit("missing model_prob")
    # prefer close ML, fall back — price loop below selects concrete columns
    if "home_close_ml" not in frame.columns and "home_ml" not in frame.columns:
        raise SystemExit("missing home_close_ml/home_ml")
    frame = frame.dropna(subset=["model_prob", "home_win", "season"])
    if args.min_season is not None:
        frame = frame[frame["season"] >= args.min_season]
    if args.max_season is not None:
        frame = frame[frame["season"] <= args.max_season]

    price_cols: list[tuple[str, str, str]] = []
    if "home_close_ml" in frame.columns and "away_close_ml" in frame.columns:
        price_cols.append(("close", "home_close_ml", "away_close_ml"))
    elif "home_ml" in frame.columns and "away_ml" in frame.columns:
        price_cols.append(("close", "home_ml", "away_ml"))
    if "home_open_ml" in frame.columns and "away_open_ml" in frame.columns:
        open_n = int(frame[["home_open_ml", "away_open_ml"]].dropna().shape[0])
        if open_n >= args.min_bets:
            price_cols.append(("open", "home_open_ml", "away_open_ml"))
    if not price_cols:
        raise SystemExit("missing moneyline columns")

    results = []
    for exec_price, home_col, away_col in price_cols:
        priced = frame.dropna(subset=[home_col, away_col])
        print(f"{args.league}: OOS n={len(priced)} using {home_col}/{away_col} ({exec_price})")
        for min_edge in (2.0, 3.0, 4.0, 5.0, 5.6, 6.0, 7.0, 8.0):
            for min_ev in (0.0, 2.0, 4.0, 5.0):
                for ml_lo, ml_hi in ((-350, 300), (-200, 200), (-150, 200), (-150, 250)):
                    for sides in ("either", "home", "away", "favorite"):
                        params = {
                            "bet_type": "moneyline",
                            "exec_price": exec_price,
                            "prob_col": "model_prob",
                            "min_edge_pp": min_edge,
                            "min_ev_pct": min_ev,
                            "ml_lo": ml_lo,
                            "ml_hi": ml_hi,
                            "sides": sides,
                        }
                        res = simulate(
                            priced,
                            min_edge_pp=min_edge,
                            min_ev_pct=min_ev,
                            ml_lo=ml_lo,
                            ml_hi=ml_hi,
                            sides=sides,
                            home_col=home_col,
                            away_col=away_col,
                        )
                        if res.get("bets", 0) < args.min_bets:
                            continue
                        if res.get("seasons_total", 0) < args.min_seasons:
                            continue
                        score = (res["worst_season_roi"], res["roi_pct"])
                        results.append((score, params, res))

    results.sort(key=lambda x: x[0], reverse=True)
    print("top ML policies:")
    for score, params, res in results[:12]:
        print(
            f"  worst={score[0]:6.2f} roi={score[1]:6.2f} "
            f"edge>={params['min_edge_pp']} ev>={params['min_ev_pct']} "
            f"ml[{params['ml_lo']},{params['ml_hi']}] sides={params['sides']}: "
            f"bets={res['bets']} pos={res['seasons_positive']}/{res['seasons_total']}"
        )
    if not results:
        print("no qualifying policies")
        return 1
    best_score, best_params, best_res = results[0]
    enabled = best_res["worst_season_roi"] > 0
    print("\nchosen:", json.dumps(best_params, indent=1))
    print(json.dumps({k: v for k, v in best_res.items() if k != "per_season"}, indent=1))
    if args.write_policy:
        if not enabled:
            print("--write-policy requested but worst-season ROI <= 0; not overwriting with enabled policy")
            # still write disabled snapshot for transparency
        policy = {
            "stake_mode": "kelly",
            "kelly_fraction": KELLY_FRACTION,
            "enabled": enabled,
            "enabled_note": (
                "positive worst-season ROI"
                if enabled
                else f"worst_season_roi={best_res['worst_season_roi']} -> KEEP DISABLED"
            ),
            **best_params,
            "backtest": {k: v for k, v in best_res.items() if k != "per_season"},
            "backtest_per_season": best_res["per_season"],
        }
        path = v2 / "bet_policy.json"
        path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        print(f"wrote {path} enabled={enabled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
