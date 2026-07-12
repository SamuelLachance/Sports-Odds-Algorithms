"""Backtest NHL v2 moneyline bet policy on walk-forward OOS predictions.

Simulates betting vs the closing line (conservative floor) across an
edge-threshold grid, then reports per-season ROI, drawdown, and a robust
recommended policy. Also simulates at the opening line (live bets are placed
in between; CLV vs close is reported for open-line bets).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "models" / "nhl_v2"
OOS_PATH = OUT_DIR / "oos_predictions.csv"

KELLY_FRACTION = 0.25
KELLY_CAP_UNITS = 3.0
KELLY_MIN_UNITS = 0.25


def american_to_decimal(ml: float) -> float:
    """Decimal odds from American; EVEN 0 -> 2.0; invalid |ml|<100 -> NaN."""
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
    if edge <= 0 or b <= 0:
        return 0.0
    fraction = edge / b * KELLY_FRACTION
    units = fraction * 100.0  # 1 unit = 1% bankroll
    return float(min(max(units, KELLY_MIN_UNITS), KELLY_CAP_UNITS))


def simulate(
    frame: pd.DataFrame,
    *,
    min_edge_pp: float,
    min_ev_pct: float,
    ml_lo: float,
    ml_hi: float,
    stake_mode: str,
    price_cols: tuple[str, str] = ("home_close_ml", "away_close_ml"),
    clv_cols: tuple[str, str] | None = None,
) -> dict:
    home_col, away_col = price_cols
    picks: list[dict] = []
    for row in frame.itertuples(index=False):
        home_ml = getattr(row, home_col)
        away_ml = getattr(row, away_col)
        if pd.isna(home_ml) or pd.isna(away_ml):
            continue
        home_ml_f = 100.0 if float(home_ml) == 0 else float(home_ml)
        away_ml_f = 100.0 if float(away_ml) == 0 else float(away_ml)
        # American odds are only valid at |ml| >= 100 (median of mixed-sign
        # books can produce garbage near zero).
        if abs(home_ml_f) < 100 or abs(away_ml_f) < 100:
            continue
        market_home, market_away = devig(home_ml_f, away_ml_f)
        p_home = float(row.model_prob)
        close_home_ml = close_away_ml = None
        if clv_cols is not None:
            close_home_ml = getattr(row, clv_cols[0])
            close_away_ml = getattr(row, clv_cols[1])
        for side, prob, market, ml, close_ml in (
            ("home", p_home, market_home, home_ml_f, close_home_ml),
            ("away", 1.0 - p_home, market_away, away_ml_f, close_away_ml),
        ):
            if not (ml_lo <= ml <= ml_hi):
                continue
            edge_pp = (prob - market) * 100.0
            dec = american_to_decimal(ml)
            ev_pct = (prob * dec - 1.0) * 100.0
            if edge_pp < min_edge_pp or ev_pct < min_ev_pct:
                continue
            units = kelly_units(prob, ml) if stake_mode == "kelly" else 1.0
            if units <= 0:
                continue
            won = (row.home_win == 1) if side == "home" else (row.home_win == 0)
            pnl = units * (dec - 1.0) if won else -units
            clv_pct = None
            if close_ml is not None and not pd.isna(close_ml):
                close_dec = american_to_decimal(float(close_ml))
                if close_dec == close_dec:  # not NaN
                    clv_pct = (dec / close_dec - 1.0) * 100.0
            picks.append(
                {
                    "season": int(row.season),
                    "date": row.date,
                    "side": side,
                    "ml": ml,
                    "edge_pp": edge_pp,
                    "ev_pct": ev_pct,
                    "units": units,
                    "won": bool(won),
                    "pnl": pnl,
                    "clv_pct": clv_pct,
                }
            )
    if not picks:
        return {"bets": 0}
    bets = pd.DataFrame(picks)
    staked = bets.units.sum()
    profit = bets.pnl.sum()
    cumulative = bets.pnl.cumsum()
    drawdown = float((cumulative.cummax() - cumulative).max())
    per_season = {
        int(season): {
            "bets": int(len(grp)),
            "roi_pct": round(float(grp.pnl.sum() / grp.units.sum() * 100.0), 2),
            "profit_units": round(float(grp.pnl.sum()), 1),
        }
        for season, grp in bets.groupby("season")
    }
    season_rois = [v["roi_pct"] for v in per_season.values()]
    clv_values = bets.clv_pct.dropna()
    return {
        "bets": int(len(bets)),
        "staked_units": round(float(staked), 1),
        "profit_units": round(float(profit), 1),
        "roi_pct": round(float(profit / staked * 100.0), 2),
        "win_rate": round(float(bets.won.mean()), 4),
        "avg_ml": round(float(bets.ml.mean()), 0),
        "max_drawdown_units": round(drawdown, 1),
        "seasons_positive": int(sum(1 for r in season_rois if r > 0)),
        "seasons_total": len(season_rois),
        "worst_season_roi": round(min(season_rois), 2),
        "median_season_roi": round(float(np.median(season_rois)), 2),
        "avg_clv_pct": round(float(clv_values.mean()), 3) if len(clv_values) else None,
        "clv_positive_rate": round(float((clv_values > 0).mean()), 4) if len(clv_values) else None,
        "per_season": per_season,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest NHL v2 bet policy")
    parser.add_argument("--stake", choices=["flat", "kelly"], default="kelly")
    parser.add_argument("--write-policy", action="store_true")
    parser.add_argument("--min-bets-open", type=int, default=300)
    args = parser.parse_args()

    frame = pd.read_csv(OOS_PATH)
    frame = frame.dropna(subset=["model_prob", "home_close_ml", "away_close_ml"])
    open_frame = frame.dropna(subset=["home_open_ml", "away_open_ml"])
    print(
        f"OOS rows with closing odds: {len(frame)} ({frame.season.min()}-{frame.season.max()}); "
        f"with opening odds: {len(open_frame)} "
        f"({open_frame.season.min() if len(open_frame) else '-'}-"
        f"{open_frame.season.max() if len(open_frame) else '-'})\n"
    )

    grid_results: list[tuple[dict, dict, dict]] = []
    for min_edge in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0):
        for min_ev in (0.0, 2.0, 4.0):
            for ml_lo, ml_hi in ((-350, 300), (-250, 250), (-200, 200), (-160, 220)):
                params = {
                    "min_edge_pp": min_edge,
                    "min_ev_pct": min_ev,
                    "ml_lo": ml_lo,
                    "ml_hi": ml_hi,
                }
                open_result = simulate(
                    open_frame,
                    stake_mode=args.stake,
                    price_cols=("home_open_ml", "away_open_ml"),
                    clv_cols=("home_close_ml", "away_close_ml"),
                    **params,
                )
                if open_result.get("bets", 0) < args.min_bets_open:
                    continue
                close_result = simulate(frame, stake_mode=args.stake, **params)
                grid_results.append((params, open_result, close_result))

    grid_results.sort(
        key=lambda pr: (pr[1]["worst_season_roi"], pr[1]["roi_pct"]), reverse=True
    )
    print("top policies (ranked by worst-season ROI at the OPEN):")
    for params, open_result, close_result in grid_results[:12]:
        print(
            f"  edge>={params['min_edge_pp']:.1f}pp ev>={params['min_ev_pct']:.0f}% "
            f"ml[{params['ml_lo']},{params['ml_hi']}]: "
            f"open bets={open_result['bets']} roi={open_result['roi_pct']}% "
            f"worst={open_result['worst_season_roi']}% clv={open_result['avg_clv_pct']}% "
            f"| close roi={close_result['roi_pct']}% worst={close_result['worst_season_roi']}%"
        )

    if not grid_results:
        print("no qualifying policies")
        return 1

    best_params, best_open, best_close = grid_results[0]
    print("\nchosen policy at OPEN (live-like):")
    print(json.dumps(best_open, indent=1))
    print("\nsame policy at CLOSE (conservative floor):")
    print(json.dumps({k: v for k, v in best_close.items() if k != "per_season"}, indent=1))

    if args.write_policy:
        policy = {
            "bet_type": "moneyline",
            "stake_mode": args.stake,
            "kelly_fraction": KELLY_FRACTION,
            "min_edge_pp": best_params["min_edge_pp"],
            "min_ev_pct": best_params["min_ev_pct"],
            "ml_lo": best_params["ml_lo"],
            "ml_hi": best_params["ml_hi"],
            "backtest_close": {k: v for k, v in best_close.items() if k != "per_season"},
            "backtest_close_per_season": best_close["per_season"],
            "backtest_open": (
                {k: v for k, v in best_open.items() if k != "per_season"}
                if best_open
                else None
            ),
            "backtest_open_per_season": best_open.get("per_season") if best_open else None,
        }
        (OUT_DIR / "bet_policy.json").write_text(
            json.dumps(policy, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {OUT_DIR / 'bet_policy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
