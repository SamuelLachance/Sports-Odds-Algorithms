"""Backtest WNBA v2 bet policies on walk-forward OOS predictions.

Evaluates moneyline (calibrated win prob vs devigged consensus ML) and spread
(margin-model cover prob vs juice-implied break-even) policies for both the
pure and market-aware model heads across a threshold grid. Odds are ESPN
multi-book consensus medians (close-like). Ranks by season consistency and
worst-season ROI, prints the leaders, and can write the chosen policy to
data/models/wnba_v2/bet_policy.json.
"""

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

OUT_DIR = PROJECT_ROOT / "data" / "models" / "wnba_v2"
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


def _valid_spread_juice(raw: object) -> float | None:
    """American juice or None when missing/garbage. EVEN 0 → +100. Never invent −110."""
    if raw is None or pd.isna(raw):
        return None
    odds = float(raw)
    if odds == 0:
        return 100.0
    if abs(odds) < 100:
        return None
    return odds


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
    units = fraction * 100.0
    return float(min(max(units, KELLY_MIN_UNITS), KELLY_CAP_UNITS))


def _summarize(picks: list[dict]) -> dict:
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
    return {
        "bets": int(len(bets)),
        "staked_units": round(float(staked), 1),
        "profit_units": round(float(profit), 1),
        "roi_pct": round(float(profit / staked * 100.0), 2),
        "win_rate": round(float(bets.won.mean()), 4),
        "max_drawdown_units": round(drawdown, 1),
        "seasons_positive": int(sum(1 for r in season_rois if r > 0)),
        "seasons_total": len(season_rois),
        "worst_season_roi": round(min(season_rois), 2),
        "median_season_roi": round(float(np.median(season_rois)), 2),
        "per_season": per_season,
    }


def simulate_moneyline(
    frame: pd.DataFrame,
    *,
    prob_col: str,
    min_edge_pp: float,
    min_ev_pct: float,
    ml_lo: float,
    ml_hi: float,
    exec_price: str = "consensus",
) -> dict:
    """Select and execute vs consensus close, best book price, or the open."""
    picks: list[dict] = []
    for row in frame.itertuples(index=False):
        if exec_price == "open":
            home_ml = row.home_ml_open
            away_ml = row.away_ml_open
        else:
            home_ml = row.home_ml
            away_ml = row.away_ml
        if pd.isna(home_ml) or pd.isna(away_ml):
            continue
        if abs(float(home_ml)) < 100 or abs(float(away_ml)) < 100:
            continue
        market_home, market_away = devig(float(home_ml), float(away_ml))
        p_home = float(getattr(row, prob_col))
        exec_home = row.best_home_ml if exec_price == "best" else home_ml
        exec_away = row.best_away_ml if exec_price == "best" else away_ml
        if pd.isna(exec_home) or abs(float(exec_home)) < 100:
            exec_home = home_ml
        if pd.isna(exec_away) or abs(float(exec_away)) < 100:
            exec_away = away_ml
        for side, prob, market, ml, fill_ml in (
            ("home", p_home, market_home, float(home_ml), float(exec_home)),
            ("away", 1.0 - p_home, market_away, float(away_ml), float(exec_away)),
        ):
            if not (ml_lo <= ml <= ml_hi):
                continue
            edge_pp = (prob - market) * 100.0
            dec = american_to_decimal(fill_ml)
            ev_pct = (prob * dec - 1.0) * 100.0
            if edge_pp < min_edge_pp or ev_pct < min_ev_pct:
                continue
            units = kelly_units(prob, fill_ml)
            if units <= 0:
                continue
            won = (row.home_win == 1) if side == "home" else (row.home_win == 0)
            picks.append(
                {
                    "season": int(row.season),
                    "side": side,
                    "units": units,
                    "won": bool(won),
                    "pnl": units * (dec - 1.0) if won else -units,
                }
            )
    return _summarize(picks)


def simulate_spread(
    frame: pd.DataFrame,
    *,
    margin_col: str,
    sigma: float,
    min_cover_gap_pp: float,
    min_point_edge: float,
    exec_price: str = "consensus",
) -> dict:
    picks: list[dict] = []
    for row in frame.itertuples(index=False):
        spread = row.home_spread
        if pd.isna(spread):
            continue
        spread = float(spread)
        pred_margin = float(getattr(row, margin_col))
        actual_margin = float(row.margin)
        # Fail closed on missing/garbage juice — never invent −110 like live gates.
        home_odds = _valid_spread_juice(row.spread_home_odds)
        away_odds = _valid_spread_juice(row.spread_away_odds)
        if exec_price == "open":
            if pd.isna(row.home_spread_open):
                continue
            spread = float(row.home_spread_open)
        if exec_price == "best":
            best_home = _valid_spread_juice(getattr(row, "best_spread_home_odds", None))
            best_away = _valid_spread_juice(getattr(row, "best_spread_away_odds", None))
            if best_home is not None:
                home_odds = best_home
            if best_away is not None:
                away_odds = best_away
        if home_odds is None or away_odds is None:
            continue

        # P(home covers) = P(margin + spread > 0)
        z = (pred_margin + spread) / sigma
        p_home_cover = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        for side, cover_prob, odds in (
            ("home", p_home_cover, home_odds),
            ("away", 1.0 - p_home_cover, away_odds),
        ):
            dec = american_to_decimal(odds)
            market_cover = 1.0 / dec / (1.0 / dec + 1.0 / american_to_decimal(
                away_odds if side == "home" else home_odds
            ))
            gap_pp = (cover_prob - market_cover) * 100.0
            point_edge = (pred_margin + spread) if side == "home" else -(pred_margin + spread)
            if gap_pp < min_cover_gap_pp or point_edge < min_point_edge:
                continue
            units = kelly_units(cover_prob, odds)
            if units <= 0:
                continue
            diff = actual_margin + spread
            if abs(diff) < 1e-9:
                continue  # push refunded
            won = diff > 0 if side == "home" else diff < 0
            picks.append(
                {
                    "season": int(row.season),
                    "side": side,
                    "units": units,
                    "won": bool(won),
                    "pnl": units * (dec - 1.0) if won else -units,
                }
            )
    return _summarize(picks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest WNBA v2 bet policies")
    parser.add_argument("--write-policy", action="store_true")
    parser.add_argument("--min-bets", type=int, default=150)
    args = parser.parse_args()

    frame = pd.read_csv(OOS_PATH)
    frame = frame[frame.books.fillna(0) > 0]
    metadata = json.loads((OUT_DIR / "metadata.json").read_text(encoding="utf-8"))
    sigma = float(metadata.get("margin_sigma") or 12.5)
    sigma_mkt = float(metadata.get("margin_sigma_market") or sigma)
    print(
        f"OOS rows with books: {len(frame)} "
        f"({int(frame.season.min())}-{int(frame.season.max())})\n"
    )

    results: list[tuple[tuple, dict, dict]] = []

    def _score(res: dict) -> tuple:
        """Consistency first (as NHL v2): worst-season ROI, then overall ROI."""
        return (round(res["worst_season_roi"], 2), round(res["roi_pct"], 2))

    for exec_price in ("best", "consensus", "open"):
        for prob_col in ("model_prob", "model_prob_mkt"):
            for min_edge in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
                for min_ev in (2.0, 4.0, 6.0):
                    for ml_lo, ml_hi in ((-300, 300), (-250, 250), (-200, 200)):
                        params = {
                            "bet_type": "moneyline",
                            "prob_col": prob_col,
                            "exec_price": exec_price,
                            "min_edge_pp": min_edge,
                            "min_ev_pct": min_ev,
                            "ml_lo": ml_lo,
                            "ml_hi": ml_hi,
                        }
                        res = simulate_moneyline(
                            frame,
                            prob_col=prob_col,
                            min_edge_pp=min_edge,
                            min_ev_pct=min_ev,
                            ml_lo=ml_lo,
                            ml_hi=ml_hi,
                            exec_price=exec_price,
                        )
                        if res.get("bets", 0) < args.min_bets:
                            continue
                        results.append((_score(res), params, res))

        for margin_col, sig in (("model_margin", sigma), ("model_margin_mkt", sigma_mkt)):
            for min_gap in (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0):
                for min_pt in (0.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                    params = {
                        "bet_type": "spread",
                        "margin_col": margin_col,
                        "exec_price": exec_price,
                        "sigma": round(sig, 3),
                        "min_cover_gap_pp": min_gap,
                        "min_point_edge": min_pt,
                    }
                    res = simulate_spread(
                        frame,
                        margin_col=margin_col,
                        sigma=sig,
                        min_cover_gap_pp=min_gap,
                        min_point_edge=min_pt,
                        exec_price=exec_price,
                    )
                    if res.get("bets", 0) < args.min_bets:
                        continue
                    results.append((_score(res), params, res))

    results.sort(key=lambda item: item[0], reverse=True)
    print("top policies (ranked by worst-season ROI, then overall ROI):")
    for score, params, res in results[:15]:
        desc = ", ".join(f"{k}={v}" for k, v in params.items())
        print(
            f"  [worst={score[0]:6.2f} roi={score[1]:5.2f}] {desc}: bets={res['bets']} "
            f"pos={res['seasons_positive']}/{res['seasons_total']}"
        )

    if not results:
        print("no qualifying policies")
        return 1

    best_score, best_params, best_res = results[0]
    print("\nchosen policy:")
    print(json.dumps(best_params, indent=1))
    print(json.dumps({k: v for k, v in best_res.items() if k != "per_season"}, indent=1))
    print("per-season:", json.dumps(best_res["per_season"], indent=1))

    if args.write_policy:
        policy = {
            "stake_mode": "kelly",
            "kelly_fraction": KELLY_FRACTION,
            **best_params,
            "model_variant": (
                "market_aware"
                if str(best_params.get("prob_col") or best_params.get("margin_col", "")).endswith(
                    "_mkt"
                )
                else "pure"
            ),
            "backtest": {k: v for k, v in best_res.items() if k != "per_season"},
            "backtest_per_season": best_res["per_season"],
        }
        (OUT_DIR / "bet_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT_DIR / 'bet_policy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
