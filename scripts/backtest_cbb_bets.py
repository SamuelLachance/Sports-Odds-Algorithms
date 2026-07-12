"""Backtest CBB official-pick spread gates on walk-forward predictions.

Prefers CBB v2 OOS margins from data/models/cbb_v2/oos_predictions.csv when
present, joined to closing spreads in data/supplemental/closing-odds/cbb.csv.
Otherwise rebuilds walk-forward margins via scripts/train_cbb_model.py on the
training table and joins the same odds. Sweeps the Hubáček spread gate grid
and ranks by worst-season ROI. Use --write-policy only when the best policy
has positive worst-season ROI (writes data/models/cbb_v2/bet_policy.json).
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

from web.bet_advisor import SPREAD_MARGIN_SIGMA  # noqa: E402
from web.cbb_v2.data import canon_abbr  # noqa: E402

ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "cbb.csv"
TABLE_PATH = PROJECT_ROOT / "data" / "cbb_history" / "training_table.csv"
V2_DIR = PROJECT_ROOT / "data" / "models" / "cbb_v2"
V2_OOS_PATH = V2_DIR / "oos_predictions.csv"
V2_META_PATH = V2_DIR / "metadata.json"
CACHE_JOINED_PATH = V2_DIR / "oos_with_closing_odds.csv"

LEAGUE = "cbb"
SIGMA = SPREAD_MARGIN_SIGMA[LEAGUE]

KELLY_FRACTION = 0.25
KELLY_CAP_UNITS = 3.0
KELLY_MIN_UNITS = 0.25
DEFAULT_SPREAD_JUICE = -110.0
MIN_SPREAD_CONFIDENCE_PP = 5.0


def american_to_decimal(ml: float) -> float:
    """Decimal odds from American; EVEN 0 -> 2.0; invalid |ml|<100 -> NaN."""
    if ml == 0:
        ml = 100.0
    if abs(ml) < 100:
        return float("nan")
    return 1.0 + (ml / 100.0 if ml > 0 else 100.0 / abs(ml))


def kelly_units(prob: float, ml: float) -> float:
    dec = american_to_decimal(ml)
    b = dec - 1.0
    edge = prob * dec - 1.0
    if edge <= 0 or b <= 0:
        return 0.0
    fraction = edge / b * KELLY_FRACTION
    units = fraction * 100.0
    return float(min(max(units, KELLY_MIN_UNITS), KELLY_CAP_UNITS))


def load_odds() -> pd.DataFrame:
    frame = pd.read_csv(ODDS_CSV)
    frame["date"] = frame.date.astype(str).str[:10]
    frame["home_key"] = frame.home_key.map(lambda x: canon_abbr(str(x)))
    frame["away_key"] = frame.away_key.map(lambda x: canon_abbr(str(x)))
    frame = frame.dropna(subset=["home_close_spread"]).copy()
    return frame.reset_index(drop=True)


def _attach_odds(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join OOS/walk-forward margins to closing-odds spreads."""
    preds = preds.copy()
    preds["date"] = preds.date.astype(str).str[:10]
    if "home_abbr" in preds.columns:
        preds["home_abbr"] = preds.home_abbr.map(lambda x: canon_abbr(str(x)))
        preds["away_abbr"] = preds.away_abbr.map(lambda x: canon_abbr(str(x)))
        left_on = ["date", "home_abbr", "away_abbr"]
    elif "home" in preds.columns and preds["home"].dtype == object:
        preds["home"] = preds.home.map(lambda x: canon_abbr(str(x)))
        preds["away"] = preds.away.map(lambda x: canon_abbr(str(x)))
        left_on = ["date", "home", "away"]
    else:
        raise ValueError("predictions need home_abbr/away_abbr (or abbr home/away) to join odds")

    odds_cols = [
        "date",
        "home_key",
        "away_key",
        "home_close_spread",
        "away_close_spread",
        "home_spread_odds",
        "away_spread_odds",
        "home_open_spread",
        "away_open_spread",
        "n_books",
    ]
    present = [c for c in odds_cols if c in odds.columns]
    merged = preds.merge(
        odds[present],
        left_on=left_on,
        right_on=["date", "home_key", "away_key"],
        how="inner",
        suffixes=("", "_odds"),
    )
    merged["home_spread"] = merged["home_close_spread"].astype(float)
    merged["home_spread_open"] = pd.to_numeric(merged.get("home_open_spread"), errors="coerce")
    merged["spread_home_odds"] = pd.to_numeric(merged.get("home_spread_odds"), errors="coerce")
    merged["spread_away_odds"] = pd.to_numeric(merged.get("away_spread_odds"), errors="coerce")
    if "margin" not in merged.columns:
        if {"home_score", "away_score"}.issubset(merged.columns):
            merged["margin"] = merged.home_score.astype(float) - merged.away_score.astype(float)
        elif {"home_final", "away_final"}.issubset(merged.columns):
            merged["margin"] = merged.home_final.astype(float) - merged.away_final.astype(float)
        else:
            raise ValueError("predictions missing margin / scores")
    if "season" not in merged.columns:
        # CBB season = ending calendar year (Nov → Apr)
        days = pd.to_datetime(merged.date)
        merged["season"] = np.where(days.dt.month >= 8, days.dt.year + 1, days.dt.year)
    keep = [
        "season",
        "date",
        "model_margin",
        "margin",
        "home_spread",
        "home_spread_open",
        "spread_home_odds",
        "spread_away_odds",
        "n_books",
    ]
    optional = [c for c in ("home_abbr", "away_abbr", "home", "away", "event_id") if c in merged.columns]
    out = merged[optional + keep].dropna(subset=["home_spread", "model_margin", "margin"])
    return out.reset_index(drop=True)


def load_v2_predictions(odds: pd.DataFrame) -> pd.DataFrame | None:
    if not V2_OOS_PATH.is_file():
        return None
    preds = pd.read_csv(V2_OOS_PATH)
    if "model_margin" not in preds.columns or "margin" not in preds.columns:
        return None
    return _attach_odds(preds, odds)


def build_walk_forward_predictions(odds: pd.DataFrame) -> pd.DataFrame:
    """Rebuild period walk-forward margins from the training table (slow)."""
    if not TABLE_PATH.is_file():
        raise FileNotFoundError(
            f"missing {TABLE_PATH}; run scripts/build_cbb_training_table.py "
            "or provide data/models/cbb_v2/oos_predictions.csv"
        )
    from scripts.train_cbb_model import walk_forward  # noqa: WPS433

    frame = pd.read_csv(TABLE_PATH)
    frame = frame.dropna(subset=["home_win", "margin"])
    end_season = int(frame.season.max())
    print(
        f"building walk-forward CBB v2 margins on training table "
        f"({len(frame)} rows through season {end_season})...",
        flush=True,
    )
    oos = walk_forward(frame[frame.season <= end_season], end_season)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    oos.to_csv(V2_OOS_PATH, index=False)
    print(f"wrote {V2_OOS_PATH} ({len(oos)} rows)", flush=True)
    return _attach_odds(oos, odds)


def build_side_table(preds: pd.DataFrame, sigma: float, *, exec_price: str) -> pd.DataFrame:
    if exec_price == "open":
        preds = preds.dropna(subset=["home_spread_open"]).copy()
        spread = preds.home_spread_open.astype(float)
    else:
        spread = preds.home_spread.astype(float)

    home_odds = preds.spread_home_odds.fillna(DEFAULT_SPREAD_JUICE)
    away_odds = preds.spread_away_odds.fillna(DEFAULT_SPREAD_JUICE)
    home_odds = home_odds.where(home_odds.abs() >= 100, DEFAULT_SPREAD_JUICE)
    away_odds = away_odds.where(away_odds.abs() >= 100, DEFAULT_SPREAD_JUICE)

    edge = preds.model_margin + spread
    z = edge / sigma
    p_home_cover = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

    dec_home = home_odds.map(american_to_decimal)
    dec_away = away_odds.map(american_to_decimal)
    inv_home = 1.0 / dec_home
    inv_away = 1.0 / dec_away
    market_home = inv_home / (inv_home + inv_away)

    diff = preds.margin + spread

    sides: list[pd.DataFrame] = []
    for side in ("home", "away"):
        if side == "home":
            cover_prob = p_home_cover
            odds, dec = home_odds, dec_home
            market = market_home
            point_edge = edge
            won = diff > 0
        else:
            cover_prob = 1.0 - p_home_cover
            odds, dec = away_odds, dec_away
            market = 1.0 - market_home
            point_edge = -edge
            won = diff < 0
        table = pd.DataFrame(
            {
                "season": preds.season,
                "side": side,
                "cover_prob": cover_prob,
                "odds": odds,
                "dec": dec,
                "gap_pp": (cover_prob - market) * 100.0,
                "point_edge": point_edge,
                "ev_pct": (cover_prob * dec - 1.0) * 100.0,
                "won": won,
                "push": diff.abs() < 1e-9,
            }
        )
        sides.append(table)
    out = pd.concat(sides, ignore_index=True)
    out["units"] = [
        kelly_units(prob, ml) for prob, ml in zip(out.cover_prob, out.odds)
    ]
    out["confidence_pp"] = (out.cover_prob * 100.0 - 50.0).abs()
    return out


def summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0}
    pnl = np.where(bets.won, bets.units * (bets.dec - 1.0), -bets.units)
    bets = bets.assign(pnl=pnl)
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


def simulate_spread(
    sides: pd.DataFrame,
    *,
    min_cover_gap_pp: float,
    min_point_edge: float,
    min_ev_pct: float,
) -> dict:
    mask = (
        (sides.gap_pp >= min_cover_gap_pp)
        & (sides.point_edge >= min_point_edge)
        & (sides.ev_pct >= min_ev_pct)
        & (sides.confidence_pp >= MIN_SPREAD_CONFIDENCE_PP)
        & (sides.units > 0)
        & ~sides.push
    )
    return summarize(sides[mask])


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest CBB spread gates")
    parser.add_argument("--min-bets", type=int, default=80)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore cached OOS and rebuild walk-forward from the training table",
    )
    parser.add_argument(
        "--write-policy",
        action="store_true",
        help="Write bet_policy.json only if best worst-season ROI > 0",
    )
    parser.add_argument(
        "--cache-joined",
        action="store_true",
        help=f"Write joined OOS+odds to {CACHE_JOINED_PATH.name}",
    )
    args = parser.parse_args()

    if not ODDS_CSV.is_file():
        print(f"missing odds file: {ODDS_CSV}", flush=True)
        return 1

    odds = load_odds()
    print(
        f"closing odds: {len(odds)} games with spreads "
        f"({odds.date.min()} -> {odds.date.max()})",
        flush=True,
    )

    source = "cbb_v2"
    sigma = SIGMA
    if V2_META_PATH.is_file():
        meta = json.loads(V2_META_PATH.read_text(encoding="utf-8"))
        sigma = float(meta.get("margin_sigma") or SIGMA)

    preds: pd.DataFrame | None = None
    if not args.rebuild:
        preds = load_v2_predictions(odds)
        if preds is not None:
            print(f"using CBB v2 OOS margins joined to closing odds ({len(preds)} rows, sigma={sigma:.2f})")
    if preds is None:
        source = "cbb_v2_walk_forward"
        preds = build_walk_forward_predictions(odds)
        print(f"walk-forward join: {len(preds)} rows, sigma={sigma:.2f}")

    if preds.empty:
        print("no games after joining model margins to closing spreads")
        return 1

    if args.cache_joined:
        V2_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_csv(CACHE_JOINED_PATH, index=False)
        print(f"wrote {CACHE_JOINED_PATH}")

    residual = preds.margin - preds.model_margin
    open_n = int(preds.home_spread_open.notna().sum()) if "home_spread_open" in preds.columns else 0
    print(
        f"source={source} games with closing spread: {len(preds)} "
        f"({int(preds.season.min())}-{int(preds.season.max())}), "
        f"with opening spread: {open_n}\n"
        f"model-vs-actual margin sigma {residual.std():.2f} (gate sigma {sigma}), "
        f"model MAE {residual.abs().mean():.2f} "
        f"vs closing-spread MAE {(preds.margin + preds.home_spread).abs().mean():.2f}\n"
    )

    grid = [
        (min_gap, min_pt, min_ev)
        for min_gap in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0)
        for min_pt in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)
        for min_ev in (2.0, 2.5, 3.0)
    ]

    best_by_exec: dict[str, tuple] = {}
    for exec_price in ("close", "open"):
        if exec_price == "open" and open_n < args.min_bets:
            print(f"skipping {exec_price.upper()} — insufficient opening spreads")
            continue
        sides = build_side_table(preds, sigma, exec_price=exec_price)
        results: list[tuple[tuple, dict, dict]] = []
        for min_gap, min_pt, min_ev in grid:
            params = {
                "exec_price": exec_price,
                "min_spread_cover_gap_pp": min_gap,
                "min_spread_point_edge": min_pt,
                "min_ev_pct": min_ev,
            }
            res = simulate_spread(
                sides,
                min_cover_gap_pp=min_gap,
                min_point_edge=min_pt,
                min_ev_pct=min_ev,
            )
            if res.get("bets", 0) < args.min_bets:
                continue
            score = (round(res["worst_season_roi"], 2), round(res["roi_pct"], 2))
            results.append((score, params, res))

        if not results:
            print(f"no qualifying policies at the {exec_price.upper()}")
            continue

        results.sort(key=lambda item: item[0], reverse=True)
        print(f"top policies at the {exec_price.upper()} (worst-season ROI, then overall ROI):")
        for score, params, res in results[:10]:
            desc = ", ".join(
                f"{k}={v}" for k, v in params.items() if k != "exec_price"
            )
            print(
                f"  [worst={score[0]:7.2f} roi={score[1]:6.2f}] {desc}: bets={res['bets']} "
                f"win={res['win_rate']:.3f} pos={res['seasons_positive']}/{res['seasons_total']}"
            )
        best_by_exec[exec_price] = results[0]
        print()

    overall_best = None
    for exec_price, (score, params, res) in best_by_exec.items():
        print(f"best policy at the {exec_price.upper()}:")
        print(json.dumps(params, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "per_season"}, indent=1))
        print("per-season:", json.dumps(res["per_season"], indent=1))
        print()
        if overall_best is None or score > overall_best[0]:
            overall_best = (score, params, res)

    enable_picks = False
    if overall_best is not None:
        score, params, res = overall_best
        enable_picks = float(res["worst_season_roi"]) > 0
        print(
            f"picks_enable_gate: worst_season_roi={res['worst_season_roi']} "
            f"seasons={res['seasons_total']} "
            f"-> {'ENABLE' if enable_picks else 'KEEP DISABLED'}"
        )
        if res["seasons_total"] < 2:
            print(
                "note: only one odds season available - positive single-season ROI "
                "is not multi-season evidence; prefer KEEP DISABLED until more seasons filled."
            )
        if args.write_policy:
            if not enable_picks:
                print("--write-policy requested but worst-season ROI <= 0; not writing")
            else:
                V2_DIR.mkdir(parents=True, exist_ok=True)
                policy = {
                    "stake_mode": "kelly",
                    "kelly_fraction": KELLY_FRACTION,
                    "bet_type": "spread",
                    "source": source,
                    "sigma": round(sigma, 3),
                    "enabled": False if res["seasons_total"] < 2 else True,
                    "enabled_note": (
                        "Single-season odds sample — policy recorded for research; "
                        "official picks stay disabled until multi-season confirmation."
                        if res["seasons_total"] < 2
                        else "Worst-season ROI > 0 on available odds seasons."
                    ),
                    **params,
                    "backtest": {k: v for k, v in res.items() if k != "per_season"},
                    "backtest_per_season": res["per_season"],
                }
                path = V2_DIR / "bet_policy.json"
                path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
                print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
