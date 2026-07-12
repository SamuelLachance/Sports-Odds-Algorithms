"""Backtest CFB official-pick spread gates on walk-forward predictions.

Prefers CFB v2 OOS margins from data/models/cfb_v2/oos_predictions.csv when
present; otherwise rebuilds the production Elo model (web/football_pred_model.py)
per game day on a 120-day rolling window. Sweeps the Hubáček spread gate grid
and ranks by worst-season ROI. Use --write-policy only when the best policy
has positive worst-season ROI (writes data/models/cfb_v2/bet_policy.json).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import SPREAD_MARGIN_SIGMA  # noqa: E402
from web.football_pred_model import (  # noqa: E402
    MIN_TEAM_GAMES,
    build_football_model,
    predict_matchup_from_model,
)

ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "cfb.csv"
ELO_OOS_PATH = PROJECT_ROOT / "data" / "models" / "cfb_backtest" / "oos_predictions.csv"
V2_DIR = PROJECT_ROOT / "data" / "models" / "cfb_v2"
V2_OOS_PATH = V2_DIR / "oos_predictions.csv"
V2_META_PATH = V2_DIR / "metadata.json"

LEAGUE = "cfb"
SIGMA = SPREAD_MARGIN_SIGMA[LEAGUE]
TRAIN_LOOKBACK_DAYS = 120

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


def _season_of(day: date) -> int:
    return day.year if day.month >= 7 else day.year - 1


def load_games() -> pd.DataFrame:
    frame = pd.read_csv(ODDS_CSV)
    frame["day"] = pd.to_datetime(frame.date).dt.date
    frame["season"] = frame.day.map(_season_of)
    frame = frame.sort_values(["day", "home_key"]).reset_index(drop=True)
    return frame


def build_walk_forward_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Elo baseline walk-forward margins (legacy path)."""
    rows: list[dict] = []
    for gameday, day_games in frame.groupby("day", sort=True):
        window_start = gameday - timedelta(days=TRAIN_LOOKBACK_DAYS)
        train_frame = frame[(frame.day >= window_start) & (frame.day < gameday)]
        if train_frame.empty:
            continue
        train = [
            (r.home_key, r.away_key, r.home_key, r.away_key, int(r.home_final), int(r.away_final))
            for r in train_frame.itertuples(index=False)
        ]
        model = build_football_model(train, LEAGUE)
        if not model:
            continue
        counts = model["team_game_counts"]
        for game in day_games.itertuples(index=False):
            if pd.isna(game.home_close_spread):
                continue
            if (
                counts.get(game.home_key, 0) < MIN_TEAM_GAMES
                or counts.get(game.away_key, 0) < MIN_TEAM_GAMES
            ):
                continue
            pred = predict_matchup_from_model(model, game.home_key, game.away_key)
            if not pred:
                continue
            rows.append(
                {
                    "season": int(game.season),
                    "date": gameday.isoformat(),
                    "home": game.home_key,
                    "away": game.away_key,
                    "model_margin": -float(pred["projected_spread"]),
                    "margin": float(game.home_final - game.away_final),
                    "home_spread": float(game.home_close_spread),
                    "home_spread_open": (
                        float(game.home_open_spread)
                        if pd.notna(game.home_open_spread)
                        else np.nan
                    ),
                    "spread_home_odds": (
                        float(game.home_spread_odds)
                        if pd.notna(game.home_spread_odds)
                        else np.nan
                    ),
                    "spread_away_odds": (
                        float(game.away_spread_odds)
                        if pd.notna(game.away_spread_odds)
                        else np.nan
                    ),
                    "n_books": int(game.n_books) if pd.notna(game.n_books) else 0,
                }
            )
    return pd.DataFrame(rows)


def load_v2_predictions() -> pd.DataFrame | None:
    if not V2_OOS_PATH.is_file():
        return None
    preds = pd.read_csv(V2_OOS_PATH)
    # Normalize column names used by the side table
    rename = {}
    if "home_spread" not in preds.columns and "home_close_spread" in preds.columns:
        rename["home_close_spread"] = "home_spread"
    if "home_spread_open" not in preds.columns and "home_open_spread" in preds.columns:
        rename["home_open_spread"] = "home_spread_open"
    if "spread_home_odds" not in preds.columns and "home_spread_odds" in preds.columns:
        rename["home_spread_odds"] = "spread_home_odds"
    if "spread_away_odds" not in preds.columns and "away_spread_odds" in preds.columns:
        rename["away_spread_odds"] = "spread_away_odds"
    if rename:
        preds = preds.rename(columns=rename)
    if "model_margin" not in preds.columns or "margin" not in preds.columns:
        return None
    preds = preds.dropna(subset=["home_spread", "model_margin", "margin"])
    return preds.reset_index(drop=True)


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
    parser = argparse.ArgumentParser(description="Backtest CFB spread gates")
    parser.add_argument("--min-bets", type=int, default=80)
    parser.add_argument("--rebuild", action="store_true", help="Recompute Elo walk-forward")
    parser.add_argument(
        "--force-elo",
        action="store_true",
        help="Ignore v2 OOS and use Elo baseline predictions",
    )
    parser.add_argument(
        "--write-policy",
        action="store_true",
        help="Write bet_policy.json only if best worst-season ROI > 0",
    )
    args = parser.parse_args()

    source = "elo"
    sigma = SIGMA
    preds = None if args.force_elo else load_v2_predictions()
    if preds is not None:
        source = "cfb_v2"
        if V2_META_PATH.is_file():
            meta = json.loads(V2_META_PATH.read_text(encoding="utf-8"))
            sigma = float(meta.get("margin_sigma") or SIGMA)
        print(f"using CFB v2 OOS margins ({len(preds)} rows, sigma={sigma:.2f})")
    elif ELO_OOS_PATH.is_file() and not args.rebuild:
        preds = pd.read_csv(ELO_OOS_PATH)
        print(f"using cached Elo OOS ({len(preds)} rows)")
    else:
        print("building walk-forward CFB Elo predictions (per-gameday rebuild)...", flush=True)
        # fix load_games reset_index bug if any
        frame = pd.read_csv(ODDS_CSV)
        frame["day"] = pd.to_datetime(frame.date).dt.date
        frame["season"] = frame.day.map(_season_of)
        frame = frame.sort_values(["day", "home_key"]).reset_index(drop=True)
        preds = build_walk_forward_predictions(frame)
        ELO_OOS_PATH.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(ELO_OOS_PATH, index=False)
        print(f"wrote {ELO_OOS_PATH}")

    residual = preds.margin - preds.model_margin
    open_preds = preds.dropna(subset=["home_spread_open"]) if "home_spread_open" in preds.columns else preds.iloc[0:0]
    print(
        f"source={source} games with closing spread: {len(preds)} "
        f"({int(preds.season.min())}-{int(preds.season.max())}), "
        f"with opening spread: {len(open_preds)}\n"
        f"model-vs-actual margin sigma {residual.std():.2f} (gate sigma {sigma}), "
        f"model MAE {residual.abs().mean():.2f} "
        f"vs closing-spread MAE {(preds.margin + preds.home_spread).abs().mean():.2f}\n"
    )

    grid = [
        (min_gap, min_pt, min_ev)
        for min_gap in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)
        for min_pt in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
        for min_ev in (2.0, 2.5, 3.0)
    ]

    best_by_exec: dict[str, tuple] = {}
    for exec_price in ("close", "open"):
        if exec_price == "open" and (
            "home_spread_open" not in preds.columns
            or preds["home_spread_open"].notna().sum() < args.min_bets
        ):
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
            f"-> {'ENABLE' if enable_picks else 'KEEP DISABLED'}"
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
