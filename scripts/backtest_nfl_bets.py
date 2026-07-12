"""Backtest NFL official-pick spread gates on walk-forward predictions.

Prefers NFL v2 OOS margins from data/models/nfl_v2/oos_predictions.csv when
present; otherwise rebuilds the production nfelo model
(web/football_pred_model.py) per game day. Sweeps the Hubáček spread gate
grid and ranks by worst-season ROI. Use --write-policy only when the best
policy has positive worst-season ROI (writes data/models/nfl_v2/bet_policy.json).

nflverse has no opening lines, so the close is the only executable price.
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
from web.football_pred_model import (  # noqa: E402
    MIN_TEAM_GAMES,
    build_football_model,
    predict_matchup_from_model,
)

GAMES_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "nflverse_games.csv"
ELO_OOS_PATH = PROJECT_ROOT / "data" / "models" / "nfl_backtest" / "oos_predictions.csv"
V2_DIR = PROJECT_ROOT / "data" / "models" / "nfl_v2"
V2_OOS_PATH = V2_DIR / "oos_predictions.csv"
V2_META_PATH = V2_DIR / "metadata.json"

LEAGUE = "nfl"
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


def load_games() -> pd.DataFrame:
    frame = pd.read_csv(GAMES_CSV)
    frame = frame[frame.game_type.isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
    frame = frame.dropna(subset=["result", "spread_line", "home_score", "away_score"])
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    frame["home_key"] = frame.home_team.str.lower()
    frame["away_key"] = frame.away_team.str.lower()
    # nflverse spread_line = expected home margin; betting convention flips sign.
    frame["home_spread"] = -frame.spread_line.astype(float)
    return frame


def build_walk_forward_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Model margin per game from the production nfelo model, walk-forward."""
    rows: list[dict] = []
    seasons = sorted(frame.season.unique())
    for season in seasons:
        prior = frame[frame.season == season - 1]
        current = frame[frame.season == season]
        prior_tuples = [
            (r.home_key, r.away_key, r.home_key, r.away_key, int(r.home_score), int(r.away_score))
            for r in prior.itertuples(index=False)
        ]
        for gameday, day_games in current.groupby("gameday", sort=True):
            played = current[current.gameday < gameday]
            train = prior_tuples + [
                (r.home_key, r.away_key, r.home_key, r.away_key, int(r.home_score), int(r.away_score))
                for r in played.itertuples(index=False)
            ]
            model = build_football_model(train, LEAGUE)
            if not model:
                continue
            counts = model["team_game_counts"]
            for game in day_games.itertuples(index=False):
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
                        "date": gameday,
                        "home": game.home_key,
                        "away": game.away_key,
                        "model_margin": -float(pred["projected_spread"]),
                        "margin": float(game.result),
                        "home_spread": float(game.home_spread),
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
                    }
                )
    return pd.DataFrame(rows)


def load_v2_predictions() -> pd.DataFrame | None:
    if not V2_OOS_PATH.is_file():
        return None
    preds = pd.read_csv(V2_OOS_PATH)
    rename = {}
    if "home_spread" not in preds.columns and "home_close_spread" in preds.columns:
        rename["home_close_spread"] = "home_spread"
    if "spread_home_odds" not in preds.columns and "home_spread_odds" in preds.columns:
        rename["home_spread_odds"] = "spread_home_odds"
    if "spread_away_odds" not in preds.columns and "away_spread_odds" in preds.columns:
        rename["away_spread_odds"] = "spread_away_odds"
    if rename:
        preds = preds.rename(columns=rename)
    if "model_margin" not in preds.columns or "margin" not in preds.columns:
        return None
    if "home_spread" not in preds.columns:
        return None
    preds = preds.dropna(subset=["home_spread", "model_margin", "margin"])
    return preds.reset_index(drop=True)


def build_side_table(preds: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """Expand per-game predictions into per-side candidates with fixed metrics."""
    home_odds = preds.spread_home_odds.fillna(DEFAULT_SPREAD_JUICE)
    away_odds = preds.spread_away_odds.fillna(DEFAULT_SPREAD_JUICE)
    home_odds = home_odds.where(home_odds.abs() >= 100, DEFAULT_SPREAD_JUICE)
    away_odds = away_odds.where(away_odds.abs() >= 100, DEFAULT_SPREAD_JUICE)

    edge = preds.model_margin + preds.home_spread
    z = edge / sigma
    p_home_cover = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

    dec_home = home_odds.map(american_to_decimal)
    dec_away = away_odds.map(american_to_decimal)
    inv_home = 1.0 / dec_home
    inv_away = 1.0 / dec_away
    market_home = inv_home / (inv_home + inv_away)

    diff = preds.margin + preds.home_spread

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
    parser = argparse.ArgumentParser(description="Backtest NFL spread gates")
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
        source = "nfl_v2"
        if V2_META_PATH.is_file():
            meta = json.loads(V2_META_PATH.read_text(encoding="utf-8"))
            sigma = float(meta.get("margin_sigma") or SIGMA)
        print(f"using NFL v2 OOS margins ({len(preds)} rows, sigma={sigma:.2f})")
    elif ELO_OOS_PATH.is_file() and not args.rebuild:
        preds = pd.read_csv(ELO_OOS_PATH)
        print(f"using cached Elo OOS ({len(preds)} rows)")
    else:
        print("building walk-forward nfelo predictions (per-gameday rebuild)...", flush=True)
        preds = build_walk_forward_predictions(load_games())
        ELO_OOS_PATH.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(ELO_OOS_PATH, index=False)
        print(f"wrote {ELO_OOS_PATH}")

    residual = preds.margin - preds.model_margin
    print(
        f"source={source} games: {len(preds)} ({int(preds.season.min())}-{int(preds.season.max())}), "
        f"model-vs-actual margin sigma {residual.std():.2f} "
        f"(gate sigma {sigma}), MAE {residual.abs().mean():.2f} "
        f"vs closing-spread MAE {(preds.margin + preds.home_spread).abs().mean():.2f}\n"
    )

    sides = build_side_table(preds, sigma)

    results: list[tuple[tuple, dict, dict]] = []
    for min_gap in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
        for min_pt in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
            for min_ev in (2.0, 2.5, 3.0):
                params = {
                    "exec_price": "close",
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
        print("no qualifying policies")
        return 1

    results.sort(key=lambda item: item[0], reverse=True)
    print("top policies (ranked by worst-season ROI, then overall ROI, at the CLOSE):")
    for score, params, res in results[:15]:
        desc = ", ".join(f"{k}={v}" for k, v in params.items() if k != "exec_price")
        print(
            f"  [worst={score[0]:7.2f} roi={score[1]:6.2f}] {desc}: bets={res['bets']} "
            f"win={res['win_rate']:.3f} pos={res['seasons_positive']}/{res['seasons_total']}"
        )

    best_score, best_params, best_res = results[0]
    print("\nbest policy by worst-season ROI:")
    print(json.dumps(best_params, indent=1))
    print(json.dumps({k: v for k, v in best_res.items() if k != "per_season"}, indent=1))
    print("per-season:", json.dumps(best_res["per_season"], indent=1))

    by_roi = max(results, key=lambda item: item[2]["roi_pct"])
    if by_roi[1] != best_params:
        print("\nbest policy by overall ROI:")
        print(json.dumps(by_roi[1], indent=1))
        print(json.dumps({k: v for k, v in by_roi[2].items() if k != "per_season"}, indent=1))

    enable_picks = float(best_res["worst_season_roi"]) > 0
    print(
        f"\npicks_enable_gate: worst_season_roi={best_res['worst_season_roi']} "
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
                **best_params,
                "backtest": {k: v for k, v in best_res.items() if k != "per_season"},
                "backtest_per_season": best_res["per_season"],
            }
            path = V2_DIR / "bet_policy.json"
            path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
