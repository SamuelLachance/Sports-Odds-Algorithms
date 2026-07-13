"""Upgrade Hubáček pick_strategy from hybrid OOS + simple ML gap search."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PICK_PATH = PROJECT_ROOT / "data" / "pick_strategy.json"


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


def pnl(prob: float, ml: float, won: bool) -> float:
    dec = american_to_decimal(ml)
    if not math.isfinite(dec):
        return 0.0
    stake = 1.0
    return stake * (dec - 1.0) if won else -stake


def search_ml_policy(frame: pd.DataFrame, *, league: str) -> dict | None:
    """Grid search moneyline gap/side filters; require worst-season ROI > 0."""
    home_ml_col = "home_close_ml" if "home_close_ml" in frame.columns else "home_ml"
    away_ml_col = "away_close_ml" if "away_close_ml" in frame.columns else "away_ml"
    need = ["model_prob", "home_win", home_ml_col, away_ml_col, "season"]
    if any(c not in frame.columns for c in need):
        return None
    df = frame.dropna(subset=need).copy()
    if len(df) < 200:
        return None

    best = None
    for gap in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0):
        for side in ("either", "home", "away", "dog", "fav"):
            rows = []
            for row in df.itertuples(index=False):
                hml = float(getattr(row, home_ml_col))
                aml = float(getattr(row, away_ml_col))
                if abs(hml) < 100 or abs(aml) < 100:
                    continue
                mkt_h, mkt_a = devig(hml, aml)
                p = float(row.model_prob)
                # home bet?
                candidates = []
                if p - mkt_h >= gap / 100.0:
                    candidates.append(("home", p, hml, float(row.home_win) > 0.5))
                if (1.0 - p) - mkt_a >= gap / 100.0:
                    candidates.append(("away", 1.0 - p, aml, float(row.home_win) < 0.5))
                for bet_side, prob, ml, won in candidates:
                    is_dog = ml > 0
                    is_fav = ml < 0
                    if side == "home" and bet_side != "home":
                        continue
                    if side == "away" and bet_side != "away":
                        continue
                    if side == "dog" and not is_dog:
                        continue
                    if side == "fav" and not is_fav:
                        continue
                    rows.append(
                        {
                            "season": int(row.season),
                            "pnl": pnl(prob, ml, won),
                            "won": won,
                        }
                    )
            if len(rows) < 40:
                continue
            res = pd.DataFrame(rows)
            by = res.groupby("season")["pnl"].sum()
            staked = float(len(res))
            profit = float(res["pnl"].sum())
            roi = 100.0 * profit / staked
            season_roi = 100.0 * by / res.groupby("season").size()
            worst = float(season_roi.min()) if len(season_roi) else -999.0
            pos = int((season_roi > 0).sum())
            rec = {
                "league": league,
                "enabled": worst > 0 and pos >= max(2, len(season_roi) // 2),
                "min_market_gap_pp": gap,
                "side": side,
                "bets": int(len(res)),
                "roi_pct": round(roi, 2),
                "worst_season_roi_pct": round(worst, 2),
                "seasons_positive": pos,
                "seasons_total": int(len(season_roi)),
                "win_rate": round(float(res["won"].mean()), 4),
            }
            if worst <= 0:
                continue
            if best is None or rec["roi_pct"] > best["roi_pct"]:
                best = rec
    return best


def main() -> int:
    strategy = json.loads(PICK_PATH.read_text(encoding="utf-8"))
    results = {}
    for league in ("wnba", "mlb", "nhl", "nfl", "cbb", "soccer", "nba", "cfb"):
        path = PROJECT_ROOT / "data" / "models" / f"{league}_v2" / "oos_predictions.csv"
        if not path.is_file():
            continue
        # Enrich odds from training table when thin
        frame = pd.read_csv(path)
        table_path = PROJECT_ROOT / "data" / f"{league}_history" / "training_table.csv"
        if table_path.is_file() and ("home_ml" not in frame.columns or frame["home_ml"].isna().all()):
            table = pd.read_csv(table_path)
            if league == "mlb":
                # rebuild from hybrid+table
                hyb = PROJECT_ROOT / "data" / "models" / f"{league}_v2" / "oos_predictions_hybrid.csv"
                if hyb.is_file():
                    # Can't join without keys — skip ML search
                    pass
            keys = [c for c in ("season", "date", "home", "away") if c in frame.columns and c in table.columns]
            odds = [c for c in table.columns if "ml" in c.lower() or c in ("books", "home_spread", "spread_home_odds", "spread_away_odds")]
            if keys and odds:
                frame = frame.drop(columns=[c for c in odds if c in frame.columns], errors="ignore")
                frame = frame.merge(table[keys + odds].drop_duplicates(keys), on=keys, how="left")
        if "books" not in frame.columns and "home_ml" in frame.columns:
            frame["books"] = np.where(frame["home_ml"].notna(), 1.0, 0.0)
        # fill close aliases
        if "home_close_ml" not in frame.columns and "home_ml" in frame.columns:
            frame["home_close_ml"] = frame["home_ml"]
            frame["away_close_ml"] = frame["away_ml"]
        if "home_open_ml" not in frame.columns and "home_ml_open" in frame.columns:
            frame["home_open_ml"] = frame["home_ml_open"]
            frame["away_open_ml"] = frame["away_ml_open"]
        if "home_open_ml" not in frame.columns and "home_close_ml" in frame.columns:
            frame["home_open_ml"] = frame["home_close_ml"]
            frame["away_open_ml"] = frame["away_close_ml"]

        pol = search_ml_policy(frame, league=league)
        results[league] = pol
        if pol and pol.get("enabled"):
            entry = strategy.setdefault(league, {})
            entry["enabled"] = True
            entry["min_market_gap_pp"] = pol["min_market_gap_pp"]
            if pol["side"] in ("home", "away"):
                entry["side"] = pol["side"]
            elif pol["side"] == "dog":
                entry["dogs_only"] = True
            entry["backtest_roi_pct"] = pol["roi_pct"]
            entry["note"] = (
                f"Hybrid OOS refresh: gap>={pol['min_market_gap_pp']}pp side={pol['side']} "
                f"ROI {pol['roi_pct']}% worst-season {pol['worst_season_roi_pct']}% "
                f"n={pol['bets']} ({pol['seasons_positive']}/{pol['seasons_total']} seasons +)."
            )
            # Write lightweight bet_policy
            bp = {
                "source": "hybrid_upgrade_ml_search",
                "policy": pol,
                "created_from": "scripts/upgrade_hybrid_bet_strategy.py",
            }
            (PROJECT_ROOT / "data" / "models" / f"{league}_v2" / "bet_policy.json").write_text(
                json.dumps(bp, indent=2), encoding="utf-8"
            )
            print(f"{league}: ENABLED {pol}", flush=True)
        else:
            print(f"{league}: no positive worst-season ML niche ({pol})", flush=True)

    PICK_PATH.write_text(json.dumps(strategy, indent=2), encoding="utf-8")
    out = PROJECT_ROOT / "data" / "models" / "hybrid_bet_strategy_summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {PICK_PATH} and {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
