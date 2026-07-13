"""Proper Hubáček NFL moneyline niche search (vectorized).

Aligns with live Hubáček math:
  - gap gate uses decorrelated prob (sport-minus-market push)
  - EV / Kelly uses pre-decorrelation sport model prob
  - searches hybrid + XGB OOS, model_raw (preferred) and model_prob
  - CBB-style tight niches + recent season windows

Enable gate: worst_season_roi > 0 with >= min_seasons and >= min_bets.

Usage:
  python scripts/search_nfl_hubacek.py
  python scripts/search_nfl_hubacek.py --write-policy
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KELLY_FRACTION = 0.25
KELLY_CAP = 3.0
KELLY_MIN = 0.25
V2 = PROJECT_ROOT / "data" / "models" / "nfl_v2"


def am2dec(ml: np.ndarray) -> np.ndarray:
    ml = np.where(ml == 0, 100.0, ml.astype(float))
    out = np.full(ml.shape, np.nan)
    pos = ml > 0
    neg = ml < 0
    out[pos] = 1.0 + ml[pos] / 100.0
    out[neg] = 1.0 + 100.0 / np.abs(ml[neg])
    out[np.abs(ml) < 100] = np.nan
    return out


def kelly_units(prob: np.ndarray, ml: np.ndarray) -> np.ndarray:
    dec = am2dec(ml)
    b = dec - 1.0
    edge = prob * dec - 1.0
    units = np.where((edge > 0) & (b > 0), edge / b * KELLY_FRACTION * 100.0, 0.0)
    units = np.clip(units, KELLY_MIN, KELLY_CAP)
    return np.where(edge > 0, units, 0.0)


def summarize(
    season: np.ndarray,
    units: np.ndarray,
    pnl: np.ndarray,
    won: np.ndarray,
    mask: np.ndarray,
    *,
    min_bets: int,
    min_seasons: int,
    require_all_positive: bool,
) -> dict | None:
    if int(mask.sum()) < min_bets:
        return None
    s = season[mask]
    u = units[mask]
    p = pnl[mask]
    w = won[mask]
    seasons = np.unique(s)
    if len(seasons) < min_seasons:
        return None
    per: dict[str, dict] = {}
    rois: list[float] = []
    for se in seasons:
        m = s == se
        n = int(m.sum())
        st = float(u[m].sum())
        if st <= 0:
            return None
        roi = float(p[m].sum() / st * 100.0)
        if require_all_positive and roi <= 0:
            return None
        rois.append(roi)
        per[str(int(se))] = {
            "bets": n,
            "roi_pct": round(roi, 2),
            "profit_units": round(float(p[m].sum()), 1),
        }
    st = float(u.sum())
    pr = float(p.sum())
    return {
        "bets": int(len(u)),
        "staked_units": round(st, 1),
        "profit_units": round(pr, 1),
        "roi_pct": round(pr / st * 100.0, 2),
        "win_rate": round(float(w.mean()), 4),
        "seasons_positive": int(sum(1 for r in rois if r > 0)),
        "seasons_total": int(len(rois)),
        "worst_season_roi": round(float(min(rois)), 2),
        "median_season_roi": round(float(np.median(rois)), 2),
        "per_season": per,
    }


def load_oos(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "home_close_ml" not in frame.columns or frame["home_close_ml"].isna().all():
        for src in (V2 / "oos_predictions.csv", PROJECT_ROOT / "data" / "nfl_history" / "training_table.csv"):
            if not src.is_file():
                continue
            other = pd.read_csv(src)
            keys = [c for c in ("season", "date", "home", "away") if c in frame.columns and c in other.columns]
            odds = [c for c in ("home_close_ml", "away_close_ml") if c in other.columns]
            if keys and odds:
                frame = frame.merge(other[keys + odds].drop_duplicates(keys), on=keys, how="left", suffixes=("", "_x"))
                break
    frame = frame.dropna(subset=["home_win", "season", "home_close_ml", "away_close_ml"]).copy()
    frame["source"] = label
    return frame


def build_candidates(frame: pd.DataFrame, prob_col: str, decor_w: float) -> dict[str, np.ndarray] | None:
    if prob_col not in frame.columns:
        return None
    hml = frame["home_close_ml"].to_numpy(dtype=float)
    aml = frame["away_close_ml"].to_numpy(dtype=float)
    valid = (np.abs(hml) >= 100) & (np.abs(aml) >= 100) & np.isfinite(frame[prob_col].to_numpy(dtype=float))
    if not valid.any():
        return None
    frame = frame.loc[valid].reset_index(drop=True)
    hml = frame["home_close_ml"].to_numpy(dtype=float)
    aml = frame["away_close_ml"].to_numpy(dtype=float)
    p = np.clip(frame[prob_col].to_numpy(dtype=float), 0.01, 0.99)
    dh, da = am2dec(hml), am2dec(aml)
    ih, ia = 1.0 / dh, 1.0 / da
    tot = ih + ia
    mkt_h, mkt_a = ih / tot, ia / tot

    # Hubáček: amplify disagreement for gap gate; EV uses pre-decor p
    p_gate = np.clip(p + decor_w * (p - mkt_h), 0.01, 0.99)
    pa = 1.0 - p
    pa_gate = np.clip(pa + decor_w * (pa - mkt_a), 0.01, 0.99)

    season = frame["season"].to_numpy(dtype=int)
    home_win = frame["home_win"].to_numpy(dtype=float)

    rows_season = []
    rows_ml = []
    rows_edge = []
    rows_ev = []
    rows_units = []
    rows_pnl = []
    rows_won = []
    rows_side = []  # 0 home, 1 away
    rows_fav = []
    rows_dog = []
    rows_slight = []
    rows_bigdog = []

    for side, prob_ev, p_g, mkt, ml, won_flag in (
        (0, p, p_gate, mkt_h, hml, home_win > 0.5),
        (1, pa, pa_gate, mkt_a, aml, home_win < 0.5),
    ):
        edge_pp = (p_g - mkt) * 100.0
        dec = am2dec(ml)
        ev_pct = (prob_ev * dec - 1.0) * 100.0
        units = kelly_units(prob_ev, ml)
        take = (units > 0) & np.isfinite(edge_pp) & np.isfinite(ev_pct)
        if not take.any():
            continue
        n = int(take.sum())
        rows_season.append(season[take])
        rows_ml.append(ml[take])
        rows_edge.append(edge_pp[take])
        rows_ev.append(ev_pct[take])
        rows_units.append(units[take])
        won = won_flag[take]
        rows_won.append(won.astype(bool))
        pnl = np.where(won, units[take] * (dec[take] - 1.0), -units[take])
        rows_pnl.append(pnl)
        rows_side.append(np.full(n, side, dtype=int))
        ml_t = ml[take]
        rows_fav.append(ml_t < 0)
        rows_dog.append(ml_t > 0)
        rows_slight.append((ml_t < 0) & (ml_t >= -200))
        rows_bigdog.append(ml_t >= 150)

    if not rows_season:
        return None
    return {
        "season": np.concatenate(rows_season),
        "ml": np.concatenate(rows_ml),
        "edge": np.concatenate(rows_edge),
        "ev": np.concatenate(rows_ev),
        "units": np.concatenate(rows_units),
        "pnl": np.concatenate(rows_pnl),
        "won": np.concatenate(rows_won),
        "side": np.concatenate(rows_side),
        "fav": np.concatenate(rows_fav),
        "dog": np.concatenate(rows_dog),
        "slight_fav": np.concatenate(rows_slight),
        "big_dog": np.concatenate(rows_bigdog),
    }


def search_grid(
    data: dict[str, np.ndarray],
    *,
    source: str,
    prob_col: str,
    decor_w: float,
    window: str,
    min_bets: int,
    min_seasons: int,
    require_all_positive: bool,
) -> list[tuple[dict, dict]]:
    edges = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
    evs = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    bands = (
        (-150, 200),
        (-200, 200),
        (-150, 150),
        (-120, 120),
        (-180, 180),
        (-250, 250),
        (-350, 300),
        (-110, 110),
    )
    side_modes = (
        ("either", None, None),
        ("home", 0, None),
        ("away", 1, None),
        ("favorite", None, "fav"),
        ("dog", None, "dog"),
        ("slight_fav", None, "slight_fav"),
        ("big_dog", None, "big_dog"),
    )
    hits: list[tuple[dict, dict]] = []
    for min_edge in edges:
        for min_ev in evs:
            for ml_lo, ml_hi in bands:
                core = (
                    (data["edge"] >= min_edge)
                    & (data["ev"] >= min_ev)
                    & (data["ml"] >= ml_lo)
                    & (data["ml"] <= ml_hi)
                )
                if not core.any():
                    continue
                for sides, side_val, fav_key in side_modes:
                    mask = core
                    if side_val is not None:
                        mask = mask & (data["side"] == side_val)
                    if fav_key is not None:
                        mask = mask & data[fav_key]
                    res = summarize(
                        data["season"],
                        data["units"],
                        data["pnl"],
                        data["won"],
                        mask,
                        min_bets=min_bets,
                        min_seasons=min_seasons,
                        require_all_positive=require_all_positive,
                    )
                    if res is None:
                        continue
                    params = {
                        "bet_type": "moneyline",
                        "exec_price": "close",
                        "source_oos": source,
                        "prob_col": prob_col,
                        "decor_w": decor_w,
                        "min_edge_pp": float(min_edge),
                        "min_ev_pct": float(min_ev),
                        "ml_lo": int(ml_lo),
                        "ml_hi": int(ml_hi),
                        "sides": sides,
                        "season_window": window,
                        "hubacek": True,
                    }
                    hits.append((params, res))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-policy", action="store_true")
    parser.add_argument("--min-bets", type=int, default=40)
    parser.add_argument("--min-seasons", type=int, default=3)
    parser.add_argument(
        "--all-positive",
        action="store_true",
        help="Require every season in the window to be ROI>0 (stricter NHL-style gate)",
    )
    args = parser.parse_args()

    sources: list[tuple[str, Path]] = []
    hybrid = V2 / "oos_predictions_hybrid.csv"
    xgb = V2 / "oos_predictions.csv"
    if hybrid.is_file():
        sources.append(("hybrid", hybrid))
    if xgb.is_file():
        sources.append(("xgb", xgb))
    if not sources:
        raise SystemExit("no OOS prediction files found")

    windows = [
        ("all", None),
        ("from2018", 2018),
        ("from2020", 2020),
        ("from2022", 2022),
        ("from2023", 2023),
    ]
    decor_ws = (0.08, 0.12, 0.18, 0.25, 0.35, 0.50)
    # Prefer sport-only raw; also try blended for completeness
    prob_cols = ("model_raw", "model_prob")

    results: list[tuple[tuple, dict, dict]] = []
    for source, path in sources:
        base = load_oos(path, source)
        print(f"loaded {source} n={len(base)} seasons={sorted(base.season.unique().tolist())}", flush=True)
        for wname, lo in windows:
            frame = base if lo is None else base[base.season >= lo]
            if len(frame) < args.min_bets:
                continue
            for prob_col in prob_cols:
                if prob_col not in frame.columns:
                    continue
                for decor_w in decor_ws:
                    data = build_candidates(frame, prob_col, decor_w)
                    if data is None:
                        continue
                    hits = search_grid(
                        data,
                        source=source,
                        prob_col=prob_col,
                        decor_w=decor_w,
                        window=wname,
                        min_bets=args.min_bets,
                        min_seasons=args.min_seasons,
                        require_all_positive=args.all_positive,
                    )
                    for params, res in hits:
                        score = (res["worst_season_roi"], res["roi_pct"], res["bets"])
                        results.append((score, params, res))

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"\nqualifying policies: {len(results)}", flush=True)
    print("top Hubáček niches:", flush=True)
    for score, params, res in results[:25]:
        print(
            f"  worst={score[0]:7.2f} roi={score[1]:7.2f} bets={res['bets']:4d} "
            f"pos={res['seasons_positive']}/{res['seasons_total']} "
            f"{params['source_oos']} {params['season_window']} {params['prob_col']} "
            f"decor={params['decor_w']} gap>={params['min_edge_pp']} "
            f"ev>={params['min_ev_pct']} ml[{params['ml_lo']},{params['ml_hi']}] {params['sides']}",
            flush=True,
        )

    positives = [r for r in results if r[0][0] > 0]
    print(f"\npositive worst-season niches: {len(positives)}", flush=True)
    if not results:
        print("NO QUALIFYING POLICIES", flush=True)
        return 1

    best_score, best_params, best_res = results[0]
    if positives:
        best_score, best_params, best_res = positives[0]
    enabled = best_res["worst_season_roi"] > 0
    print("\nchosen:", json.dumps(best_params, indent=2), flush=True)
    print(json.dumps({k: v for k, v in best_res.items() if k != "per_season"}, indent=2), flush=True)

    out = V2 / "hubacek_search_summary.json"
    out.write_text(
        json.dumps(
            {
                "n_policies": len(results),
                "n_positive": len(positives),
                "best": {"params": best_params, "backtest": best_res},
                "top10": [
                    {"params": p, "worst": s[0], "roi": s[1], "bets": r["bets"]}
                    for s, p, r in results[:10]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)

    if args.write_policy:
        policy = {
            "stake_mode": "kelly",
            "kelly_fraction": KELLY_FRACTION,
            "enabled": enabled,
            "enabled_note": (
                "positive worst-season ROI (Hubáček decorrelated search)"
                if enabled
                else f"worst_season_roi={best_res['worst_season_roi']} -> KEEP DISABLED"
            ),
            "source": "search_nfl_hubacek.py",
            "hubacek_decorrelated": True,
            **best_params,
            "backtest": {k: v for k, v in best_res.items() if k != "per_season"},
            "backtest_per_season": best_res["per_season"],
        }
        path = V2 / "bet_policy.json"
        path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        print(f"wrote {path} enabled={enabled}", flush=True)

        pick_path = PROJECT_ROOT / "data" / "pick_strategy.json"
        pick = json.loads(pick_path.read_text(encoding="utf-8"))
        entry = pick.setdefault("nfl", {})
        entry["bet_type"] = "moneyline"
        entry["enabled"] = bool(enabled)
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
        if enabled:
            entry["note"] = (
                f"Hubáček NFL enable {date.today().isoformat()}: "
                f"{best_params['source_oos']} decor_w={best_params['decor_w']} "
                f"{best_params['prob_col']} gap>={best_params['min_edge_pp']} "
                f"EV>={best_params['min_ev_pct']}% ML[{best_params['ml_lo']},{best_params['ml_hi']}] "
                f"sides={sides} window={best_params['season_window']} "
                f"ROI {best_res['roi_pct']}% worst {best_res['worst_season_roi']}% "
                f"n={best_res['bets']}. ENABLED."
            )
        else:
            entry["note"] = (
                f"Hubáček NFL search {date.today().isoformat()}: best niche still "
                f"worst_season_roi={best_res['worst_season_roi']}% -> DISABLED."
            )
        pick["generated_at"] = date.today().isoformat()
        pick_path.write_text(json.dumps(pick, indent=2), encoding="utf-8")
        print(f"updated {pick_path} enabled={enabled}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
