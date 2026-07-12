"""Vectorized all-seasons-positive policy search for paused leagues."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KELLY_F, KELLY_CAP, KELLY_MIN = 0.25, 3.0, 0.25


def am2dec(ml: np.ndarray) -> np.ndarray:
    ml = np.where(ml == 0, 100.0, ml.astype(float))
    out = np.full(ml.shape, np.nan)
    pos = ml > 0
    neg = ml < 0
    out[pos] = 1.0 + ml[pos] / 100.0
    out[neg] = 1.0 + 100.0 / np.abs(ml[neg])
    out[np.abs(ml) < 100] = np.nan
    return out


def kelly(prob: np.ndarray, ml: np.ndarray) -> np.ndarray:
    dec = am2dec(ml)
    b = dec - 1.0
    edge = prob * dec - 1.0
    units = np.where((edge > 0) & (b > 0), edge / b * KELLY_F * 100.0, 0.0)
    units = np.clip(units, KELLY_MIN, KELLY_CAP)
    units = np.where(edge > 0, units, 0.0)
    return units


def ncdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def summarize_mask(
    season: np.ndarray,
    units: np.ndarray,
    pnl: np.ndarray,
    won: np.ndarray,
    mask: np.ndarray,
    min_bps: int,
    min_seasons: int,
) -> dict | None:
    if not mask.any():
        return None
    s = season[mask]
    u = units[mask]
    p = pnl[mask]
    w = won[mask]
    seasons = np.unique(s)
    if len(seasons) < min_seasons:
        return None
    per = {}
    for se in seasons:
        m = s == se
        n = int(m.sum())
        if n < min_bps:
            return None
        st = float(u[m].sum())
        if st <= 0:
            return None
        roi = float(p[m].sum() / st * 100.0)
        if roi <= 0:
            return None
        per[int(se)] = {
            "bets": n,
            "roi_pct": round(roi, 2),
            "profit_units": round(float(p[m].sum()), 1),
        }
    st = float(u.sum())
    pr = float(p.sum())
    rois = [v["roi_pct"] for v in per.values()]
    return {
        "bets": int(len(u)),
        "staked_units": round(st, 1),
        "profit_units": round(pr, 1),
        "roi_pct": round(pr / st * 100.0, 2),
        "win_rate": round(float(w.mean()), 4),
        "seasons_positive": len(rois),
        "seasons_total": len(rois),
        "worst_season_roi": round(min(rois), 2),
        "median_season_roi": round(float(np.median(rois)), 2),
        "all_seasons_positive": True,
        "per_season": per,
    }


def build_nhl() -> dict[str, np.ndarray]:
    frame = pd.read_csv(ROOT / "data/models/nhl_v2/oos_predictions.csv")
    frame = frame.dropna(subset=["home_open_ml", "away_open_ml", "model_prob"]).copy()
    rows = []
    for prob_col in ["model_prob"] + (
        ["model_prob_mkt"] if "model_prob_mkt" in frame.columns else []
    ):
        for row in frame.itertuples(index=False):
            hm, am = float(row.home_open_ml), float(row.away_open_ml)
            if hm == 0:
                hm = 100.0
            if am == 0:
                am = 100.0
            if abs(hm) < 100 or abs(am) < 100:
                continue
            p = float(getattr(row, prob_col))
            if not math.isfinite(p):
                continue
            dh, da = am2dec(np.array([hm, am]))
            mh, ma = (1 / dh) / (1 / dh + 1 / da), (1 / da) / (1 / dh + 1 / da)
            for side, prob, mkt, ml in (
                ("home", p, mh, hm),
                ("away", 1 - p, ma, am),
            ):
                units = float(kelly(np.array([prob]), np.array([ml]))[0])
                if units <= 0:
                    continue
                won = (row.home_win == 1) if side == "home" else (row.home_win == 0)
                dec = float(am2dec(np.array([ml]))[0])
                pnl = units * (dec - 1) if won else -units
                rows.append(
                    (
                        int(row.season),
                        0 if prob_col == "model_prob" else 1,
                        0 if side == "home" else 1,
                        ml,
                        (prob - mkt) * 100,
                        (prob * dec - 1) * 100,
                        abs(prob * 100 - 50),
                        1 if ml < 0 else 0,
                        1 if ml > 0 else 0,
                        1 if (ml < 0 and abs(ml) <= 200) else 0,
                        1 if ml >= 150 else 0,
                        units,
                        pnl,
                        1 if won else 0,
                    )
                )
    arr = np.array(rows, dtype=float)
    return {
        "season": arr[:, 0].astype(int),
        "prob_col": arr[:, 1].astype(int),
        "side": arr[:, 2].astype(int),
        "ml": arr[:, 3],
        "edge": arr[:, 4],
        "ev": arr[:, 5],
        "conf": arr[:, 6],
        "fav": arr[:, 7].astype(bool),
        "dog": arr[:, 8].astype(bool),
        "slight_fav": arr[:, 9].astype(bool),
        "big_dog": arr[:, 10].astype(bool),
        "units": arr[:, 11],
        "pnl": arr[:, 12],
        "won": arr[:, 13].astype(bool),
    }


def search_nhl(data: dict, min_bps=30, min_seasons=3, top=25) -> list:
    hits = []
    edges = [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10]
    evs = [0, 1, 2, 3, 4, 5, 6]
    bands = [(-350, 300), (-250, 250), (-200, 200), (-180, 180), (-160, 160), (-150, 200)]
    sides = [("both", None), ("home", 0), ("away", 1)]
    favs = [
        ("any", None),
        ("favorite", "fav"),
        ("dog", "dog"),
        ("slight_fav", "slight_fav"),
        ("big_dog", "big_dog"),
    ]
    confs = [0, 5, 8, 10, 12, 15]
    for pc in (0, 1):
        base = data["prob_col"] == pc
        if not base.any():
            continue
        for min_edge in edges:
            for min_ev in evs:
                for ml_lo, ml_hi in bands:
                    core = (
                        base
                        & (data["edge"] >= min_edge)
                        & (data["ev"] >= min_ev)
                        & (data["ml"] >= ml_lo)
                        & (data["ml"] <= ml_hi)
                    )
                    if not core.any():
                        continue
                    for side_name, side_val in sides:
                        for fav_name, fav_key in favs:
                            for min_conf in confs:
                                mask = core & (data["conf"] >= min_conf)
                                if side_val is not None:
                                    mask = mask & (data["side"] == side_val)
                                if fav_key is not None:
                                    mask = mask & data[fav_key]
                                res = summarize_mask(
                                    data["season"],
                                    data["units"],
                                    data["pnl"],
                                    data["won"],
                                    mask,
                                    min_bps,
                                    min_seasons,
                                )
                                if res is None:
                                    continue
                                hits.append(
                                    (
                                        {
                                            "league": "nhl",
                                            "bet_type": "moneyline",
                                            "prob_col": "model_prob"
                                            if pc == 0
                                            else "model_prob_mkt",
                                            "exec_price": "open",
                                            "min_edge_pp": float(min_edge),
                                            "min_ev_pct": float(min_ev),
                                            "ml_lo": ml_lo,
                                            "ml_hi": ml_hi,
                                            "allowed_sides": side_name,
                                            "fav_mode": fav_name,
                                            "min_win_confidence_pp": float(min_conf),
                                        },
                                        res,
                                    )
                                )
    hits.sort(
        key=lambda x: (x[1]["worst_season_roi"], x[1]["roi_pct"], x[1]["bets"]),
        reverse=True,
    )
    return hits[:top]


def build_spread(league: str, exec_price: str) -> dict | None:
    path = ROOT / f"data/models/{league}_v2/oos_predictions.csv"
    frame = pd.read_csv(path)
    spread_col = "home_spread" if exec_price == "close" else "home_spread_open"
    margin_cols = ["model_margin"]
    if "model_margin_mkt" in frame.columns:
        margin_cols.append("model_margin_mkt")
    if spread_col not in frame.columns:
        return None
    need = ["margin", spread_col] + margin_cols[:1]
    frame = frame.dropna(subset=need).copy()
    has_elo = "elo_margin" in frame.columns
    rows = []
    for mi, margin_col in enumerate(margin_cols):
        if margin_col not in frame.columns:
            continue
        sub = frame.dropna(subset=[margin_col])
        sigma = float((sub["margin"] - sub[margin_col]).std(ddof=1))
        for row in sub.itertuples(index=False):
            spread = float(getattr(row, spread_col))
            model_m = float(getattr(row, margin_col))
            elo_m = float(getattr(row, "elo_margin", float("nan"))) if has_elo else float("nan")
            for side_i, juice_attr in ((0, "spread_home_odds"), (1, "spread_away_odds")):
                juice = getattr(row, juice_attr, None)
                # WNBA best_* fallback
                if (juice is None or pd.isna(juice)) and league == "wnba":
                    juice = getattr(
                        row,
                        "best_spread_home_odds" if side_i == 0 else "best_spread_away_odds",
                        None,
                    )
                if juice is None or pd.isna(juice):
                    continue
                juice = float(juice)
                if juice == 0:
                    juice = 100.0
                if abs(juice) < 100:
                    continue
                if side_i == 0:
                    pe = model_m - (-spread)
                    cover = float(ncdf(np.array([(model_m - (-spread)) / sigma]))[0])
                    laying = spread < 0
                    elo_ok = (not math.isfinite(elo_m)) or (elo_m > -spread)
                else:
                    pe = (-spread) - model_m
                    cover = float(ncdf(np.array([((-spread) - model_m) / sigma]))[0])
                    laying = spread > 0
                    elo_ok = (not math.isfinite(elo_m)) or (elo_m < -spread)
                units = float(kelly(np.array([cover]), np.array([juice]))[0])
                if units <= 0:
                    continue
                actual = float(row.margin)
                if side_i == 0:
                    won = actual + spread > 0
                    push = abs(actual + spread) < 1e-9
                else:
                    won = actual + spread < 0
                    push = abs(actual + spread) < 1e-9
                dec = float(am2dec(np.array([juice]))[0])
                pnl = 0.0 if push else (units * (dec - 1) if won else -units)
                be = 1.0 / dec
                rows.append(
                    (
                        int(row.season),
                        mi,
                        side_i,
                        pe,
                        (cover - be) * 100,
                        (cover * dec - 1) * 100,
                        abs(cover * 100 - 50),
                        1 if laying else 0,
                        1 if not laying else 0,
                        1 if elo_ok else 0,
                        units,
                        pnl,
                        1 if (won and not push) else 0,
                        sigma,
                    )
                )
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    return {
        "season": arr[:, 0].astype(int),
        "margin_i": arr[:, 1].astype(int),
        "side": arr[:, 2].astype(int),
        "point_edge": arr[:, 3],
        "gap": arr[:, 4],
        "ev": arr[:, 5],
        "conf": arr[:, 6],
        "fav": arr[:, 7].astype(bool),
        "dog": arr[:, 8].astype(bool),
        "elo_ok": arr[:, 9].astype(bool),
        "units": arr[:, 10],
        "pnl": arr[:, 11],
        "won": arr[:, 12].astype(bool),
        "sigma": float(arr[0, 13]),
        "margin_cols": margin_cols,
    }


def search_spread(
    data: dict,
    league: str,
    exec_price: str,
    min_bps: int,
    min_seasons: int,
    top: int = 25,
) -> list:
    hits = []
    gaps = list(np.arange(3.0, 18.5, 1.0))
    pts = list(np.arange(0.0, 8.5, 0.5))
    evs = [0, 1, 2, 2.5, 3, 4, 5, 6]
    sides = [("both", None), ("home", 0), ("away", 1)]
    favs = [("any", None), ("favorite", "fav"), ("dog", "dog")]
    confs = [0, 5, 8, 10, 12, 15]
    elo_opts = [False, True] if data["elo_ok"].any() and league in ("nfl", "cfb") else [False]
    for mi, mcol in enumerate(data["margin_cols"]):
        base = data["margin_i"] == mi
        if not base.any():
            continue
        for min_gap in gaps:
            for min_pt in pts:
                for min_ev in evs:
                    core = (
                        base
                        & (data["gap"] >= min_gap)
                        & (data["point_edge"] >= min_pt)
                        & (data["ev"] >= min_ev)
                    )
                    if not core.any():
                        continue
                    for side_name, side_val in sides:
                        for fav_name, fav_key in favs:
                            for min_conf in confs:
                                for elo_agree in elo_opts:
                                    mask = core & (data["conf"] >= min_conf)
                                    if side_val is not None:
                                        mask = mask & (data["side"] == side_val)
                                    if fav_key is not None:
                                        mask = mask & data[fav_key]
                                    if elo_agree:
                                        mask = mask & data["elo_ok"]
                                    res = summarize_mask(
                                        data["season"],
                                        data["units"],
                                        data["pnl"],
                                        data["won"],
                                        mask,
                                        min_bps,
                                        min_seasons,
                                    )
                                    if res is None:
                                        continue
                                    hits.append(
                                        (
                                            {
                                                "league": league,
                                                "bet_type": "spread",
                                                "margin_col": mcol,
                                                "exec_price": exec_price,
                                                "sigma": round(data["sigma"], 3),
                                                "min_cover_gap_pp": float(min_gap),
                                                "min_point_edge": float(min_pt),
                                                "min_ev_pct": float(min_ev),
                                                "allowed_sides": side_name,
                                                "fav_mode": fav_name,
                                                "min_spread_confidence_pp": float(min_conf),
                                                "elo_agree": elo_agree,
                                            },
                                            res,
                                        )
                                    )
    hits.sort(
        key=lambda x: (x[1]["worst_season_roi"], x[1]["roi_pct"], x[1]["bets"]),
        reverse=True,
    )
    return hits[:top]


def show(league: str, hits: list) -> None:
    print(f"\n=== {league.upper()} all-seasons-positive: {len(hits)} ===")
    if not hits:
        print("  (none)")
        return
    for p, r in hits[:12]:
        print(
            f"  worst={r['worst_season_roi']:+.2f} roi={r['roi_pct']:+.2f} "
            f"bets={r['bets']} seasons={r['seasons_total']} | {p}"
        )
        print(f"    {r['per_season']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="all")
    args = ap.parse_args()
    leagues = (
        ["nhl", "wnba", "nfl", "cfb"]
        if args.league == "all"
        else [args.league]
    )
    out: dict = {}
    for league in leagues:
        print(f"\nSearching {league}...", flush=True)
        if league == "nhl":
            data = build_nhl()
            print(f"  candidates={len(data['season'])}", flush=True)
            hits = search_nhl(data, min_bps=40, min_seasons=3)
        elif league == "wnba":
            data = build_spread("wnba", "open")
            if data is None:
                hits = []
            else:
                print(f"  candidates={len(data['season'])}", flush=True)
                hits = search_spread(data, "wnba", "open", min_bps=15, min_seasons=2)
        elif league == "nfl":
            data = build_spread("nfl", "close")
            print(f"  candidates={len(data['season']) if data else 0}", flush=True)
            hits = (
                search_spread(data, "nfl", "close", min_bps=10, min_seasons=8)
                if data
                else []
            )
        elif league == "cfb":
            hits = []
            for price in ("close", "open"):
                data = build_spread("cfb", price)
                if data is None:
                    continue
                print(f"  {price} candidates={len(data['season'])}", flush=True)
                hits.extend(
                    search_spread(
                        data,
                        "cfb",
                        price,
                        min_bps=20 if price == "close" else 15,
                        min_seasons=3,
                    )
                )
            hits.sort(
                key=lambda x: (x[1]["worst_season_roi"], x[1]["roi_pct"], x[1]["bets"]),
                reverse=True,
            )
            hits = hits[:25]
        else:
            hits = []
        show(league, hits)
        out[league] = [{"params": p, "result": r} for p, r in hits[:25]]
    path = ROOT / ".build-cache" / "all_seasons_positive.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
