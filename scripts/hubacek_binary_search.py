"""Shared vectorized Hubáček binary (moneyline/spread) niche search helpers."""

from __future__ import annotations

import json
from datetime import date
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KELLY_FRACTION = 0.25
KELLY_CAP = 3.0
KELLY_MIN = 0.25
LIVE_DECOR_W = 0.12


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


def build_ml_candidates(
    frame: pd.DataFrame,
    prob_col: str,
    decor_w: float,
    *,
    home_ml_col: str,
    away_ml_col: str,
) -> dict[str, np.ndarray] | None:
    if prob_col not in frame.columns:
        return None
    if home_ml_col not in frame.columns or away_ml_col not in frame.columns:
        return None
    hml = frame[home_ml_col].to_numpy(dtype=float)
    aml = frame[away_ml_col].to_numpy(dtype=float)
    p_raw = frame[prob_col].to_numpy(dtype=float)
    valid = (np.abs(hml) >= 100) & (np.abs(aml) >= 100) & np.isfinite(p_raw)
    if not valid.any():
        return None
    frame = frame.loc[valid].reset_index(drop=True)
    hml = frame[home_ml_col].to_numpy(dtype=float)
    aml = frame[away_ml_col].to_numpy(dtype=float)
    p = np.clip(frame[prob_col].to_numpy(dtype=float), 0.01, 0.99)
    dh, da = am2dec(hml), am2dec(aml)
    ih, ia = 1.0 / dh, 1.0 / da
    tot = ih + ia
    mkt_h, mkt_a = ih / tot, ia / tot

    p_gate = np.clip(p + decor_w * (p - mkt_h), 0.01, 0.99)
    pa = 1.0 - p
    pa_gate = np.clip(pa + decor_w * (pa - mkt_a), 0.01, 0.99)

    season = frame["season"].to_numpy(dtype=int)
    home_win = frame["home_win"].to_numpy(dtype=float)

    chunks: dict[str, list] = {
        k: []
        for k in (
            "season",
            "ml",
            "edge",
            "ev",
            "units",
            "pnl",
            "won",
            "side",
            "fav",
            "dog",
            "slight_fav",
            "big_dog",
        )
    }
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
        chunks["season"].append(season[take])
        chunks["ml"].append(ml[take])
        chunks["edge"].append(edge_pp[take])
        chunks["ev"].append(ev_pct[take])
        chunks["units"].append(units[take])
        won = won_flag[take]
        chunks["won"].append(won.astype(bool))
        chunks["pnl"].append(np.where(won, units[take] * (dec[take] - 1.0), -units[take]))
        chunks["side"].append(np.full(n, side, dtype=int))
        ml_t = ml[take]
        chunks["fav"].append(ml_t < 0)
        chunks["dog"].append(ml_t > 0)
        chunks["slight_fav"].append((ml_t < 0) & (ml_t >= -200))
        chunks["big_dog"].append(ml_t >= 150)

    if not chunks["season"]:
        return None
    return {k: np.concatenate(v) for k, v in chunks.items()}


def search_ml_grid(
    data: dict[str, np.ndarray],
    *,
    source: str,
    prob_col: str,
    decor_w: float,
    window: str,
    exec_price: str,
    min_bets: int,
    min_seasons: int,
    require_all_positive: bool,
) -> list[tuple[dict, dict]]:
    edges = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
    evs = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
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
                        "exec_price": exec_price,
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


def _ncdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(lambda x: erf(x / sqrt(2.0)))(z))


def build_spread_candidates(
    frame: pd.DataFrame,
    margin_col: str,
    *,
    spread_col: str,
    home_odds_col: str | None = None,
    away_odds_col: str | None = None,
    decor_w: float = LIVE_DECOR_W,
    sigma: float = 14.0,
) -> dict[str, np.ndarray] | None:
    need = ["season", "margin", margin_col, spread_col]
    if any(c not in frame.columns for c in need):
        return None
    sub = frame.dropna(subset=need).copy()
    if sub.empty:
        return None

    season = sub["season"].to_numpy(dtype=int)
    margin = sub["margin"].to_numpy(dtype=float)
    pred = sub[margin_col].to_numpy(dtype=float)
    line = sub[spread_col].to_numpy(dtype=float)
    p_home = np.clip(_ncdf((pred + line) / sigma), 0.01, 0.99)
    p_away = np.clip(_ncdf((-pred - line) / sigma), 0.01, 0.99)

    if (
        home_odds_col
        and home_odds_col in sub.columns
        and away_odds_col
        and away_odds_col in sub.columns
    ):
        hml = sub[home_odds_col].to_numpy(dtype=float)
        aml = sub[away_odds_col].to_numpy(dtype=float)
        hml = np.where(np.isfinite(hml) & (np.abs(hml) >= 100), hml, -110.0)
        aml = np.where(np.isfinite(aml) & (np.abs(aml) >= 100), aml, -110.0)
    else:
        hml = np.full(len(sub), -110.0)
        aml = np.full(len(sub), -110.0)
    dh, da = am2dec(hml), am2dec(aml)
    ih, ia = 1.0 / dh, 1.0 / da
    tot = ih + ia
    mkt_h, mkt_a = ih / tot, ia / tot

    p_home_g = np.clip(p_home + decor_w * (p_home - mkt_h), 0.01, 0.99)
    p_away_g = np.clip(p_away + decor_w * (p_away - mkt_a), 0.01, 0.99)
    home_cover = (margin + line) > 0
    away_cover = (margin + line) < 0
    not_push = (margin + line) != 0

    chunks: dict[str, list] = {
        k: []
        for k in ("season", "ml", "edge", "ev", "units", "pnl", "won", "side", "point_edge")
    }
    for side, p_ev, p_g, mkt, ml, covered, point_edge in (
        (0, p_home, p_home_g, mkt_h, hml, home_cover, pred + line),
        (1, p_away, p_away_g, mkt_a, aml, away_cover, -pred - line),
    ):
        edge_pp = (p_g - mkt) * 100.0
        dec = am2dec(ml)
        ev_pct = (p_ev * dec - 1.0) * 100.0
        units = kelly_units(p_ev, ml)
        take = not_push & (units > 0) & np.isfinite(edge_pp) & np.isfinite(ev_pct)
        if not take.any():
            continue
        n = int(take.sum())
        chunks["season"].append(season[take])
        chunks["ml"].append(ml[take])
        chunks["edge"].append(edge_pp[take])
        chunks["ev"].append(ev_pct[take])
        chunks["units"].append(units[take])
        won = covered[take]
        chunks["won"].append(won.astype(bool))
        chunks["pnl"].append(np.where(won, units[take] * (dec[take] - 1.0), -units[take]))
        chunks["side"].append(np.full(n, side, dtype=int))
        chunks["point_edge"].append(point_edge[take])

    if not chunks["season"]:
        return None
    return {k: np.concatenate(v) for k, v in chunks.items()}


def search_spread_grid(
    data: dict[str, np.ndarray],
    *,
    source: str,
    margin_col: str,
    decor_w: float,
    window: str,
    exec_price: str,
    min_bets: int,
    min_seasons: int,
    require_all_positive: bool,
) -> list[tuple[dict, dict]]:
    gaps = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
    confs = (0.0, 5.0, 8.0, 10.0, 12.0)
    point_edges = (0.0, 0.5, 1.0, 1.5, 2.0)
    evs = (0.0, 1.0, 2.0, 3.0)
    side_modes = (("either", None), ("home", 0), ("away", 1))
    cover_conf = np.abs(data["edge"]) / (1.0 + decor_w)
    hits: list[tuple[dict, dict]] = []
    for min_gap in gaps:
        for min_conf in confs:
            for min_pe in point_edges:
                for min_ev in evs:
                    core = (
                        (data["edge"] >= min_gap)
                        & (cover_conf >= min_conf)
                        & (data["point_edge"] >= min_pe)
                        & (data["ev"] >= min_ev)
                    )
                    if not core.any():
                        continue
                    for sides, side_val in side_modes:
                        mask = core if side_val is None else (core & (data["side"] == side_val))
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
                            "bet_type": "spread",
                            "exec_price": exec_price,
                            "source_oos": source,
                            "margin_col": margin_col,
                            "decor_w": decor_w,
                            "min_spread_cover_gap_pp": float(min_gap),
                            "min_spread_confidence_pp": float(min_conf),
                            "min_spread_point_edge": float(min_pe),
                            "min_ev_pct": float(min_ev),
                            "sides": sides,
                            "season_window": window,
                            "hubacek": True,
                        }
                        hits.append((params, res))
    return hits


def pick_best(
    results: list[tuple[tuple, dict, dict]],
    *,
    prefer_live_decor: bool = True,
) -> tuple[dict, dict] | None:
    if not results:
        return None
    scored: list[tuple] = []
    for _score, params, res in results:
        live = (
            1
            if prefer_live_decor and abs(float(params.get("decor_w", 0)) - LIVE_DECOR_W) < 1e-9
            else 0
        )
        gap = float(params.get("min_edge_pp") or params.get("min_spread_cover_gap_pp") or 0)
        ranked = (
            live,
            min(int(res["bets"]), 200),
            gap,
            res["worst_season_roi"],
            float(params.get("min_ev_pct") or 0),
            res["roi_pct"],
        )
        scored.append((ranked, params, res))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored[0][2]


def write_bet_policy(path: Path, params: dict, res: dict, *, enabled: bool) -> None:
    policy = {
        "stake_mode": "kelly",
        "kelly_fraction": KELLY_FRACTION,
        "enabled": enabled,
        "enabled_note": (
            "positive worst-season ROI (Hubáček decorrelated search)"
            if enabled
            else f"worst_season_roi={res.get('worst_season_roi')} -> KEEP DISABLED"
        ),
        "source": "hubacek_binary_search",
        "hubacek_decorrelated": True,
        **params,
        "backtest": {k: v for k, v in res.items() if k != "per_season"},
        "backtest_per_season": res.get("per_season", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")


def update_pick_strategy_ml(
    pick_path: Path,
    league: str,
    params: dict,
    res: dict,
    *,
    enabled: bool,
) -> None:
    pick = json.loads(pick_path.read_text(encoding="utf-8"))
    entry = pick.setdefault(league, {})
    entry["bet_type"] = "moneyline"
    entry["enabled"] = bool(enabled)
    entry["min_market_gap_pp"] = float(params["min_edge_pp"])
    entry["min_ev_pct"] = float(params["min_ev_pct"])
    entry["ml_lo"] = int(params["ml_lo"])
    entry["ml_hi"] = int(params["ml_hi"])
    entry["min_win_confidence_pp"] = 0.0
    sides = params["sides"]
    if sides in ("home", "away"):
        entry["allowed_sides"] = [sides]
        entry["fav_mode"] = "any"
    elif sides == "favorite":
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "favorite"
    elif sides in ("dog", "big_dog"):
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "dog"
    elif sides == "slight_fav":
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "favorite"
    else:
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "any"
    entry["backtest_roi_pct"] = res["roi_pct"]
    entry["backtest_bets"] = res["bets"]
    status = "ENABLED" if enabled else "DISABLED"
    entry["note"] = (
        f"Hubáček {league.upper()} {date.today().isoformat()}: "
        f"{params.get('source_oos')} {params.get('prob_col')} decor_w={params.get('decor_w')} "
        f"gap>={params['min_edge_pp']} EV>={params['min_ev_pct']}% "
        f"ML[{params['ml_lo']},{params['ml_hi']}] sides={sides} "
        f"exec={params.get('exec_price')} window={params.get('season_window')} "
        f"ROI {res['roi_pct']}% worst {res['worst_season_roi']}% n={res['bets']}. {status}."
    )
    pick["generated_at"] = date.today().isoformat()
    pick_path.write_text(json.dumps(pick, indent=2), encoding="utf-8")


def update_pick_strategy_spread(
    pick_path: Path,
    league: str,
    params: dict,
    res: dict,
    *,
    enabled: bool,
) -> None:
    pick = json.loads(pick_path.read_text(encoding="utf-8"))
    entry = pick.setdefault(league, {})
    entry["bet_type"] = "spread"
    entry["enabled"] = bool(enabled)
    entry["min_spread_cover_gap_pp"] = float(params["min_spread_cover_gap_pp"])
    entry["min_spread_confidence_pp"] = float(params["min_spread_confidence_pp"])
    entry["min_spread_point_edge"] = float(params["min_spread_point_edge"])
    entry["min_ev_pct"] = float(params["min_ev_pct"])
    sides = params["sides"]
    if sides in ("home", "away"):
        entry["allowed_sides"] = [sides]
    else:
        entry.pop("allowed_sides", None)
    entry["backtest_roi_pct"] = res["roi_pct"]
    entry["backtest_bets"] = res["bets"]
    status = "ENABLED" if enabled else "DISABLED"
    entry["note"] = (
        f"Hubáček {league.upper()} {date.today().isoformat()}: spread "
        f"cover gap>={params['min_spread_cover_gap_pp']} conf>={params['min_spread_confidence_pp']} "
        f"point_edge>={params['min_spread_point_edge']} EV>={params['min_ev_pct']}% "
        f"sides={sides} exec={params.get('exec_price')} window={params.get('season_window')} "
        f"ROI {res['roi_pct']}% worst {res['worst_season_roi']}% n={res['bets']}. {status}."
    )
    pick["generated_at"] = date.today().isoformat()
    pick_path.write_text(json.dumps(pick, indent=2), encoding="utf-8")


def run_binary_league_search(
    *,
    league: str,
    frame: pd.DataFrame,
    source: str,
    project_root: Path,
    write_policy: bool,
    min_bets: int = 40,
    min_seasons: int = 3,
    require_all_positive: bool = True,
    ml_prices: tuple[tuple[str, str, str], ...] = (("close", "home_close_ml", "away_close_ml"),),
    windows: tuple[tuple[str, int | None], ...] = (("all", None),),
    decor_ws: tuple[float, ...] = (0.08, 0.12, 0.18, 0.25, 0.35),
    prob_cols: tuple[str, ...] = ("model_raw", "model_prob"),
    also_spread: bool = False,
) -> dict[str, Any]:
    results: list[tuple[tuple, dict, dict]] = []
    for wname, lo in windows:
        base = frame if lo is None else frame[frame.season >= lo]
        if len(base) < min_bets:
            continue
        for exec_price, home_col, away_col in ml_prices:
            if home_col not in base.columns:
                continue
            for prob_col in prob_cols:
                if prob_col not in base.columns:
                    continue
                for decor_w in decor_ws:
                    data = build_ml_candidates(
                        base,
                        prob_col,
                        decor_w,
                        home_ml_col=home_col,
                        away_ml_col=away_col,
                    )
                    if data is None:
                        continue
                    hits = search_ml_grid(
                        data,
                        source=source,
                        prob_col=prob_col,
                        decor_w=decor_w,
                        window=wname,
                        exec_price=exec_price,
                        min_bets=min_bets,
                        min_seasons=min_seasons,
                        require_all_positive=require_all_positive,
                    )
                    for params, res in hits:
                        if res["worst_season_roi"] <= 0:
                            continue
                        score = (res["worst_season_roi"], res["roi_pct"], res["bets"])
                        results.append((score, params, res))

    spread_results: list[tuple[tuple, dict, dict]] = []
    if also_spread:
        for wname, lo in windows:
            base = frame if lo is None else frame[frame.season >= lo]
            for exec_price, spread_col in (
                ("close", "home_close_spread"),
                ("open", "home_open_spread"),
                ("close", "home_spread"),
            ):
                if spread_col not in base.columns:
                    continue
                for margin_col in ("model_margin", "model_margin_mkt", "pred_margin"):
                    if margin_col not in base.columns:
                        continue
                    for decor_w in decor_ws:
                        data = build_spread_candidates(
                            base,
                            margin_col,
                            spread_col=spread_col,
                            decor_w=decor_w,
                            home_odds_col="home_spread_odds",
                            away_odds_col="away_spread_odds",
                        )
                        if data is None:
                            continue
                        hits = search_spread_grid(
                            data,
                            source=source,
                            margin_col=margin_col,
                            decor_w=decor_w,
                            window=wname,
                            exec_price=exec_price,
                            min_bets=min_bets,
                            min_seasons=min_seasons,
                            require_all_positive=require_all_positive,
                        )
                        for params, res in hits:
                            if res["worst_season_roi"] <= 0:
                                continue
                            score = (res["worst_season_roi"], res["roi_pct"], res["bets"])
                            spread_results.append((score, params, res))

    all_pos = results + spread_results
    all_pos.sort(key=lambda x: x[0], reverse=True)
    print(
        f"[{league}] positive niches: ML={len(results)} spread={len(spread_results)}",
        flush=True,
    )
    for score, params, res in all_pos[:12]:
        print(
            f"  worst={score[0]:7.2f} roi={score[1]:7.2f} bets={res['bets']:4d} "
            f"{params.get('bet_type')} {params.get('season_window')} "
            f"decor={params.get('decor_w')} gap/edge="
            f"{params.get('min_edge_pp') or params.get('min_spread_cover_gap_pp')} "
            f"sides={params.get('sides')} exec={params.get('exec_price')}",
            flush=True,
        )

    chosen = pick_best(all_pos)
    summary: dict[str, Any] = {
        "league": league,
        "n_positive_ml": len(results),
        "n_positive_spread": len(spread_results),
        "best": None,
    }
    v2 = project_root / "data" / "models" / f"{league}_v2"
    pick_path = project_root / "data" / "pick_strategy.json"

    if chosen is None:
        print(f"[{league}] NO positive niche", flush=True)
        if write_policy:
            existing = v2 / "bet_policy.json"
            if existing.is_file():
                pol = json.loads(existing.read_text(encoding="utf-8"))
                pol["enabled"] = False
                pol["enabled_note"] = "Hubáček decorrelated search: no worst_season_roi>0 niche"
                pol["hubacek_decorrelated"] = True
                existing.write_text(json.dumps(pol, indent=2), encoding="utf-8")
        return summary

    params, res = chosen
    enabled = res["worst_season_roi"] > 0
    summary["best"] = {"params": params, "backtest": res}
    print(f"[{league}] chosen enabled={enabled} {json.dumps(params)}", flush=True)
    print(json.dumps({k: v for k, v in res.items() if k != "per_season"}, indent=2), flush=True)

    v2.mkdir(parents=True, exist_ok=True)
    (v2 / "hubacek_search_summary.json").write_text(
        json.dumps(
            {
                "n_positive_ml": len(results),
                "n_positive_spread": len(spread_results),
                "best": summary["best"],
                "top10": [
                    {"params": p, "worst": s[0], "roi": s[1], "bets": r["bets"]}
                    for s, p, r in all_pos[:10]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if write_policy:
        write_bet_policy(v2 / "bet_policy.json", params, res, enabled=enabled)
        if params["bet_type"] == "moneyline":
            update_pick_strategy_ml(pick_path, league, params, res, enabled=enabled)
        else:
            update_pick_strategy_spread(pick_path, league, params, res, enabled=enabled)
        print(f"[{league}] wrote policy enabled={enabled}", flush=True)
    return summary
