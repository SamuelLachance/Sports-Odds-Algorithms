"""Decorrelated Hubáček soccer 1X2 niche search (shared soccer_v2 OOS)."""

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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hubacek_binary_search import (  # noqa: E402
    KELLY_FRACTION,
    KELLY_CAP,
    KELLY_MIN,
    pick_best,
    summarize,
)

V2 = PROJECT_ROOT / "data" / "models" / "soccer_v2"
CLUB_LEAGUES = ("epl", "bundesliga", "laliga", "seriea", "ligue1")
LIVE_3WAY_DECOR = 0.15  # soccer_v2 live PICK_DECORRELATION_WEIGHT


def kelly_dec(prob: np.ndarray, dec: np.ndarray) -> np.ndarray:
    b = dec - 1.0
    edge = prob * dec - 1.0
    units = np.where((edge > 0) & (b > 0), edge / b * KELLY_FRACTION * 100.0, 0.0)
    units = np.clip(units, KELLY_MIN, KELLY_CAP)
    return np.where(edge > 0, units, 0.0)


def decorrelate_3way_vec(
    ph: np.ndarray,
    pd_: np.ndarray,
    pa: np.ndarray,
    mh: np.ndarray,
    md: np.ndarray,
    ma: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ah = np.clip(ph + weight * (ph - mh), 0.01, 0.99)
    ad = np.clip(pd_ + weight * (pd_ - md), 0.01, 0.99)
    aa = np.clip(pa + weight * (pa - ma), 0.01, 0.99)
    tot = ah + ad + aa
    return ah / tot, ad / tot, aa / tot


def build_candidates(
    frame: pd.DataFrame,
    *,
    exec_price: str,
    decor_w: float,
    prob_prefix: str = "model_prob",
) -> dict[str, np.ndarray] | None:
    h_col = f"{prob_prefix}_h" if f"{prob_prefix}_h" in frame.columns else "model_prob_h"
    d_col = f"{prob_prefix}_d" if f"{prob_prefix}_d" in frame.columns else "model_prob_d"
    a_col = f"{prob_prefix}_a" if f"{prob_prefix}_a" in frame.columns else "model_prob_a"
    if exec_price == "open":
        oh, od, oa = "open_home", "open_draw", "open_away"
    elif exec_price == "close":
        oh, od, oa = "close_home", "close_draw", "close_away"
    else:
        oh, od, oa = "max_home", "max_draw", "max_away"

    need = ["season", "result", h_col, d_col, a_col, oh, od, oa]
    if any(c not in frame.columns for c in need):
        return None
    sub = frame.dropna(subset=need).copy()
    if sub.empty:
        return None

    # result: H/D/A or 0/1/2
    res_raw = sub["result"]
    mapping = {"H": 0, "D": 1, "A": 2, "h": 0, "d": 1, "a": 2, "home": 0, "draw": 1, "away": 2}
    sample = str(res_raw.iloc[0]) if len(res_raw) else ""
    if sample in mapping or not pd.api.types.is_numeric_dtype(res_raw):
        label = res_raw.astype(str).map(mapping)
        if label.isna().any() and "label" in sub.columns and pd.api.types.is_numeric_dtype(sub["label"]):
            label = label.fillna(sub["label"])
        outcome = pd.to_numeric(label, errors="coerce").to_numpy(dtype=float)
    else:
        outcome = pd.to_numeric(res_raw, errors="coerce").to_numpy(dtype=float)

    ph = np.clip(sub[h_col].to_numpy(dtype=float), 0.01, 0.99)
    pd_ = np.clip(sub[d_col].to_numpy(dtype=float), 0.01, 0.99)
    pa = np.clip(sub[a_col].to_numpy(dtype=float), 0.01, 0.99)
    tot_p = ph + pd_ + pa
    ph, pd_, pa = ph / tot_p, pd_ / tot_p, pa / tot_p

    dh = sub[oh].to_numpy(dtype=float)
    dd = sub[od].to_numpy(dtype=float)
    da = sub[oa].to_numpy(dtype=float)
    valid = (dh > 1.01) & (dd > 1.01) & (da > 1.01) & np.isfinite(outcome)
    if not valid.any():
        return None
    ph, pd_, pa = ph[valid], pd_[valid], pa[valid]
    dh, dd, da = dh[valid], dd[valid], da[valid]
    outcome = outcome[valid]
    season = sub["season"].to_numpy(dtype=int)[valid]

    ih, id_, ia = 1.0 / dh, 1.0 / dd, 1.0 / da
    mt = ih + id_ + ia
    mh, md, ma = ih / mt, id_ / mt, ia / mt
    gh, gd, ga = decorrelate_3way_vec(ph, pd_, pa, mh, md, ma, decor_w)

    chunks: dict[str, list] = {
        k: [] for k in ("season", "dec", "edge", "ev", "units", "pnl", "won", "side", "outcome_name")
    }
    for side, p_ev, p_g, mkt, dec, won_flag, name in (
        (0, ph, gh, mh, dh, outcome == 0, "home"),
        (1, pd_, gd, md, dd, outcome == 1, "draw"),
        (2, pa, ga, ma, da, outcome == 2, "away"),
    ):
        edge_pp = (p_g - mkt) * 100.0
        ev_pct = (p_ev * dec - 1.0) * 100.0
        units = kelly_dec(p_ev, dec)
        take = (units > 0) & np.isfinite(edge_pp) & np.isfinite(ev_pct)
        if not take.any():
            continue
        n = int(take.sum())
        chunks["season"].append(season[take])
        chunks["dec"].append(dec[take])
        chunks["edge"].append(edge_pp[take])
        chunks["ev"].append(ev_pct[take])
        chunks["units"].append(units[take])
        won = won_flag[take]
        chunks["won"].append(won.astype(bool))
        chunks["pnl"].append(np.where(won, units[take] * (dec[take] - 1.0), -units[take]))
        chunks["side"].append(np.full(n, side, dtype=int))
        chunks["outcome_name"].append(np.full(n, name))

    if not chunks["season"]:
        return None
    return {k: np.concatenate(v) for k, v in chunks.items()}


def search_grid(
    data: dict[str, np.ndarray],
    *,
    decor_w: float,
    window: str,
    exec_price: str,
    min_bets: int,
    min_seasons: int,
) -> list[tuple[dict, dict]]:
    gaps = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    evs = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    max_odds = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
    outcomes = (
        ("home", 0),
        ("draw", 1),
        ("away", 2),
        ("home_away", None),  # special: sides 0 or 2
        ("any", None),
    )
    hits: list[tuple[dict, dict]] = []
    for min_gap in gaps:
        for min_ev in evs:
            for max_dec in max_odds:
                core = (
                    (data["edge"] >= min_gap)
                    & (data["ev"] >= min_ev)
                    & (data["dec"] <= max_dec)
                )
                if not core.any():
                    continue
                for oname, oval in outcomes:
                    if oname == "home_away":
                        mask = core & ((data["side"] == 0) | (data["side"] == 2))
                        allowed = ["home", "away"]
                    elif oname == "any":
                        mask = core
                        allowed = ["home", "draw", "away"]
                    else:
                        mask = core & (data["side"] == oval)
                        allowed = [oname]
                    res = summarize(
                        data["season"],
                        data["units"],
                        data["pnl"],
                        data["won"],
                        mask,
                        min_bets=min_bets,
                        min_seasons=min_seasons,
                        require_all_positive=True,
                    )
                    if res is None or res["worst_season_roi"] <= 0:
                        continue
                    params = {
                        "bet_type": "soccer_1x2",
                        "exec_price": exec_price,
                        "decor_w": decor_w,
                        "min_gap_pp": float(min_gap),
                        "min_edge_pp": float(min_gap),
                        "min_ev_pct": float(min_ev),
                        "max_decimal_odds": float(max_dec),
                        "outcomes": allowed,
                        "season_window": window,
                        "hubacek": True,
                        "model_variant": "market_aware",
                    }
                    hits.append((params, res))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-policy", action="store_true")
    parser.add_argument("--min-bets", type=int, default=60)
    parser.add_argument("--min-seasons", type=int, default=3)
    args = parser.parse_args()

    path = V2 / "oos_predictions_hybrid.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    frame = pd.read_csv(path)
    # Prefer calendar year from date if season is weird
    print(
        f"loaded soccer n={len(frame)} seasons={sorted(frame.season.unique().tolist())[:8]}...",
        flush=True,
    )

    windows = (
        ("from2018", 2018),
        ("from2020", 2020),
        ("from2022", 2022),
        ("from2023", 2023),
    )
    decor_ws = (0.10, 0.15, 0.20, 0.28, 0.35)
    exec_prices = ("open", "close", "max")

    results: list[tuple[tuple, dict, dict]] = []
    for wname, lo in windows:
        base = frame[frame.season >= lo]
        if len(base) < args.min_bets:
            continue
        for exec_price in exec_prices:
            for decor_w in decor_ws:
                for prefix in ("model_prob", "cal", "calm"):
                    data = build_candidates(
                        base, exec_price=exec_price, decor_w=decor_w, prob_prefix=prefix
                    )
                    if data is None:
                        continue
                    hits = search_grid(
                        data,
                        decor_w=decor_w,
                        window=wname,
                        exec_price=exec_price,
                        min_bets=args.min_bets,
                        min_seasons=args.min_seasons,
                    )
                    for params, res in hits:
                        params["prob_prefix"] = prefix
                        score = (res["worst_season_roi"], res["roi_pct"], res["bets"])
                        results.append((score, params, res))

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"positive soccer niches: {len(results)}", flush=True)
    for score, params, res in results[:15]:
        print(
            f"  worst={score[0]:7.2f} roi={score[1]:7.2f} bets={res['bets']:4d} "
            f"{params['season_window']} decor={params['decor_w']} "
            f"gap>={params['min_gap_pp']} ev>={params['min_ev_pct']} "
            f"maxodds={params['max_decimal_odds']} {params['outcomes']} "
            f"exec={params['exec_price']} {params.get('prob_prefix')}",
            flush=True,
        )

    # Prefer live-ish decor weight 0.15
    scored = []
    for score, params, res in results:
        live = 1 if abs(float(params["decor_w"]) - LIVE_3WAY_DECOR) < 1e-9 else 0
        ranked = (
            live,
            min(int(res["bets"]), 300),
            float(params["min_gap_pp"]),
            res["worst_season_roi"],
            float(params["min_ev_pct"]),
            res["roi_pct"],
        )
        scored.append((ranked, params, res))
    scored.sort(key=lambda x: x[0], reverse=True)

    summary = {"n_positive": len(results), "best": None, "top10": []}
    if scored:
        params, res = scored[0][1], scored[0][2]
        summary["best"] = {"params": params, "backtest": res}
        summary["top10"] = [
            {"params": p, "worst": r["worst_season_roi"], "roi": r["roi_pct"], "bets": r["bets"]}
            for _, p, r in scored[:10]
        ]
        print("chosen:", json.dumps(params, indent=2), flush=True)
        print(json.dumps({k: v for k, v in res.items() if k != "per_season"}, indent=2), flush=True)
    else:
        print("NO positive soccer niche", flush=True)

    (V2 / "hubacek_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.write_policy:
        pick_path = PROJECT_ROOT / "data" / "pick_strategy.json"
        pick = json.loads(pick_path.read_text(encoding="utf-8"))
        if scored:
            params, res = scored[0][1], scored[0][2]
            enabled = res["worst_season_roi"] > 0
            policy = {
                "stake_mode": "kelly",
                "kelly_fraction": KELLY_FRACTION,
                "enabled": enabled,
                "enabled_note": (
                    "positive worst-season ROI (Hubáček 3-way decorrelated)"
                    if enabled
                    else f"worst_season_roi={res['worst_season_roi']} -> KEEP DISABLED"
                ),
                "source": "search_soccer_hubacek.py",
                "hubacek_decorrelated": True,
                **params,
                "backtest": {k: v for k, v in res.items() if k != "per_season"},
                "backtest_per_season": res["per_season"],
            }
            (V2 / "bet_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
            for league in CLUB_LEAGUES:
                entry = pick.setdefault(league, {})
                entry["bet_type"] = "soccer_1x2"
                entry["enabled"] = bool(enabled)
                entry["min_market_gap_pp"] = float(params["min_gap_pp"])
                entry["min_ev_pct"] = float(params["min_ev_pct"])
                entry["min_win_confidence_pp"] = 0.0
                entry["allowed_sides"] = list(params["outcomes"])
                entry["backtest_roi_pct"] = res["roi_pct"]
                entry["backtest_bets"] = res["bets"]
                status = "ENABLED" if enabled else "DISABLED"
                entry["note"] = (
                    f"Hubáček soccer {date.today().isoformat()}: decor_w={params['decor_w']} "
                    f"gap>={params['min_gap_pp']} EV>={params['min_ev_pct']}% "
                    f"outcomes={params['outcomes']} exec={params['exec_price']} "
                    f"maxodds={params['max_decimal_odds']} window={params['season_window']} "
                    f"ROI {res['roi_pct']}% worst {res['worst_season_roi']}% n={res['bets']}. {status}."
                )
            pick["generated_at"] = date.today().isoformat()
            pick_path.write_text(json.dumps(pick, indent=2), encoding="utf-8")
            print(f"wrote soccer policy enabled={enabled}", flush=True)
        else:
            # keep disabled
            pol_path = V2 / "bet_policy.json"
            if pol_path.is_file():
                pol = json.loads(pol_path.read_text(encoding="utf-8"))
                pol["enabled"] = False
                pol["enabled_note"] = "Hubáček 3-way search: no worst_season_roi>0 niche"
                pol["hubacek_decorrelated"] = True
                pol_path.write_text(json.dumps(pol, indent=2), encoding="utf-8")
            for league in CLUB_LEAGUES:
                entry = pick.setdefault(league, {})
                entry["enabled"] = False
                entry["note"] = (
                    f"Hubáček soccer {date.today().isoformat()}: no positive worst-season niche. DISABLED."
                )
            pick["generated_at"] = date.today().isoformat()
            pick_path.write_text(json.dumps(pick, indent=2), encoding="utf-8")
            print("kept soccer club leagues disabled", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
