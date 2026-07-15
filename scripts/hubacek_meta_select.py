"""Season-by-season max-Sharpe meta-selection over Hubáček β configs (§7.3.1),
extended beyond the paper with the π-triangulated CLV filter, z-score
thresholding, and the β-ensemble candidate.

For each league: for every evaluated season s, pick the (candidate, setting)
with the best Sharpe on all PRIOR seasons' rounds, then realize that setting's
profit on season s. Candidates are the 8 decorrelated arms plus their
dispersion-gated ensemble; settings sweep confidence filters (φ grid and the
odds-aware z-score rule), the predicted-CLV triangulation filter (π close
predictor vs taken price), and the price source (close, plus open when open
moneylines exist). The setting choice itself is therefore out-of-sample.

Writes data/models/{league}_hybrid/hubacek_meta.json.

Usage:
  python scripts/hubacek_meta_select.py --leagues nba
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.hubacek_v2.eval import (  # noqa: E402
    attach_pi_probs,
    build_ensemble_frame,
    evaluate_rounds,
)
from web.hybrid_v2.train import hubacek_round_key  # noqa: E402

# Confidence settings: (threshold_mode, phi, zscore_k)
CONF_SETTINGS: tuple[tuple[str, float, float], ...] = (
    ("phi", 0.0, 0.0),
    ("phi", 0.1, 0.0),
    ("phi", 0.15, 0.0),
    ("phi", 0.2, 0.0),
    ("zscore", 0.0, 0.5),
    ("zscore", 0.0, 1.0),
    ("zscore", 0.0, 1.5),
)


def _load_beta_oos(league: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in glob.glob(str(PROJECT_ROOT / "data" / "models" / f"{league}_hybrid" / "oos_cb_dec*.csv")):
        name = Path(path).stem.replace("oos_", "")
        frame = pd.read_csv(path, low_memory=False)
        frame["hub_round"] = hubacek_round_key(league, frame)
        out[name] = frame
    return out


def _load_pi_oos(league: str) -> pd.DataFrame | None:
    """The shipped π winner's OOS (close predictor) for CLV triangulation."""
    hybrid_dir = PROJECT_ROOT / "data" / "models" / f"{league}_hybrid"
    best_path = hybrid_dir / "best_config.json"
    if not best_path.is_file():
        return None
    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
        name = (best.get("config") or {}).get("name")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
        path = hybrid_dir / f"oos_{safe}.csv"
        if not path.is_file():
            return None
        return pd.read_csv(path, low_memory=False)
    except (json.JSONDecodeError, OSError):
        return None


def _eval(frame: pd.DataFrame, *, price: str, conf: tuple[str, float, float], clv: bool):
    mode, phi, k = conf
    return evaluate_rounds(
        frame,
        price=price,
        phi=phi,
        threshold_mode=mode,
        zscore_k=k,
        strategy="opt",
        round_key="hub_round",
        pi_col="pi_prob" if clv else None,
    )


def meta_select(league: str) -> dict | None:
    arms = _load_beta_oos(league)
    if not arms:
        print(f"[{league}] no decorrelated OOS files - run train_hubacek_arms first")
        return None
    ensemble = build_ensemble_frame(arms)
    candidates: dict[str, pd.DataFrame] = dict(arms)
    if ensemble is not None:
        ensemble["hub_round"] = hubacek_round_key(league, ensemble)
        candidates["ensemble"] = ensemble
    pi_oos = _load_pi_oos(league)
    if pi_oos is not None:
        candidates = {
            name: attach_pi_probs(frame, pi_oos) for name, frame in candidates.items()
        }
    clv_options = (False, True) if pi_oos is not None else (False,)
    has_open = any(
        {"home_open_ml", "away_open_ml"}.issubset(f.columns) and f["home_open_ml"].notna().any()
        for f in candidates.values()
    )
    prices = ("close", "open") if has_open else ("close",)

    seasons = sorted(
        {int(s) for frame in candidates.values() for s in frame["season"].dropna().unique()}
    )
    realized: list[dict] = []
    for i, season in enumerate(seasons):
        prior = seasons[:i]
        if not prior:
            continue
        best = None
        for cand_name, frame in candidates.items():
            hist = frame[frame["season"].isin(prior)]
            if hist.empty:
                continue
            for price in prices:
                for conf in CONF_SETTINGS:
                    for clv in clv_options:
                        res = _eval(hist, price=price, conf=conf, clv=clv)
                        if not res or res.get("sharpe_per_round") is None:
                            continue
                        if res.get("n_bets", 0) < 30 * max(len(prior), 1) // 2:
                            continue  # starve-proof: enough historical bets
                        if best is None or res["sharpe_per_round"] > best["sharpe"]:
                            best = {
                                "candidate": cand_name,
                                "price": price,
                                "conf": conf,
                                "clv": clv,
                                "sharpe": res["sharpe_per_round"],
                            }
        if best is None:
            continue
        cur = candidates[best["candidate"]]
        season_frame = cur[cur["season"] == season]
        res = _eval(season_frame, price=best["price"], conf=best["conf"], clv=best["clv"])
        if res:
            mode, phi, k = best["conf"]
            realized.append(
                {
                    "season": int(season),
                    "candidate": best["candidate"],
                    "price": best["price"],
                    "threshold_mode": mode,
                    "phi": phi,
                    "zscore_k": k,
                    "clv_filter": best["clv"],
                    "mean_profit_per_round_pct": res["mean_profit_per_round_pct"],
                    "sharpe_per_round": res["sharpe_per_round"],
                    "n_rounds": res["n_rounds"],
                    "n_bets": res["n_bets"],
                    "rho_model_book": res["rho_model_book"],
                    "cumulative_profit_units": res["cumulative_profit_units"],
                    "quadrants": res.get("quadrants"),
                }
            )
    if not realized:
        print(f"[{league}] not enough seasons for meta-selection")
        return None
    profits = [r["mean_profit_per_round_pct"] for r in realized if r["mean_profit_per_round_pct"] is not None]
    payload = {
        "league": league,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "per-season max-Sharpe selection on prior rounds (Hubacek 2019 7.3.1), "
            "extended with pi-triangulated predicted-CLV filter, odds-aware z-score "
            "thresholding, and the dispersion-gated beta ensemble; realized next season"
        ),
        "seasons": realized,
        "mean_profit_per_round_pct": round(float(np.mean(profits)), 3) if profits else None,
        "worst_season_profit_pct": round(float(np.min(profits)), 3) if profits else None,
        "positive_seasons": int(sum(1 for p in profits if p > 0)),
        "total_seasons": len(profits),
    }
    out = PROJECT_ROOT / "data" / "models" / f"{league}_hybrid" / "hubacek_meta.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"[{league}] meta: mean {payload['mean_profit_per_round_pct']}%/round, "
        f"worst {payload['worst_season_profit_pct']}%, "
        f"{payload['positive_seasons']}/{payload['total_seasons']} seasons positive"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leagues", default="nba")
    args = parser.parse_args()
    for league in [s.strip().lower() for s in args.leagues.split(",") if s.strip()]:
        try:
            meta_select(league)
        except Exception as exc:  # noqa: BLE001
            print(f"[{league}] FAILED: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
