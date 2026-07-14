"""Refresh Hubáček niches for MLB/NHL/CBB/NBA/WNBA with decorrelated gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hubacek_binary_search import run_binary_league_search  # noqa: E402

MODELS = PROJECT_ROOT / "data" / "models"


def load_league_frame(league: str) -> tuple[pd.DataFrame, str] | None:
    v2 = MODELS / f"{league}_v2"
    # Prefer hybrid OOS after feature expansion; open-price leagues still need
    # opening MLs (joined from training table when absent).
    prefer_open = league in ("mlb", "nhl", "nba", "wnba")
    candidates = [
        ("oos_predictions_hybrid.csv", "hybrid"),
        ("oos_predictions.csv", "xgb"),
        ("oos_predictions_improved.csv", "improved"),
    ]
    for name, label in candidates:
        path = v2 / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        # Normalize open ML aliases
        if "home_open_ml" not in frame.columns and "home_ml_open" in frame.columns:
            frame["home_open_ml"] = frame["home_ml_open"]
            frame["away_open_ml"] = frame["away_ml_open"]
        if "home_close_ml" not in frame.columns and "home_ml" in frame.columns:
            frame["home_close_ml"] = frame["home_ml"]
            frame["away_close_ml"] = frame["away_ml"]
        if "home_open_spread" not in frame.columns and "home_spread_open" in frame.columns:
            frame["home_open_spread"] = frame["home_spread_open"]
        if "home_close_spread" not in frame.columns and "home_spread" in frame.columns:
            frame["home_close_spread"] = frame["home_spread"]
        if "model_raw" not in frame.columns and "model_prob" in frame.columns:
            frame["model_raw"] = frame["model_prob"]
        # Join opening prices from training table when hybrid OOS lacks them.
        if prefer_open and (
            "home_open_ml" not in frame.columns
            or float(frame["home_open_ml"].notna().mean()) < 0.05
        ):
            hist = PROJECT_ROOT / "data" / f"{league}_history" / "training_table.csv"
            if hist.is_file():
                cols = ["season", "date", "home", "away", "home_open_ml", "away_open_ml"]
                tt = pd.read_csv(hist, usecols=lambda c: c in cols or c.endswith("_open_ml"))
                keys = [c for c in ("season", "date", "home", "away") if c in frame.columns and c in tt.columns]
                if keys and "home_open_ml" in tt.columns:
                    for drop in ("home_open_ml", "away_open_ml"):
                        if drop in frame.columns:
                            frame = frame.drop(columns=[drop])
                    frame = frame.merge(
                        tt[keys + ["home_open_ml", "away_open_ml"]],
                        on=keys,
                        how="left",
                    )
        if prefer_open and "home_open_ml" not in frame.columns:
            continue
        return frame, label
    return None


def should_upgrade(old_path: Path, new_res: dict, new_enabled: bool) -> bool:
    """Only overwrite if new niche clears and is not weaker on worst-season floor."""
    if not new_enabled:
        return False
    if not old_path.is_file():
        return True
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if not old.get("enabled"):
        return True
    old_bt = old.get("backtest") or {}
    old_worst = float(old_bt.get("worst_season_roi") or -999)
    old_bets = int(old_bt.get("bets") or 0)
    new_worst = float(new_res.get("worst_season_roi") or -999)
    new_bets = int(new_res.get("bets") or 0)
    # Upgrade if worst improves, or similar worst with more volume
    if new_worst > old_worst + 0.5:
        return True
    if new_worst >= old_worst - 0.25 and new_bets >= old_bets * 0.9:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-policy", action="store_true")
    parser.add_argument(
        "--leagues",
        default="mlb,nhl,cbb,nba,wnba",
        help="Comma-separated leagues",
    )
    args = parser.parse_args()

    configs = {
        "mlb": {
            "ml_prices": (("open", "home_open_ml", "away_open_ml"), ("close", "home_close_ml", "away_close_ml")),
            "also_spread": False,
            "windows": (("from2022", 2022), ("from2023", 2023), ("from2024", 2024), ("all", None)),
            "min_bets": 80,
        },
        "nhl": {
            "ml_prices": (("open", "home_open_ml", "away_open_ml"), ("close", "home_close_ml", "away_close_ml")),
            "also_spread": False,
            "windows": (("from2022", 2022), ("from2023", 2023), ("from2024", 2024), ("all", None)),
            "min_bets": 80,
        },
        "cbb": {
            "ml_prices": (("close", "home_close_ml", "away_close_ml"), ("open", "home_open_ml", "away_open_ml")),
            "also_spread": False,
            "windows": (("from2023", 2023), ("from2024", 2024), ("all", None)),
            "min_bets": 80,
        },
        "nba": {
            "ml_prices": (
                ("open", "home_open_ml", "away_open_ml"),
                ("close", "home_close_ml", "away_close_ml"),
            ),
            "also_spread": True,
            "windows": (("from2022", 2022), ("from2023", 2023), ("from2024", 2024), ("all", None)),
            "min_bets": 60,
        },
        "wnba": {
            "ml_prices": (
                ("open", "home_open_ml", "away_open_ml"),
                ("close", "home_close_ml", "away_close_ml"),
            ),
            "also_spread": True,
            "windows": (("from2022", 2022), ("from2023", 2023), ("from2024", 2024), ("all", None)),
            "min_bets": 60,
        },
    }

    for league in [x.strip() for x in args.leagues.split(",") if x.strip()]:
        cfg = configs.get(league)
        if not cfg:
            print(f"skip unknown league {league}", flush=True)
            continue
        loaded = load_league_frame(league)
        if loaded is None:
            print(f"[{league}] no OOS file", flush=True)
            continue
        frame, source = loaded
        # Ensure home_win
        if "home_win" not in frame.columns and "home_score" in frame.columns:
            frame["home_win"] = (frame["home_score"] > frame["away_score"]).astype(int)
        print(
            f"loaded {league} source={source} n={len(frame)} "
            f"seasons={sorted(frame.season.dropna().unique().tolist())[-8:]}",
            flush=True,
        )

        # Dry-run search without writing first when --write-policy wants upgrade check
        summary = run_binary_league_search(
            league=league,
            frame=frame,
            source=source,
            project_root=PROJECT_ROOT,
            write_policy=False,
            min_bets=cfg["min_bets"],
            min_seasons=3,
            require_all_positive=True,
            ml_prices=cfg["ml_prices"],
            windows=cfg["windows"],
            also_spread=cfg["also_spread"],
        )
        best = summary.get("best")
        if not best:
            print(f"[{league}] no positive niche found; leaving existing policy", flush=True)
            continue
        params, res = best["params"], best["backtest"]
        enabled = res["worst_season_roi"] > 0
        policy_path = MODELS / f"{league}_v2" / "bet_policy.json"
        if args.write_policy:
            if should_upgrade(policy_path, res, enabled):
                # re-run with write
                run_binary_league_search(
                    league=league,
                    frame=frame,
                    source=source,
                    project_root=PROJECT_ROOT,
                    write_policy=True,
                    min_bets=cfg["min_bets"],
                    min_seasons=3,
                    require_all_positive=True,
                    ml_prices=cfg["ml_prices"],
                    windows=cfg["windows"],
                    also_spread=cfg["also_spread"],
                )
                print(f"[{league}] UPGRADED policy", flush=True)
            else:
                print(
                    f"[{league}] keep existing (old stronger or equal). "
                    f"new worst={res['worst_season_roi']} bets={res['bets']}",
                    flush=True,
                )
                # Still stamp hubacek search summary
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
