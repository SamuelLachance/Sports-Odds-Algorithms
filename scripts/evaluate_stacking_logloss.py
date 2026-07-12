"""Report walk-forward log-loss for meta stacking and EnsembleML per league."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.ensemble_ml.config import (  # noqa: E402
    SPORTSBOOK_LOGLOSS_BENCHMARKS,
    TRAIN_LEAGUES,
    get_dataset_profile,
    metadata_path,
)
from web.league_profiles import LEAGUE_PROFILES, is_soccer_league  # noqa: E402
from web.sports_meta_model import META_WEIGHTS_PATH  # noqa: E402
from web.soccer_meta_model import META_WEIGHTS_PATH as SOCCER_META_PATH  # noqa: E402
from web.tracking_service import toronto_cutoff_mdy, toronto_today  # noqa: E402


def _cutoff_today() -> str:
    return toronto_cutoff_mdy()


def _category(league: str) -> str:
    return LEAGUE_PROFILES[league.lower()]["category"]


def _meta_entry(league: str) -> dict | None:
    if is_soccer_league(league):
        path = SOCCER_META_PATH
    else:
        path = META_WEIGHTS_PATH
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get(league) or payload.get("default")


def _ensemble_entry(league: str) -> dict | None:
    path = metadata_path(league.lower())
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_report(cutoff: str) -> dict:
    from web.supplemental_games import load_supplemental_dated_games_for_backtest
    from web.season_games import load_league_dated_games_for_backtest

    rows: list[dict] = []
    for league in sorted(TRAIN_LEAGUES):
        profile = get_dataset_profile(league)
        if profile.get("dated_source") == "supplemental":
            dated = load_supplemental_dated_games_for_backtest(league, cutoff)
        else:
            dated = load_league_dated_games_for_backtest(league, cutoff)

        meta = _meta_entry(league)
        ensemble = _ensemble_entry(league)

        row = {
            "league": league,
            "category": _category(league),
            "games_available": len(dated),
            "dataset_profile": profile,
            "meta_log_loss_tuned": meta.get("log_loss_tuned") if meta else None,
            "meta_log_loss_baseline": meta.get("log_loss_baseline_equal") if meta else None,
            "meta_calibration_samples": meta.get("samples") if meta else None,
            "meta_blend": meta.get("blend") if meta else None,
            "meta_temperature": meta.get("temperature") if meta else None,
            "ensemble_train_rows": ensemble.get("train_rows") if ensemble else None,
            "ensemble_holdout_logloss": ensemble.get("holdout_logloss") if ensemble else None,
            "ensemble_closing_market_logloss": (
                ensemble.get("closing_market_logloss")
                if ensemble
                else None
            ),
            "sportsbook_benchmark_logloss": SPORTSBOOK_LOGLOSS_BENCHMARKS.get(league),
            "ensemble_beats_market": ensemble.get("beats_market") if ensemble else None,
            "ensemble_trained_at": ensemble.get("trained_at") if ensemble else None,
        }
        rows.append(row)
    return {
        "generated_at": toronto_today().isoformat(),
        "cutoff": cutoff,
        "leagues": rows,
    }


def _print_table(report: dict) -> None:
    print(
        f"\n{'League':<12} {'Cat':<10} {'Games':>7} "
        f"{'Meta LL':>8} {'Meta N':>7} "
        f"{'Ens Rows':>9} {'Ens LL':>8} {'Mkt LL':>8} {'Bench':>8}"
    )
    print("-" * 92)
    for row in report["leagues"]:
        market_ll = (
            row.get("ensemble_closing_market_logloss")
            or row.get("sportsbook_benchmark_logloss")
        )
        print(
            f"{row['league']:<12} {row['category']:<10} {row['games_available']:>7} "
            f"{row['meta_log_loss_tuned'] or '—':>8} "
            f"{row['meta_calibration_samples'] or '—':>7} "
            f"{row['ensemble_train_rows'] or '—':>9} "
            f"{row['ensemble_holdout_logloss'] or '—':>8} "
            f"{market_ll or '—':>8} "
            f"{row.get('sportsbook_benchmark_logloss') or '—':>8}"
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Log-loss report for meta stacking and EnsembleML models."
    )
    parser.add_argument("--cutoff", default=_cutoff_today())
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "data" / "supplemental" / "stacking_logloss_report.json"),
    )
    args = parser.parse_args()

    report = build_report(args.cutoff)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_table(report)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
