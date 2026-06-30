"""Walk-forward backtest tuning for official pick thresholds (spread vs moneyline)."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import LEAGUE_PROFILES, eligible_for_official_picks  # noqa: E402
from web.pick_strategy import (  # noqa: E402
    STRATEGY_PATH,
    official_bet_type,
    tune_league_thresholds,
)

CORE_LEAGUES = (
    "nba",
    "wnba",
    "cbb",
    "nfl",
    "cfb",
    "nhl",
    "mlb",
    "ncaah",
    "ncaabb",
)

CATEGORY_ANCHORS = {
    "basketball": "nba",
    "football": "nfl",
    "hockey": "nhl",
    "baseball": "mlb",
}


def _cutoff_today() -> str:
    today = date.today()
    return f"{today.month}-{today.day}-{today.year}"


def main() -> int:
    cutoff = _cutoff_today()
    payload: dict[str, object] = {"default": None, "generated_at": date.today().isoformat()}

    tuned: dict[str, dict] = {}
    for league in CORE_LEAGUES:
        if league not in LEAGUE_PROFILES or not eligible_for_official_picks(league):
            continue
        print(f"Tuning {league} ({official_bet_type(league)})...", flush=True)
        entry = tune_league_thresholds(league, cutoff)
        tuned[league] = entry
        payload[league] = entry
        print(
            f"  bets={entry.get('backtest_bets')} "
            f"units={entry.get('backtest_units')} "
            f"roi={entry.get('backtest_roi_pct')}% "
            f"edge>={entry.get('min_edge')} "
            f"ev>={entry.get('min_ev_pct')} "
            f"pts>={entry.get('min_spread_point_edge')} "
            f"profit>={entry.get('min_profit_score')} "
            f"kelly>={entry.get('min_kelly_pct')}%",
            flush=True,
        )

    for category, anchor in CATEGORY_ANCHORS.items():
        if anchor in tuned:
            payload[category] = tuned[anchor]

    default_anchor = tuned.get("nba") or tuned.get("nhl") or next(iter(tuned.values()), None)
    if default_anchor:
        payload["default"] = default_anchor

    STRATEGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {STRATEGY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
