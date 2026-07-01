"""Walk-forward backtest tuning for official pick thresholds (spread vs moneyline)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.league_profiles import (  # noqa: E402
    LEAGUE_PROFILES,
    SOCCER_LEAGUES,
    eligible_for_official_picks,
)
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

BASEBALL_LEAGUES = tuple(
    key for key, profile in LEAGUE_PROFILES.items() if profile["category"] == "baseball"
)

CATEGORY_ANCHORS = {
    "basketball": "nba",
    "football": "nfl",
    "hockey": "nhl",
    "baseball": "mlb",
    "soccer": "epl",
}


def _cutoff_today() -> str:
    today = date.today()
    return f"{today.month}-{today.day}-{today.year}"


def _leagues_for_categories(categories: set[str] | None) -> list[str]:
    if not categories:
        return [league for league in (*CORE_LEAGUES, *SOCCER_LEAGUES) if league in LEAGUE_PROFILES]

    leagues: list[str] = []
    if "baseball" in categories:
        leagues.extend(BASEBALL_LEAGUES)
    if "soccer" in categories:
        leagues.extend(SOCCER_LEAGUES)
    for league in CORE_LEAGUES:
        if categories.intersection({"basketball", "football", "hockey"}):
            profile = LEAGUE_PROFILES.get(league)
            if profile and profile["category"] in categories:
                leagues.append(league)
    seen: set[str] = set()
    ordered: list[str] = []
    for league in leagues:
        if league not in seen and league in LEAGUE_PROFILES:
            seen.add(league)
            ordered.append(league)
    return ordered


def _load_existing_payload() -> dict[str, object]:
    if not STRATEGY_PATH.is_file():
        return {}
    try:
        return json.loads(STRATEGY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune official pick strategy thresholds.")
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated categories to tune (baseball,soccer,...). Default: core + top soccer.",
    )
    args = parser.parse_args()
    categories = (
        {part.strip().lower() for part in args.categories.split(",") if part.strip()}
        if args.categories
        else None
    )
    leagues = _leagues_for_categories(categories)
    cutoff = _cutoff_today()
    payload: dict[str, object] = _load_existing_payload()
    payload["generated_at"] = date.today().isoformat()

    tuned: dict[str, dict] = {}
    for league in leagues:
        if not eligible_for_official_picks(league):
            continue
        print(f"Tuning {league} ({official_bet_type(league)})...", flush=True)
        entry = tune_league_thresholds(league, cutoff)
        tuned[league] = entry
        payload[league] = entry
        print(
            f"  pool={entry.get('backtest_games')} samples={entry.get('backtest_samples')} "
            f"bets={entry.get('backtest_bets')} "
            f"units={entry.get('backtest_units')} "
            f"roi={entry.get('backtest_roi_pct')}% "
            f"dd={entry.get('backtest_max_drawdown')} "
            f"enabled={entry.get('enabled')} "
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
        elif categories and category in categories and anchor in payload:
            payload[category] = payload[anchor]

    if not categories:
        default_anchor = tuned.get("nba") or tuned.get("nhl") or next(iter(tuned.values()), None)
        if default_anchor:
            payload["default"] = default_anchor

    STRATEGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {STRATEGY_PATH} ({len(tuned)} leagues tuned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
