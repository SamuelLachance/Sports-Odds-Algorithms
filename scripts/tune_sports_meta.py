"""Tune binary home-win meta weights (log-loss) for all non-soccer sports."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.blend_service import _run_sport_pred_model  # noqa: E402
from web.league_profiles import LEAGUE_PROFILES, is_soccer_league  # noqa: E402
from web.power_model import build_power_ratings, predict_matchup  # noqa: E402
from web.season_games import load_league_completed_games  # noqa: E402
from web.sports_meta_model import (  # noqa: E402
    META_WEIGHTS_PATH,
    binary_log_loss,
    binary_temperature_scale,
    fit_binary_blend_weights_grid,
    fit_binary_temperature_grid,
)

CALIBRATION_WINDOW = 80
MIN_CALIBRATION_GAMES = 40
MIN_LEAGUE_GAMES = 60

CORE_LEAGUES = (
    "nba",
    "wnba",
    "cbb",
    "nhl",
    "mlb",
    "nfl",
    "cfb",
    "ncaah",
    "ncaabb",
)


def _cutoff_today() -> str:
    today = date.today()
    return f"{today.month}-{today.day}-{today.year}"


def _home_won(home_goals: int, away_goals: int) -> bool:
    return home_goals > away_goals


def _collect_samples(league: str, cutoff: str) -> list[tuple[float | None, float | None, float | None, bool]]:
    games = load_league_completed_games(league, cutoff)
    samples: list[tuple[float | None, float | None, float | None, bool]] = []
    start = max(40, len(games) - CALIBRATION_WINDOW)
    for index in range(start, len(games)):
        game = games[index]
        home, away, _hn, _an, home_goals, away_goals = game
        if home_goals == away_goals:
            continue
        teams, _total, param = build_power_ratings(games[:index])
        power = predict_matchup(teams, param, home, away) if param else None
        if not power:
            continue
        power_home = float(power["home_win_probability"])
        legacy_home = power_home
        _sport_key, sport_payload = _run_sport_pred_model(
            league, cutoff, home, away
        )
        sport_home = (
            float(sport_payload["home_win_probability"]) if sport_payload else None
        )
        samples.append(
            (legacy_home, power_home, sport_home, _home_won(home_goals, away_goals))
        )
    return samples


def tune_league(league: str, cutoff: str) -> dict | None:
    if is_soccer_league(league):
        return None

    games = load_league_completed_games(league, cutoff)
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    samples = _collect_samples(league, cutoff)
    if len(samples) < MIN_CALIBRATION_GAMES:
        return None

    has_sport = any(sample[2] is not None for sample in samples)
    two_layer = not has_sport

    if has_sport:
        blend_samples = [
            (legacy, power, sport, won)
            for legacy, power, sport, won in samples
            if sport is not None
        ]
    else:
        blend_samples = [(legacy, power, None, won) for legacy, power, _s, won in samples]

    blend_weights = fit_binary_blend_weights_grid(
        blend_samples,
        two_layer=two_layer,
    )

    def _blend_sample(
        legacy: float | None,
        power: float | None,
        sport: float | None,
        weights: dict[str, float],
    ) -> float:
        values: list[float] = []
        wts: list[float] = []
        for key, val in (("legacy", legacy), ("power", power), ("sport_pred", sport)):
            if val is None:
                continue
            weight = weights.get(key, 0.0)
            if weight <= 0:
                continue
            values.append(float(val))
            wts.append(weight)
        return sum(w * v for w, v in zip(wts, values)) / sum(wts)

    temp_samples = [
        (_blend_sample(legacy, power, sport, blend_weights), won)
        for legacy, power, sport, won in blend_samples
    ]
    temperature = fit_binary_temperature_grid(temp_samples)

    equal_loss = 0.0
    tuned_loss = 0.0
    for legacy, power, sport, won in blend_samples:
        if two_layer:
            equal = (float(legacy) + float(power)) / 2.0
        else:
            equal = (float(legacy) + float(power) + float(sport)) / 3.0
        equal_loss += binary_log_loss(equal, won)
        blended = _blend_sample(legacy, power, sport, blend_weights)
        tuned_loss += binary_log_loss(
            binary_temperature_scale(blended, temperature),
            won,
        )

    count = len(blend_samples)
    return {
        "blend": {key: round(value, 3) for key, value in blend_weights.items()},
        "temperature": round(temperature, 3),
        "two_layer": two_layer,
        "log_loss_baseline_equal": round(equal_loss / count, 4),
        "log_loss_tuned": round(tuned_loss / count, 4),
        "samples": count,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tune binary sports meta weights.")
    parser.add_argument(
        "--leagues",
        default=",".join(CORE_LEAGUES),
        help="Comma-separated league keys.",
    )
    parser.add_argument("--cutoff", default=_cutoff_today())
    args = parser.parse_args()
    leagues = [item.strip() for item in args.leagues.split(",") if item.strip()]
    payload: dict[str, dict] = {}
    tuned = 0

    for league in leagues:
        if league not in LEAGUE_PROFILES or is_soccer_league(league):
            continue
        print(f"Tuning {league}...", flush=True)
        entry = tune_league(league, args.cutoff)
        if entry:
            payload[league] = entry
            tuned += 1
            print(
                f"  blend={entry['blend']} temp={entry['temperature']} "
                f"loss {entry['log_loss_baseline_equal']} -> {entry['log_loss_tuned']}",
                flush=True,
            )
        else:
            print("  skipped", flush=True)

    if tuned:
        legacies = [entry["blend"].get("legacy", 0.05) for entry in payload.values()]
        powers = [entry["blend"].get("power", 0.15) for entry in payload.values()]
        sports = [entry["blend"].get("sport_pred", 0.0) for entry in payload.values()]
        temps = [entry["temperature"] for entry in payload.values()]
        payload["default"] = {
            "blend": {
                "legacy": round(sum(legacies) / len(legacies), 3),
                "power": round(sum(powers) / len(powers), 3),
                "sport_pred": round(sum(sports) / len(sports), 3) if any(sports) else 0.8,
            },
            "temperature": round(sum(temps) / len(temps), 3),
            "two_layer": False,
        }

    META_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_WEIGHTS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {META_WEIGHTS_PATH} ({tuned} leagues)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
