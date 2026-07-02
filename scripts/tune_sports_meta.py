"""Tune binary home-win meta weights (log-loss) for all non-soccer sports."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.blend_service import _run_sport_pred_model  # noqa: E402
from web.league_profiles import (  # noqa: E402
    BACKTEST_PRO_SEASONS,
    BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
    LEAGUE_PROFILES,
    is_soccer_league,
)
from web.backtest_algo_v2 import TeamHistoryBuilder, legacy_home_win_probability  # noqa: E402
from web.ensemble_ml.config import MIN_TRAIN_ROWS, get_dataset_profile  # noqa: E402
from web.power_model import build_power_ratings, predict_matchup  # noqa: E402
from web.supplemental_games import load_supplemental_dated_games_for_backtest  # noqa: E402
from web.walkforward import calibration_indices  # noqa: E402
from web.sports_meta_model import (  # noqa: E402
    META_WEIGHTS_PATH,
    binary_log_loss,
    binary_temperature_scale,
    fit_binary_blend_weights_grid,
    fit_binary_temperature_grid,
)

CALIBRATION_WINDOW = None  # full walk-forward after warmup
MIN_CALIBRATION_GAMES = 40
MIN_LEAGUE_GAMES = 60

ALL_BINARY_LEAGUES = tuple(
    key for key in LEAGUE_PROFILES if not is_soccer_league(key)
)

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


def _data_depth_metadata(game_count: int, calibration_games: int) -> dict[str, int]:
    return {
        "games_loaded": game_count,
        "calibration_games": calibration_games,
        "pro_seasons": BACKTEST_PRO_SEASONS,
        "scoreboard_days": BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
        "supplemental": 1,
        "full_walkforward": 1,
    }


def _home_won(home_goals: int, away_goals: int) -> bool:
    return home_goals > away_goals


def _load_dated_games(league: str, cutoff: str) -> list:
    profile = get_dataset_profile(league)
    if profile.get("dated_source") == "supplemental":
        return load_supplemental_dated_games_for_backtest(league, cutoff)
    from web.season_games import load_league_dated_games_for_backtest

    return load_league_dated_games_for_backtest(league, cutoff)


def _collect_samples(
    league: str,
    cutoff: str,
    dated_games: list[tuple[str, tuple]],
) -> list[tuple[float | None, float | None, float | None, bool]]:
    samples: list[tuple[float | None, float | None, float | None, bool]] = []
    warmup = MIN_TRAIN_ROWS // 2
    profile = get_dataset_profile(league)
    indices = list(
        calibration_indices(
            len(dated_games),
            warmup=warmup,
            max_calibration_games=profile.get("max_calibration_games"),
            target_rows=profile.get("target_rows"),
        )
    )
    print(f"  walk-forward samples: {len(indices)} of {len(dated_games)} games", flush=True)

    history = TeamHistoryBuilder(league)
    cursor = 0
    for index in indices:
        while cursor < index:
            game_date, game = dated_games[cursor]
            home, away, _hn, _an, home_goals, away_goals = game
            if home_goals != away_goals:
                history.add_game(game_date, home, away, home_goals, away_goals)
            cursor += 1

        game_date, game = dated_games[index]
        home, away, _hn, _an, home_goals, away_goals = game
        if home_goals == away_goals:
            continue

        train = [g for _d, g in dated_games[:index]]
        teams, _total, param = build_power_ratings(train)
        power = predict_matchup(teams, param, home, away) if param else None
        if not power:
            continue
        power_home = float(power["home_win_probability"])

        legacy_home = legacy_home_win_probability(
            league,
            history,
            home_abbr=home,
            away_abbr=away,
            home_team=[home, home],
            away_team=[away, away],
            game_date_iso=game_date,
        )
        if legacy_home is None:
            legacy_home = power_home

        _sport_key, sport_payload = _run_sport_pred_model(league, cutoff, home, away)
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

    dated_games = _load_dated_games(league, cutoff)
    if len(dated_games) < MIN_LEAGUE_GAMES:
        return None

    samples = _collect_samples(league, cutoff, dated_games)
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
        max_sport_step=10,
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
    temp_range = (10, 22) if league == "mlb" else (7, 15)
    temperature = fit_binary_temperature_grid(
        temp_samples,
        min_step=temp_range[0],
        max_step=temp_range[1],
    )

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
        "data_depth": _data_depth_metadata(len(dated_games), count),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tune binary sports meta weights.")
    parser.add_argument(
        "--leagues",
        default=",".join(ALL_BINARY_LEAGUES),
        help="Comma-separated league keys (default: all non-soccer leagues).",
    )
    parser.add_argument("--cutoff", default=_cutoff_today())
    args = parser.parse_args()
    leagues = [item.strip() for item in args.leagues.split(",") if item.strip()]
    existing: dict[str, dict] = {}
    if META_WEIGHTS_PATH.is_file():
        try:
            existing = json.loads(META_WEIGHTS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    payload: dict[str, dict] = dict(existing)
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
                f"loss {entry['log_loss_baseline_equal']} -> {entry['log_loss_tuned']} "
                f"(games={entry['data_depth']['games_loaded']})",
                flush=True,
            )
        else:
            print("  skipped", flush=True)

    if tuned and all(league in payload for league in CORE_LEAGUES):
        legacies = [
            entry["blend"].get("legacy", 0.05)
            for entry in payload.values()
            if isinstance(entry, dict) and "blend" in entry
        ]
        powers = [
            entry["blend"].get("power", 0.15)
            for entry in payload.values()
            if isinstance(entry, dict) and "blend" in entry
        ]
        sports = [
            entry["blend"].get("sport_pred", 0.0)
            for entry in payload.values()
            if isinstance(entry, dict) and "blend" in entry
        ]
        temps = [
            entry["temperature"]
            for entry in payload.values()
            if isinstance(entry, dict) and "temperature" in entry
        ]
        payload["default"] = {
            "blend": {
                "legacy": round(sum(legacies) / len(legacies), 3),
                "power": round(sum(powers) / len(powers), 3),
                "sport_pred": round(sum(sports) / len(sports), 3) if any(sports) else 0.8,
            },
            "temperature": round(sum(temps) / len(temps), 3),
            "two_layer": False,
            "data_depth": {
                "pro_seasons": BACKTEST_PRO_SEASONS,
                "scoreboard_days": BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
            },
        }
    elif "default" not in payload:
        payload["default"] = {
            "blend": {"legacy": 0.033, "power": 0.133, "sport_pred": 0.833},
            "temperature": 0.967,
            "two_layer": False,
        }

    payload["_meta"] = {
        "tuned_at": date.today().isoformat(),
        "calibration_window": "full",
        "pro_seasons": BACKTEST_PRO_SEASONS,
        "scoreboard_days": BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
        "leagues_tuned": tuned,
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
