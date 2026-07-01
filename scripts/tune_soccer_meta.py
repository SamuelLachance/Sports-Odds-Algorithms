"""Tune soccer 1X2 meta weights (log-loss) and write data/soccer_meta_weights.json."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import soccer_threeway_probs  # noqa: E402
from web.league_profiles import (  # noqa: E402
    BACKTEST_PRO_SEASONS,
    BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
    SOCCER_LEAGUES,
)
from web.power_model import build_power_ratings, predict_matchup  # noqa: E402
from web.season_games import load_league_completed_games_for_backtest  # noqa: E402
from web.soccer_blend import power_threeway_probs  # noqa: E402
from web.soccer_meta_model import (  # noqa: E402
    META_WEIGHTS_PATH,
    STAT_LAYER_KEYS,
    _blend_sharp_sample,
    blend_weighted_threeway,
    fit_blend_weights_grid,
    fit_stat_weights_grid,
    fit_temperature_grid,
    multiclass_log_loss,
    outcome_from_score,
)
from web.soccer_pred_model import (  # noqa: E402
    build_soccer_model,
    compute_sharp_stat_layers,
    predict_matchup_from_model,
)

MIN_CALIBRATION_GAMES = 40
CALIBRATION_WINDOW = 120
MIN_LEAGUE_GAMES = 60


def _cutoff_today() -> str:
    today = date.today()
    return f"{today.month}-{today.day}-{today.year}"


def _calibration_start(game_count: int) -> int:
    window = min(CALIBRATION_WINDOW, max(80, game_count // 3))
    return max(30, game_count - window)


def _data_depth_metadata(game_count: int) -> dict[str, int]:
    return {
        "games_loaded": game_count,
        "pro_seasons": BACKTEST_PRO_SEASONS,
        "scoreboard_days": BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
        "supplemental": 1,
    }


def _collect_stat_samples(
    league: str,
    cutoff: str,
    games: list[tuple],
) -> list[tuple]:
    samples = []
    start = _calibration_start(len(games))
    for index in range(start, len(games)):
        game = games[index]
        home, away, _hn, _an, home_goals, away_goals = game
        model = build_soccer_model(games[:index], league)
        if not model:
            continue
        if home not in model["team_keys"] or away not in model["team_keys"]:
            continue
        layers, _lam, _mu = compute_sharp_stat_layers(model, home, away)
        samples.append(
            (
                layers["elo"],
                layers["pi"],
                layers["dc"],
                layers["form"],
                layers["club_elo"],
                outcome_from_score(home_goals, away_goals),
            )
        )
    return samples


def _collect_blend_samples(
    league: str,
    cutoff: str,
    games: list[tuple],
) -> list[tuple]:
    samples = []
    start = max(40, _calibration_start(len(games)))
    for index in range(start, len(games)):
        game = games[index]
        home, away, _hn, _an, home_goals, away_goals = game
        teams, _total, param = build_power_ratings(games[:index])
        power = predict_matchup(teams, param, home, away) if param else None
        model = build_soccer_model(games[:index], league)
        soccer = (
            predict_matchup_from_model(model, home, away)
            if model and home in model["team_keys"] and away in model["team_keys"]
            else None
        )
        if not power or not soccer:
            continue
        power_tw = power_threeway_probs(float(power["home_win_probability"]), league)
        soccer_tw = (
            soccer.home_win_probability,
            soccer.draw_probability,
            soccer.away_win_probability,
        )
        total = (
            -float(power["home_win_probability"])
            if float(power["home_win_probability"]) > 50
            else float(power["home_win_probability"])
        )
        legacy_tw = soccer_threeway_probs(total, league)
        samples.append(
            (legacy_tw, power_tw, soccer_tw, outcome_from_score(home_goals, away_goals))
        )
    return samples


def tune_league(league: str, cutoff: str) -> dict | None:
    games = load_league_completed_games_for_backtest(league, cutoff)
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    stat_samples = _collect_stat_samples(league, cutoff, games)
    if len(stat_samples) < MIN_CALIBRATION_GAMES:
        return None

    stat_weights = fit_stat_weights_grid(stat_samples)
    temp_samples = []
    for elo, pi, dc, form, club, outcome in stat_samples:
        layers = {
            "elo": elo,
            "pi": pi,
            "dc": dc,
            "form": form,
            "club_elo": club,
        }
        home, draw, away = _blend_sharp_sample(layers, stat_weights)
        temp_samples.append((home, draw, away, outcome))
    temperature = fit_temperature_grid(temp_samples)

    blend_samples = _collect_blend_samples(league, cutoff, games)
    blend_weights = (
        fit_blend_weights_grid(blend_samples)
        if len(blend_samples) >= MIN_CALIBRATION_GAMES
        else None
    )

    baseline = sum(
        multiclass_log_loss(
            *_blend_sharp_sample(
                {
                    "elo": elo,
                    "pi": pi,
                    "dc": dc,
                    "form": form,
                    "club_elo": club,
                },
                {key: 1.0 / 5.0 for key in STAT_LAYER_KEYS},
            ),
            outcome,
        )
        for elo, pi, dc, form, club, outcome in stat_samples
    ) / len(stat_samples)

    from web.soccer_meta_model import temperature_scale

    tuned_stat_loss = 0.0
    for home, draw, away, outcome in temp_samples:
        calibrated = temperature_scale(home, draw, away, temperature)
        tuned_stat_loss += multiclass_log_loss(*calibrated, outcome)
    tuned_stat_loss /= len(temp_samples)

    entry = {
        "stat": {key: round(stat_weights[key], 3) for key in STAT_LAYER_KEYS},
        "temperature": round(temperature, 3),
        "log_loss_baseline_equal": round(baseline, 4),
        "log_loss_tuned_stat": round(tuned_stat_loss, 4),
        "samples": len(stat_samples),
        "data_depth": _data_depth_metadata(len(games)),
    }
    if blend_weights:
        entry["blend"] = {key: round(value, 3) for key, value in blend_weights.items()}
    return entry


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tune soccer 1X2 meta weights.")
    parser.add_argument(
        "--leagues",
        default=",".join(SOCCER_LEAGUES),
        help="Comma-separated league keys (default: all soccer leagues).",
    )
    args = parser.parse_args()
    leagues = [league.strip() for league in args.leagues.split(",") if league.strip()]
    cutoff = _cutoff_today()
    existing: dict[str, dict] = {}
    if META_WEIGHTS_PATH.is_file():
        try:
            existing = json.loads(META_WEIGHTS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    payload: dict[str, dict] = dict(existing)
    tuned_leagues = 0

    for league in leagues:
        print(f"Tuning {league}...", flush=True)
        entry = tune_league(league, cutoff)
        if entry:
            payload[league] = entry
            tuned_leagues += 1
            print(
                f"  stat={entry['stat']} temp={entry['temperature']} "
                f"loss {entry['log_loss_baseline_equal']} -> {entry['log_loss_tuned_stat']} "
                f"(games={entry['data_depth']['games_loaded']})",
                flush=True,
            )
        else:
            print("  skipped (insufficient games)", flush=True)

    if tuned_leagues:
        stat_avgs: dict[str, list[float]] = {key: [] for key in STAT_LAYER_KEYS}
        temps = []
        for entry in payload.values():
            if not isinstance(entry, dict) or "stat" not in entry:
                continue
            for key in STAT_LAYER_KEYS:
                if key in entry["stat"]:
                    stat_avgs[key].append(float(entry["stat"][key]))
            if "temperature" in entry:
                temps.append(float(entry["temperature"]))
        payload["default"] = {
            "stat": {
                key: round(sum(values) / len(values), 3) if values else 0.0
                for key, values in stat_avgs.items()
            },
            "blend": {"legacy": 0.05, "power": 0.15, "soccer_pred": 0.80},
            "temperature": round(sum(temps) / len(temps), 3) if temps else 1.0,
            "data_depth": {
                "pro_seasons": BACKTEST_PRO_SEASONS,
                "scoreboard_days": BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
            },
        }

    payload["_meta"] = {
        "tuned_at": date.today().isoformat(),
        "calibration_window": CALIBRATION_WINDOW,
        "pro_seasons": BACKTEST_PRO_SEASONS,
        "scoreboard_days": BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
        "leagues_tuned": tuned_leagues,
    }

    META_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_WEIGHTS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {META_WEIGHTS_PATH} ({tuned_leagues} leagues)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
