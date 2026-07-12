"""Tune soccer 1X2 meta weights (log-loss) and write data/soccer_meta_weights.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import soccer_threeway_probs  # noqa: E402
from web.league_profiles import (  # noqa: E402
    BACKTEST_PRO_SEASONS,
    BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
    SOCCER_LEAGUES,
)
from web.power_model import build_power_ratings, predict_matchup  # noqa: E402
from web.ensemble_ml.config import MIN_TRAIN_ROWS, get_dataset_profile  # noqa: E402
from web.walkforward import calibration_indices  # noqa: E402
from web.soccer_blend import power_threeway_probs  # noqa: E402
from web.soccer_meta_model import (  # noqa: E402
    META_WEIGHTS_PATH,
    STAT_LAYER_KEYS,
    _blend_sharp_sample,
    blend_weighted_threeway,
    fit_blend_weights_grid,
    fit_decorrelation_weight_grid,
    fit_market_blend_weight_grid,
    fit_stat_weight_grid,
    fit_stat_weights_grid,
    fit_temperature_grid,
    multiclass_log_loss,
    outcome_from_score,
    temperature_scale,
)
from web.soccer_pred_model import (  # noqa: E402
    apply_path_a_output_calibration,
    build_soccer_path_a_model,
    build_soccer_model,
    compute_sharp_stat_layers,
    predict_matchup_from_model,
)
from web.soccer_decorrelation import devig_threeway_from_odds  # noqa: E402
from web.supplemental_games import soccer_backtest_odds  # noqa: E402
from web.ensemble_ml.config import sportsbook_logloss_benchmark  # noqa: E402
from web.tracking_service import toronto_cutoff_mdy, toronto_today  # noqa: E402

MIN_CALIBRATION_GAMES = 40
MIN_LEAGUE_GAMES = 60


def _cutoff_today() -> str:
    return toronto_cutoff_mdy()


def _load_dated_games(league: str, cutoff: str) -> list:
    profile = get_dataset_profile(league)
    if profile.get("dated_source") == "supplemental":
        from web.supplemental_games import load_supplemental_dated_games_for_backtest

        return load_supplemental_dated_games_for_backtest(league, cutoff)
    from web.season_games import load_league_dated_games_for_backtest

    return load_league_dated_games_for_backtest(league, cutoff)


def _calibration_indices(game_count: int, league: str) -> range:
    profile = get_dataset_profile(league)
    return calibration_indices(
        game_count,
        warmup=MIN_TRAIN_ROWS // 2,
        max_calibration_games=profile.get("max_calibration_games"),
        target_rows=profile.get("target_rows"),
    )


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
    indices = list(_calibration_indices(len(games), league))
    for index in indices:
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
    indices = list(_calibration_indices(len(games), league))
    for index in indices:
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


def _calibrated_loss_for_rows(
    rows: list[dict[str, Any]],
    *,
    league: str,
    tune_context: dict[str, Any],
    market_only: bool = False,
) -> float | None:
    if not rows:
        return None
    total = 0.0
    count = 0
    for row in rows:
        if market_only and row["market_probs"] is None:
            continue
        display, _pick = apply_path_a_output_calibration(
            row["raw_probs"],
            league=league,
            context=tune_context,
            market_probs=row["market_probs"],
        )
        total += multiclass_log_loss(*display, row["outcome"])
        count += 1
    return total / count if count else None


def _optimize_market_model_weight(
    rows: list[dict[str, Any]],
    *,
    league: str,
    tune_context: dict[str, Any],
    market_only: bool = True,
) -> tuple[float, float | None]:
    best_weight = float(tune_context.get("path_a_market_model_weight", 0.22))
    best_loss = _calibrated_loss_for_rows(
        rows,
        league=league,
        tune_context=tune_context,
        market_only=market_only,
    )
    tie_eps = 1e-4
    for step in range(0, 21):
        weight = step / 20.0
        trial_context = dict(tune_context)
        trial_context["path_a_market_model_weight"] = weight
        loss = _calibrated_loss_for_rows(
            rows,
            league=league,
            tune_context=trial_context,
            market_only=market_only,
        )
        if loss is None:
            continue
        if best_loss is None or loss < best_loss - tie_eps:
            best_loss = loss
            best_weight = weight
        elif abs(loss - best_loss) <= tie_eps and weight > best_weight:
            best_weight = weight
    return best_weight, best_loss


def tune_league(league: str, cutoff: str, *, path_a_only: bool = False) -> dict | None:
    dated_games = _load_dated_games(league, cutoff)
    games = [g for _d, g in dated_games]
    if len(games) < MIN_LEAGUE_GAMES:
        return None

    print(f"  walk-forward: {len(games)} games loaded", flush=True)

    existing_entry: dict = {}
    if META_WEIGHTS_PATH.is_file():
        try:
            existing_payload = json.loads(META_WEIGHTS_PATH.read_text(encoding="utf-8"))
            existing_entry = existing_payload.get(league) or {}
        except (json.JSONDecodeError, OSError):
            existing_entry = {}

    if path_a_only and existing_entry.get("stat"):
        stat_weights = existing_entry["stat"]
        temperature = float(existing_entry.get("temperature", 1.0))
        baseline = float(existing_entry.get("log_loss_baseline_equal", 1.0))
        tuned_stat_loss = float(existing_entry.get("log_loss_tuned_stat", baseline))
        blend_weights = existing_entry.get("blend")
        stat_samples = []
        temp_samples = []
    else:
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

        tuned_stat_loss = 0.0
        for home, draw, away, outcome in temp_samples:
            calibrated = temperature_scale(home, draw, away, temperature)
            tuned_stat_loss += multiclass_log_loss(*calibrated, outcome)
        tuned_stat_loss /= len(temp_samples)

    indices = list(_calibration_indices(len(games), league))
    if path_a_only and len(indices) > 500:
        step = max(1, len(indices) // 500)
        indices = indices[::step]

    path_a_eval_rows: list[dict[str, Any]] = []
    market_calib_samples: list[
        tuple[tuple[float, float, float], tuple[float, float, float], int]
    ] = []
    market_samples: list[tuple[tuple[float, float, float], int]] = []
    stat_xgb_samples: list[
        tuple[tuple[float, float, float], tuple[float, float, float], int]
    ] = []
    dated_games = _load_dated_games(league, cutoff)
    path_a_model = build_soccer_path_a_model(dated_games, league)
    for index in indices:
        game = games[index]
        home, away, _hn, _an, home_goals, away_goals = game
        if not path_a_model:
            continue
        if home not in path_a_model["team_keys"] or away not in path_a_model["team_keys"]:
            continue
        game_date = dated_games[index][0] if index < len(dated_games) else ""
        odds_row = soccer_backtest_odds(league, game_date, home, away) or {}
        market_probs = devig_threeway_from_odds(
            odds_row.get("home_odds"),
            odds_row.get("draw_odds"),
            odds_row.get("away_odds"),
        )
        stat_only = predict_matchup_from_model(
            path_a_model,
            home,
            away,
            match_date=game_date,
            market_probs=market_probs,
            use_xgb=False,
        )
        full_pred = predict_matchup_from_model(
            path_a_model,
            home,
            away,
            match_date=game_date,
            market_probs=market_probs,
            use_xgb=True,
        )
        if not stat_only or not full_pred:
            continue
        outcome = outcome_from_score(home_goals, away_goals)
        stat_probs = (
            stat_only.home_win_probability,
            stat_only.draw_probability,
            stat_only.away_win_probability,
        )
        full_probs = (
            full_pred.home_win_probability,
            full_pred.draw_probability,
            full_pred.away_win_probability,
        )
        stat_xgb_samples.append((stat_probs, full_probs, outcome))
        temp_full = temperature_scale(*full_probs, temperature)
        if market_probs is not None:
            market_samples.append((market_probs, outcome))
            market_calib_samples.append((temp_full, market_probs, outcome))
        path_a_eval_rows.append(
            {
                "raw_probs": full_probs,
                "market_probs": market_probs,
                "outcome": outcome,
            }
        )

    stat_weight = (
        fit_stat_weight_grid(stat_xgb_samples) if stat_xgb_samples else 0.45
    )
    if path_a_model:
        path_a_model["path_a_stat_weight"] = stat_weight
    market_model_weight = (
        fit_market_blend_weight_grid(market_calib_samples)
        if market_calib_samples
        else 0.22
    )
    tune_context = {
        "temperature": temperature,
        "path_a_stat_weight": stat_weight,
        "path_a_market_model_weight": market_model_weight,
        "path_a_decorrelation_weight": 0.28,
    }
    market_model_weight, calibrated_loss = _optimize_market_model_weight(
        path_a_eval_rows,
        league=league,
        tune_context=tune_context,
    )
    tune_context["path_a_market_model_weight"] = market_model_weight

    decor_samples: list[tuple[tuple[float, float, float], tuple[float, float, float], int]] = []
    for row in path_a_eval_rows:
        display, _pick = apply_path_a_output_calibration(
            row["raw_probs"],
            league=league,
            context=tune_context,
            market_probs=row["market_probs"],
        )
        if row["market_probs"] is not None:
            decor_samples.append((display, row["market_probs"], row["outcome"]))

    decorrelation_weight = (
        fit_decorrelation_weight_grid(decor_samples) if decor_samples else 0.28
    )
    tune_context["path_a_decorrelation_weight"] = decorrelation_weight

    tuned_path_a_loss = None
    calibrated_loss = None
    if path_a_eval_rows:
        tuned_path_a_loss = 0.0
        for row in path_a_eval_rows:
            temp_full = temperature_scale(*row["raw_probs"], temperature)
            tuned_path_a_loss += multiclass_log_loss(*temp_full, row["outcome"])
        tuned_path_a_loss /= len(path_a_eval_rows)
        calibrated_loss = _calibrated_loss_for_rows(
            path_a_eval_rows,
            league=league,
            tune_context=tune_context,
            market_only=bool(market_samples),
        )

    market_log_loss = None
    if market_samples:
        market_log_loss = sum(
            multiclass_log_loss(*market_probs, outcome)
            for market_probs, outcome in market_samples
        ) / len(market_samples)

    beats_market = (
        calibrated_loss is not None
        and market_log_loss is not None
        and calibrated_loss <= market_log_loss
    )

    entry = {
        "stat": {key: round(stat_weights[key], 3) for key in STAT_LAYER_KEYS},
        "temperature": round(temperature, 3),
        "path_a": {
            "stat_weight": round(stat_weight, 3),
            "decorrelation_weight": round(decorrelation_weight, 3),
            "dc_rho": -0.05,
            "market_model_weight": round(market_model_weight, 3),
        },
        "log_loss_baseline_equal": round(baseline, 4),
        "log_loss_tuned_stat": round(tuned_stat_loss, 4),
        "log_loss_tuned_path_a": round(tuned_path_a_loss, 4)
        if tuned_path_a_loss is not None
        else None,
        "log_loss_calibrated": round(calibrated_loss, 4)
        if calibrated_loss is not None
        else None,
        "log_loss_market_closing": round(market_log_loss, 4)
        if market_log_loss is not None
        else None,
        "log_loss_market_benchmark": sportsbook_logloss_benchmark(league),
        "beats_market_closing": beats_market,
        "samples": len(stat_samples) if stat_samples else existing_entry.get("samples", 0),
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
    parser.add_argument(
        "--path-a-only",
        action="store_true",
        help="Reuse stored stat weights; tune Path A market calibration only (faster).",
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
        entry = tune_league(league, cutoff, path_a_only=args.path_a_only)
        if entry:
            payload[league] = entry
            tuned_leagues += 1
            print(
                f"  stat={entry['stat']} temp={entry['temperature']} "
                f"path_a={entry['path_a']} "
                f"calibrated={entry.get('log_loss_calibrated')} "
                f"market={entry.get('log_loss_market_closing')} "
                f"beats_market={entry.get('beats_market_closing')} "
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
            "path_a": {
                "stat_weight": 0.45,
                "decorrelation_weight": 0.28,
                "dc_rho": -0.05,
                "market_model_weight": 0.22,
            },
            "data_depth": {
                "pro_seasons": BACKTEST_PRO_SEASONS,
                "scoreboard_days": BACKTEST_SCOREBOARD_LOOKBACK_DAYS,
            },
        }

    payload["_meta"] = {
        "tuned_at": toronto_today().isoformat(),
        "calibration_window": "full",
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
