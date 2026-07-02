"""Build leak-free training rows from historical games and layer outputs."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from web.blend_service import _run_sport_pred_model, home_win_prob_to_total_score
from web.bet_advisor import model_home_margin
from web.closing_odds_db import closing_odds_lookup
from web.ensemble_ml.config import (
    MIN_TRAIN_ROWS,
    TRAIN_LEAGUES,
    get_dataset_profile,
)
from web.ensemble_ml.features import (
    legacy_home_from_total,
    market_devig_home,
    market_devig_threeway,
)
from web.league_profiles import is_soccer_league
from web.power_model import build_power_ratings, predict_matchup
from web.season_games import load_league_dated_games_for_backtest
from web.soccer_blend import power_threeway_probs, soccer_threeway_probs
from web.soccer_meta_model import stack_soccer_blend_layers
from web.sports_meta_model import stack_binary_blend_layers

# College leagues: sport matrix rebuild is too slow per walk-forward step; proxy at train time.
SPORT_TRAIN_PROXY_LEAGUES = frozenset({"cbb", "ncaabb"})


def _sport_cutoff_bucket(game_date: str) -> str:
    """Monthly bucket so sport-specific models are not rebuilt every game day."""
    parsed = date.fromisoformat(game_date)
    return f"{parsed.month}-1-{parsed.year}"


def _load_dated_games(league: str, cutoff: str, *, dated_source: str) -> list:
    if dated_source == "supplemental":
        from web.supplemental_games import load_supplemental_dated_games_for_backtest

        return load_supplemental_dated_games_for_backtest(league, cutoff)
    return load_league_dated_games_for_backtest(league, cutoff)


def _calibration_indices(
    total: int,
    *,
    warmup: int,
    profile: dict[str, int | None | str],
) -> range:
    max_cal = profile.get("max_calibration_games")
    if max_cal is None:
        start = warmup
    else:
        start = max(warmup, total - int(max_cal))
    span = total - start
    target_rows = profile.get("target_rows")
    if target_rows is None or span <= int(target_rows):
        step = 1
    else:
        step = max(1, span // int(target_rows))
    return range(start, total, step)


def _training_games(
    dated_games: list,
    index: int,
    *,
    profile: dict[str, int | None | str],
) -> list:
    window = profile.get("power_train_window")
    if window is None:
        return [item[1] for item in dated_games[:index]]
    window = int(window)
    return [item[1] for item in dated_games[max(0, index - window) : index]]


def collect_binary_rows(
    league: str,
    cutoff: str,
    *,
    profile: dict[str, int | None | str] | None = None,
) -> pd.DataFrame:
    """Point-in-time layer features for non-soccer leagues."""
    league = league.lower()
    profile = profile or get_dataset_profile(league)
    dated_games = _load_dated_games(
        league, cutoff, dated_source=str(profile.get("dated_source", "espn"))
    )
    rows: list[dict[str, Any]] = []
    warmup = MIN_TRAIN_ROWS // 2

    indices = list(_calibration_indices(len(dated_games), warmup=warmup, profile=profile))
    if len(indices) > 500:
        print(f"  walk-forward: {len(indices)} games (of {len(dated_games)} total)", flush=True)

    for step_i, index in enumerate(indices):
        if len(indices) > 500 and step_i > 0 and step_i % 500 == 0:
            print(f"  ... {step_i}/{len(indices)} games processed", flush=True)
        game_date, game = dated_games[index]
        home, away, _hn, _an, home_score, away_score = game
        if home_score == away_score:
            continue
        train = _training_games(dated_games, index, profile=profile)
        if len(train) < warmup:
            continue

        teams, _total, param = build_power_ratings(train)
        if not param or home not in teams or away not in teams:
            continue
        power = predict_matchup(teams, param, home, away)
        if not power:
            continue

        power_home = float(power["home_win_probability"])
        power_total, _ = home_win_prob_to_total_score(power_home)
        power_margin = model_home_margin(power_total, league)

        game_cutoff = _sport_cutoff_bucket(game_date)
        if league in SPORT_TRAIN_PROXY_LEAGUES:
            sport_home = power_home
            sport_margin = power_margin
        else:
            _sport_key, sport_payload = _run_sport_pred_model(
                league, game_cutoff, home, away
            )
            sport_home = (
                float(sport_payload["home_win_probability"]) if sport_payload else None
            )
            sport_margin = None
            if sport_payload:
                if sport_payload.get("predicted_margin") is not None:
                    sport_margin = -float(sport_payload["predicted_margin"])
                elif sport_payload.get("projected_spread") is not None:
                    sport_margin = float(sport_payload["projected_spread"])

        legacy_home = legacy_home_from_total(power_total)
        legacy_margin = power_margin
        meta_stacked = stack_binary_blend_layers(
            legacy_home=legacy_home,
            power_home=power_home,
            sport_home=sport_home,
            league=league,
        )
        if meta_stacked is None:
            meta_stacked = (
                power_home if sport_home is None else 0.35 * power_home + 0.65 * sport_home
            )

        odds = closing_odds_lookup(league, game_date, home, away) or {}
        market_spread = odds.get("home_close_spread")
        home_ml = odds.get("home_close_ml")
        away_ml = odds.get("away_close_ml")
        market_home = market_devig_home(home_ml, away_ml)

        home_margin_actual = float(home_score - away_score)
        home_win = int(home_score > away_score)
        home_cover = None
        if market_spread is not None:
            home_cover = int(home_margin_actual + float(market_spread) > 0)

        rows.append(
            {
                "game_date": game_date,
                "home_key": home,
                "away_key": away,
                "legacy_home_prob": legacy_home,
                "power_home_prob": power_home,
                "sport_home_prob": sport_home,
                "meta_stacked_home_prob": meta_stacked,
                "legacy_margin": legacy_margin,
                "power_margin": power_margin,
                "sport_margin": sport_margin,
                "market_devig_home_prob": market_home,
                "market_spread": market_spread,
                "market_home_ml": home_ml,
                "market_away_ml": away_ml,
                "home_margin": home_margin_actual,
                "home_win": home_win,
                "home_cover": home_cover,
            }
        )

    return pd.DataFrame(rows)


def collect_soccer_rows(
    league: str,
    cutoff: str,
    *,
    profile: dict[str, int | None | str] | None = None,
) -> pd.DataFrame:
    league = league.lower()
    profile = profile or get_dataset_profile(league)
    dated_games = _load_dated_games(
        league, cutoff, dated_source=str(profile.get("dated_source", "espn"))
    )
    rows: list[dict[str, Any]] = []
    warmup = MIN_TRAIN_ROWS // 2

    indices = list(_calibration_indices(len(dated_games), warmup=warmup, profile=profile))
    if len(indices) > 500:
        print(f"  walk-forward: {len(indices)} games (of {len(dated_games)} total)", flush=True)

    for step_i, index in enumerate(indices):
        if len(indices) > 500 and step_i > 0 and step_i % 500 == 0:
            print(f"  ... {step_i}/{len(indices)} games processed", flush=True)
        game_date, game = dated_games[index]
        home, away, _hn, _an, home_score, away_score = game
        train = _training_games(dated_games, index, profile=profile)
        if len(train) < warmup:
            continue

        teams, _total, param = build_power_ratings(train)
        if not param or home not in teams or away not in teams:
            continue
        power = predict_matchup(teams, param, home, away)
        if not power:
            continue

        power_home = float(power["home_win_probability"])
        power_tw = power_threeway_probs(power_home, league)
        legacy_tw = soccer_threeway_probs(power_home * 2 - 50, league)

        game_cutoff = _sport_cutoff_bucket(game_date)
        _sport_key, sport_payload = _run_sport_pred_model(
            league, game_cutoff, home, away
        )
        sport_h = sport_d = sport_a = None
        if sport_payload:
            sport_h = float(sport_payload.get("home_win_probability") or 0)
            sport_d = float(sport_payload.get("draw_probability") or 0)
            sport_a = float(sport_payload.get("away_win_probability") or 0)

        meta = stack_soccer_blend_layers(
            legacy=legacy_tw,
            power=power_tw,
            soccer_pred=(sport_h, sport_d, sport_a) if sport_h is not None else None,
            league=league,
        )
        meta_h, meta_d, meta_a = meta if meta else (None, None, None)

        odds = closing_odds_lookup(league, game_date, home, away) or {}
        mkt_h, mkt_d, mkt_a = market_devig_threeway(
            odds.get("home_close_ml"),
            odds.get("draw_close_ml"),
            odds.get("away_close_ml"),
        )

        if home_score > away_score:
            outcome = "home"
        elif home_score < away_score:
            outcome = "away"
        else:
            outcome = "draw"

        rows.append(
            {
                "game_date": game_date,
                "home_key": home,
                "away_key": away,
                "legacy_home_prob": legacy_tw[0],
                "legacy_draw_prob": legacy_tw[1],
                "legacy_away_prob": legacy_tw[2],
                "power_home_prob": power_tw[0],
                "power_draw_prob": power_tw[1],
                "power_away_prob": power_tw[2],
                "sport_home_prob": sport_h,
                "sport_draw_prob": sport_d,
                "sport_away_prob": sport_a,
                "meta_home_prob": meta_h,
                "meta_draw_prob": meta_d,
                "meta_away_prob": meta_a,
                "market_devig_home_prob": mkt_h,
                "market_devig_draw_prob": mkt_d,
                "market_devig_away_prob": mkt_a,
                "outcome": outcome,
                "home_win": int(outcome == "home"),
                "draw": int(outcome == "draw"),
                "away_win": int(outcome == "away"),
            }
        )

    return pd.DataFrame(rows)


def build_league_dataset(
    league: str,
    cutoff: str | None = None,
    *,
    profile: dict[str, int | None | str] | None = None,
) -> pd.DataFrame:
    league = league.lower()
    if league not in TRAIN_LEAGUES:
        raise ValueError(f"Unsupported league: {league}")
    if cutoff is None:
        today = date.today()
        cutoff = f"{today.month}-{today.day}-{today.year}"
    if is_soccer_league(league):
        return collect_soccer_rows(league, cutoff, profile=profile)
    return collect_binary_rows(league, cutoff, profile=profile)
