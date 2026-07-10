"""Official pick rules per sport (spread vs moneyline) with a uniform edge gate."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from web.bet_advisor import (
    BetPick,
    enrich_pick_profit_metrics,
    evaluate_picks,
    evaluate_soccer_picks,
    evaluate_spread_picks,
    model_home_margin,
    official_pick_binary_prob_sets,
    soccer_model_moneylines,
    spread_line_for_side,
)
from web.blend_service import blended_home_spread_margin, home_win_prob_to_total_score
from web.hubacek_picks import (
    HUBACEK_MIN_MARKET_GAP_PP,
    hubacek_min_ev_pct,
    hubacek_min_market_gap_pp,
    hubacek_min_win_confidence_pp,
    hubacek_ml_range,
    official_hubacek_thresholds,
)
from web.league_profiles import (
    MIN_EXPECTED_VALUE_PCT,
    MIN_RECOMMENDED_EDGE,
    get_league_profile,
    is_soccer_league,
)
from web.closing_odds_db import closing_odds_lookup
from web.power_model import PowerGame, PowerTeam, build_power_ratings, fit_logistic_param, predict_matchup
from web.season_games import load_league_completed_games_for_backtest
from web.supplemental_games import load_supplemental_dated_games_for_backtest, soccer_backtest_odds
from web.sports_meta_model import apply_binary_calibration, stack_binary_blend_layers
from web.tracking_service import calculate_units

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_PATH = PROJECT_ROOT / "data" / "pick_strategy.json"

OfficialBetType = Literal["spread", "moneyline", "soccer_1x2", "none"]

# Official picks: Hubáček decorrelation gap only (see web.hubacek_picks).
OFFICIAL_MIN_EDGE = 0.0
OFFICIAL_MIN_SPREAD_POINT_EDGE = 0.0

DEFAULT_THRESHOLDS: dict[str, Any] = {
    **official_hubacek_thresholds(),
    "min_spread_point_edge": OFFICIAL_MIN_SPREAD_POINT_EDGE,
    "min_profit_score": 0.0,
    "min_kelly_pct": 0.0,
    "enabled": True,
}

MARKET_SHRINK = 0.55
DEFAULT_SPREAD_JUICE = -110
BACKTEST_MIN_TRAIN = 40
MIN_TUNE_BETS = 10
DRAWDOWN_PENALTY = 0.35
PROFIT_SCORE_QUARTILE_FRACTION = 0.25


def official_bet_type(league: str) -> OfficialBetType:
    """Spread for basketball/football; moneyline for hockey/baseball; 1X2 for soccer."""
    league = league.lower()
    if is_soccer_league(league):
        return "soccer_1x2"
    profile = get_league_profile(league)
    category = profile["category"]
    if category in ("basketball", "football"):
        return "spread"
    if category in ("hockey", "baseball"):
        return "moneyline"
    return "moneyline"


def _category_for_league(league: str) -> str:
    return get_league_profile(league.lower())["category"]


@lru_cache(maxsize=1)
def load_pick_strategy() -> dict[str, Any]:
    if not STRATEGY_PATH.is_file():
        return {"default": _default_entry("moneyline")}
    try:
        payload = json.loads(STRATEGY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"default": _default_entry("moneyline")}
    if "default" not in payload:
        payload["default"] = _default_entry("moneyline")
    return payload


def _default_entry(bet_type: str) -> dict[str, Any]:
    return {
        "bet_type": bet_type,
        **DEFAULT_THRESHOLDS,
        "backtest_roi_pct": None,
        "backtest_bets": 0,
        "enabled": bet_type != "none",
    }


def get_pick_thresholds(league: str) -> dict[str, Any]:
    """Live official-pick thresholds: Hubáček gates with per-league overrides."""
    config = load_pick_strategy()
    league = league.lower()
    entry = config.get(league) or config.get(_category_for_league(league)) or config["default"]
    bet_type = entry.get("bet_type") or official_bet_type(league)
    hubacek = official_hubacek_thresholds()
    hubacek["min_market_gap_pp"] = hubacek_min_market_gap_pp(league)
    hubacek["min_win_confidence_pp"] = hubacek_min_win_confidence_pp(league)
    hubacek["min_ev_pct"] = hubacek_min_ev_pct(league)
    if entry.get("min_spread_cover_gap_pp") is not None:
        hubacek["min_spread_cover_gap_pp"] = float(entry["min_spread_cover_gap_pp"])
    if entry.get("min_spread_confidence_pp") is not None:
        hubacek["min_spread_confidence_pp"] = float(entry["min_spread_confidence_pp"])
    ml_range = hubacek_ml_range(league)
    return {
        "bet_type": bet_type,
        **hubacek,
        "ml_lo": ml_range[0] if ml_range else None,
        "ml_hi": ml_range[1] if ml_range else None,
        "allowed_sides": entry.get("allowed_sides"),
        "min_spread_point_edge": OFFICIAL_MIN_SPREAD_POINT_EDGE,
        "min_profit_score": 0.0,
        "min_kelly_pct": 0.0,
        "enabled": True,
        "backtest_roi_pct": entry.get("backtest_roi_pct"),
        "backtest_bets": entry.get("backtest_bets", 0),
        "backtest_units": entry.get("backtest_units"),
    }


def passes_profit_gate(pick: BetPick, thresholds: dict[str, Any]) -> bool:
    """Kelly + profit-score gate after edge/EV thresholds (backtest-tuned)."""
    enriched = enrich_pick_profit_metrics(pick)
    if enriched.profit_score < thresholds.get("min_profit_score", 0.0):
        return False
    kelly_pct = float(enriched.extra.get("kelly_pct") or 0.0)
    if kelly_pct < thresholds.get("min_kelly_pct", 0.0):
        return False
    return True


def _round_spread(value: float) -> float:
    return round(value * 2.0) / 2.0


def simulate_market_spread(
    model_margin_home: float,
    league: str,
    *,
    market_margin_home: float | None = None,
) -> float:
    """Synthetic closing spread: market regresses a weaker public line toward pick-em."""
    base = market_margin_home if market_margin_home is not None else model_margin_home
    shrunk = base * (1.0 - MARKET_SHRINK)
    if league in ("nfl", "cfb"):
        return _round_spread(shrunk)
    return _round_spread(shrunk)


def simulate_market_moneylines(
    home_prob: float,
    *,
    market_home_prob: float | None = None,
) -> tuple[int, int]:
    """Synthetic two-way market with vig from regressed model probability."""
    from web.bet_advisor import projections_from_win_probs

    base = market_home_prob if market_home_prob is not None else home_prob
    market_home = 50.0 + (base - 50.0) * (1.0 - MARKET_SHRINK)
    market_home = min(max(market_home, 5.0), 95.0)
    away_prob = 100.0 - market_home
    away_ml, home_ml = projections_from_win_probs(market_home, away_prob)
    if home_ml < 0:
        home_ml = int(home_ml * 1.04)
    else:
        home_ml = int(home_ml * 0.96)
    if away_ml < 0:
        away_ml = int(away_ml * 1.04)
    else:
        away_ml = int(away_ml * 0.96)
    return away_ml, home_ml


def grade_spread_bet(
    side: str,
    home_goals: int,
    away_goals: int,
    home_spread: float,
) -> str:
    margin = home_goals - away_goals
    line = spread_line_for_side(home_spread, side)
    if side == "home":
        diff = margin + line
    else:
        diff = -margin + line
    if diff > 0:
        return "win"
    if diff < 0:
        return "loss"
    return "push"


def grade_moneyline_bet(side: str, home_goals: int, away_goals: int) -> str:
    if home_goals == away_goals:
        return "push"
    home_won = home_goals > away_goals
    if side == "home":
        return "win" if home_won else "loss"
    return "win" if not home_won else "loss"


def grade_soccer_1x2_bet(side: str, home_goals: int, away_goals: int) -> str:
    if side == "draw":
        return "win" if home_goals == away_goals else "loss"
    if side == "home":
        return "win" if home_goals > away_goals else "loss"
    return "win" if away_goals > home_goals else "loss"


def simulate_market_threeway(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    *,
    market_home: float | None = None,
    market_draw: float | None = None,
    market_away: float | None = None,
) -> tuple[int, int, int]:
    """Synthetic 3-way market with vig from regressed model probabilities."""
    base_home = market_home if market_home is not None else home_prob
    base_draw = market_draw if market_draw is not None else draw_prob
    base_away = market_away if market_away is not None else away_prob
    shrink = 1.0 - MARKET_SHRINK
    reg_home = 33.33 + (base_home - 33.33) * shrink
    reg_draw = 33.33 + (base_draw - 33.33) * shrink
    reg_away = 33.33 + (base_away - 33.33) * shrink
    away_ml, draw_ml, home_ml = soccer_model_moneylines(reg_home, reg_draw, reg_away)
    return away_ml, draw_ml, home_ml


def _max_drawdown(unit_deltas: list[float]) -> float:
    peak = 0.0
    cumulative = 0.0
    worst = 0.0
    for delta in unit_deltas:
        cumulative += delta
        peak = max(peak, cumulative)
        worst = max(worst, peak - cumulative)
    return worst


def _stack_and_calibrate_home_prob(
    *,
    league: str,
    power_home: float,
    sport_home: float | None,
) -> float:
    """Match production blend: meta stack then temperature calibration."""
    if sport_home is None:
        blended = power_home
    else:
        stacked = stack_binary_blend_layers(
            legacy_home=power_home,
            power_home=power_home,
            sport_home=sport_home,
            league=league,
        )
        blended = stacked if stacked is not None else (0.35 * power_home + 0.65 * sport_home)
    return apply_binary_calibration(blended, league)


def _run_sport_home_prob(
    league: str,
    cutoff: str,
    home: str,
    away: str,
    train_games: list,
) -> float | None:
    from web.blend_service import _run_sport_pred_model

    _key, payload = _run_sport_pred_model(league, cutoff, home, away)
    if not payload:
        return None
    return float(payload["home_win_probability"])


def _predict_blend_home_prob(
    league: str,
    cutoff: str,
    home: str,
    away: str,
    train_games: list,
) -> tuple[float, float, float, float] | None:
    teams, _total, param = build_power_ratings(train_games)
    if not param or home not in teams or away not in teams:
        return None
    power = predict_matchup(teams, param, home, away)
    if not power:
        return None
    power_home = float(power["home_win_probability"])
    sport_home = _run_sport_home_prob(league, cutoff, home, away, train_games)
    blended = _stack_and_calibrate_home_prob(
        league=league, power_home=power_home, sport_home=sport_home
    )
    total, win_prob = home_win_prob_to_total_score(blended)
    margin = model_home_margin(total, league)
    power_total, _power_win = home_win_prob_to_total_score(power_home)
    power_margin = model_home_margin(power_total, league)
    return blended, margin, power_margin, power_home, sport_home


def _closing_market_fields(
    league: str,
    game_date: str,
    home: str,
    away: str,
) -> dict[str, Any]:
    """Real closing lines when cached locally; empty dict otherwise."""
    odds = closing_odds_lookup(league, game_date, home, away)
    if not odds:
        return {}
    fields: dict[str, Any] = {}
    if odds.get("home_close_ml") is not None:
        fields["market_home_odds"] = odds["home_close_ml"]
    if odds.get("away_close_ml") is not None:
        fields["market_away_odds"] = odds["away_close_ml"]
    if odds.get("home_close_spread") is not None:
        fields["market_spread"] = float(odds["home_close_spread"])
    if odds.get("home_spread_odds") is not None:
        fields["home_spread_odds"] = odds["home_spread_odds"]
    if odds.get("away_spread_odds") is not None:
        fields["away_spread_odds"] = odds["away_spread_odds"]
    return fields


def _evaluate_backtest_pick(
    *,
    league: str,
    bet_type: OfficialBetType,
    blended_home: float,
    model_margin: float,
    power_margin: float | None = None,
    power_home: float | None = None,
    home_goals: int,
    away_goals: int,
    thresholds: dict[str, Any],
    home_prob: float | None = None,
    draw_prob: float | None = None,
    away_prob: float | None = None,
    market_home_odds: int | None = None,
    market_draw_odds: int | None = None,
    market_away_odds: int | None = None,
    market_spread: float | None = None,
    home_spread_odds: int | None = None,
    away_spread_odds: int | None = None,
) -> tuple[float, str] | None:
    if bet_type == "soccer_1x2":
        if home_prob is None or draw_prob is None or away_prob is None:
            return None
        away_ml, draw_ml, home_ml = simulate_market_threeway(
            home_prob,
            draw_prob,
            away_prob,
            market_home=power_home,
            market_draw=draw_prob,
            market_away=100.0 - home_prob - draw_prob if power_home is not None else None,
        )
        if market_home_odds is not None:
            home_ml = market_home_odds
        if market_draw_odds is not None:
            draw_ml = market_draw_odds
        if market_away_odds is not None:
            away_ml = market_away_odds
        away_proj, draw_proj, home_proj = soccer_model_moneylines(
            home_prob, draw_prob, away_prob
        )
        picks = evaluate_soccer_picks(
            away_name="Away",
            home_name="Home",
            away_slug="away",
            home_slug="home",
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            away_proj=away_proj,
            draw_proj=draw_proj,
            home_proj=home_proj,
            away_market=away_ml,
            draw_market=draw_ml,
            home_market=home_ml,
            min_edge=thresholds["min_edge"],
            min_ev_pct=thresholds["min_ev_pct"],
        )
        if not picks:
            return None
        pick = enrich_pick_profit_metrics(picks[0])
        if not passes_profit_gate(pick, thresholds):
            return None
        result = grade_soccer_1x2_bet(pick.side, home_goals, away_goals)
        odds = pick.market_odds
    elif bet_type == "spread":
        market_spread_line = (
            market_spread
            if market_spread is not None
            else simulate_market_spread(
                model_margin,
                league,
                market_margin_home=power_margin,
            )
        )
        total, win_prob = home_win_prob_to_total_score(blended_home)
        picks = evaluate_spread_picks(
            league=league,
            away_name="Away",
            home_name="Home",
            away_slug="away",
            home_slug="home",
            total_score=total,
            win_probability=win_prob,
            consensus_spread=market_spread_line,
            away_spread_odds=away_spread_odds or DEFAULT_SPREAD_JUICE,
            home_spread_odds=home_spread_odds or DEFAULT_SPREAD_JUICE,
            model_margin_home=model_margin,
            min_edge=thresholds["min_edge"],
            min_point_edge=thresholds["min_spread_point_edge"],
        )
        if not picks:
            return None
        pick = enrich_pick_profit_metrics(picks[0])
        if not passes_profit_gate(pick, thresholds):
            return None
        result = grade_spread_bet(pick.side, home_goals, away_goals, market_spread_line)
        odds = pick.spread_odds or home_spread_odds or away_spread_odds or DEFAULT_SPREAD_JUICE
    else:
        away_ml, home_ml = simulate_market_moneylines(
            blended_home,
            market_home_prob=power_home,
        )
        if market_home_odds is not None:
            home_ml = market_home_odds
        if market_away_odds is not None:
            away_ml = market_away_odds
        total, win_prob = home_win_prob_to_total_score(blended_home)
        picks = evaluate_picks(
            away_name="Away",
            home_name="Home",
            away_slug="away",
            home_slug="home",
            total_score=total,
            win_probability=win_prob,
            away_market=away_ml,
            home_market=home_ml,
            away_prob=100.0 - blended_home,
            home_prob=blended_home,
            min_edge=thresholds["min_edge"],
            min_ev_pct=thresholds["min_ev_pct"],
        )
        if not picks:
            return None
        pick = enrich_pick_profit_metrics(picks[0])
        if not passes_profit_gate(pick, thresholds):
            return None
        result = grade_moneyline_bet(pick.side, home_goals, away_goals)
        odds = pick.market_odds

    if result == "push":
        return 0.0, result
    units = calculate_units(1.0, int(odds), result)  # type: ignore[arg-type]
    return units, result


def _append_power_game(
    teams: dict[str, PowerTeam],
    total_games: list[PowerGame],
    home_key: str,
    away_key: str,
    home_name: str,
    away_name: str,
    home_score: int,
    away_score: int,
    *,
    iterations: int = 6,
) -> None:
    if home_key not in teams:
        teams[home_key] = PowerTeam(key=home_key, name=home_name)
    if away_key not in teams:
        teams[away_key] = PowerTeam(key=away_key, name=away_name)
    home_team = teams[home_key]
    away_team = teams[away_key]
    game = PowerGame(home_team, away_team, home_score, away_score)
    total_games.append(game)
    home_team.games.append(game)
    away_team.games.append(game)
    if home_score > away_score:
        home_team.wins += 1
        away_team.losses += 1
    elif home_score < away_score:
        home_team.losses += 1
        away_team.wins += 1
    else:
        home_team.ties += 1
        away_team.ties += 1
    home_team.apd = home_team.calc_apd()
    away_team.apd = away_team.calc_apd()
    for _ in range(iterations):
        for team in teams.values():
            team.schedule = team.calc_sched()
            team.power = team.calc_power()
        for team in teams.values():
            team.prev_power = team.power


def _predict_blend_from_power_state(
    league: str,
    cutoff: str,
    home: str,
    away: str,
    teams: dict[str, PowerTeam],
    total_games: list[PowerGame],
    *,
    use_sport_layer: bool = True,
) -> tuple[float, float, float, float] | None:
    param = fit_logistic_param(total_games, fast=True)
    if param is None:
        return None
    power = predict_matchup(teams, param, home, away)
    if not power:
        return None
    power_home = float(power["home_win_probability"])
    sport_home = (
        _run_sport_home_prob(league, cutoff, home, away, [])
        if use_sport_layer
        else None
    )
    blended = _stack_and_calibrate_home_prob(
        league=league, power_home=power_home, sport_home=sport_home
    )
    total, _win_prob = home_win_prob_to_total_score(blended)
    margin = model_home_margin(total, league)
    power_total, _power_win = home_win_prob_to_total_score(power_home)
    power_margin = model_home_margin(power_total, league)
    return blended, margin, power_margin, power_home, sport_home


def _collect_soccer_backtest_samples(
    league: str,
    cutoff: str,
    *,
    min_train: int = BACKTEST_MIN_TRAIN,
    use_sport_layer: bool = True,
) -> list[dict[str, Any]]:
    from web.bet_advisor import soccer_threeway_probs
    from web.soccer_blend import power_threeway_probs
    from web.soccer_meta_model import apply_threeway_calibration, stack_soccer_blend_layers
    from web.soccer_pred_model import build_soccer_model, predict_matchup_from_model

    games = load_supplemental_dated_games_for_backtest(league, cutoff)
    if len(games) > 2200:
        games = games[-2200:]
    teams: dict[str, PowerTeam] = {}
    power_games: list[PowerGame] = []
    samples: list[dict[str, Any]] = []

    for index, (game_date, (home, away, home_name, away_name, hg, ag)) in enumerate(games):
        if index < min_train:
            _append_power_game(
                teams, power_games, home, away, home_name, away_name, hg, ag
            )
            continue
        prediction = _predict_blend_from_power_state(
            league,
            cutoff,
            home,
            away,
            teams,
            power_games,
            use_sport_layer=use_sport_layer,
        )
        if not prediction:
            _append_power_game(
                teams, power_games, home, away, home_name, away_name, hg, ag
            )
            continue
        _blended_home, _margin, _power_margin, power_home, _sport_home = prediction
        power_tw = power_threeway_probs(power_home, league)
        soccer_tw = power_tw
        if use_sport_layer:
            game_tuples = [game for _date, game in games]
            model = build_soccer_model(game_tuples[:index], league)
            soccer = (
                predict_matchup_from_model(
                    model, home, away, include_club_elo=False
                )
                if model and home in model["team_keys"] and away in model["team_keys"]
                else None
            )
            if soccer:
                soccer_tw = (
                    soccer.home_win_probability,
                    soccer.draw_probability,
                    soccer.away_win_probability,
                )
        total = -power_home if power_home > 50 else power_home
        legacy_tw = soccer_threeway_probs(total, league)
        stacked = stack_soccer_blend_layers(
            legacy=legacy_tw,
            power=power_tw,
            soccer_pred=soccer_tw if use_sport_layer else None,
            league=league,
        )
        if stacked:
            home_p, draw_p, away_p = stacked
        else:
            home_p, draw_p, away_p = soccer_tw
        home_p, draw_p, away_p = apply_threeway_calibration(home_p, draw_p, away_p, league)
        odds = soccer_backtest_odds(league, game_date, home, away) or {}
        samples.append(
            {
                "blended_home": home_p,
                "home_prob": home_p,
                "draw_prob": draw_p,
                "away_prob": away_p,
                "power_home": power_tw[0],
                "home_goals": hg,
                "away_goals": ag,
                "market_home_odds": odds.get("home_odds"),
                "market_draw_odds": odds.get("draw_odds"),
                "market_away_odds": odds.get("away_odds"),
            }
        )
        _append_power_game(
            teams, power_games, home, away, home_name, away_name, hg, ag
        )
    return samples


def _collect_backtest_samples(
    league: str,
    cutoff: str,
    *,
    min_train: int = BACKTEST_MIN_TRAIN,
    use_sport_layer: bool = True,
) -> list[dict[str, Any]]:
    """Walk-forward samples using all available backtest history after min_train."""
    bet_type = official_bet_type(league)
    if bet_type == "soccer_1x2":
        return _collect_soccer_backtest_samples(
            league, cutoff, min_train=min_train, use_sport_layer=use_sport_layer
        )
    games = load_supplemental_dated_games_for_backtest(league, cutoff)
    if len(games) > 2200:
        games = games[-2200:]
    teams: dict[str, PowerTeam] = {}
    power_games: list[PowerGame] = []
    samples: list[dict[str, Any]] = []

    for index, (game_date, (home, away, home_name, away_name, hg, ag)) in enumerate(games):
        if index < min_train:
            _append_power_game(
                teams, power_games, home, away, home_name, away_name, hg, ag
            )
            continue
        if hg == ag and bet_type == "moneyline":
            _append_power_game(
                teams, power_games, home, away, home_name, away_name, hg, ag
            )
            continue
        prediction = _predict_blend_from_power_state(
            league, cutoff, home, away, teams, power_games, use_sport_layer=use_sport_layer
        )
        if prediction:
            blended_home, model_margin, power_margin, power_home, sport_home = prediction
            sample: dict[str, Any] = {
                "blended_home": blended_home,
                "model_margin": model_margin,
                "power_margin": power_margin,
                "power_home": power_home,
                "sport_home": sport_home,
                "home_goals": hg,
                "away_goals": ag,
            }
            sample.update(_closing_market_fields(league, game_date, home, away))
            samples.append(sample)
        _append_power_game(
            teams, power_games, home, away, home_name, away_name, hg, ag
        )
    return samples


def _backtest_samples(
    league: str,
    samples: list[dict[str, Any]],
    thresholds: dict[str, Any],
    bet_type: OfficialBetType,
) -> dict[str, Any]:
    units_total = 0.0
    wins = losses = pushes = bets = 0
    unit_deltas: list[float] = []
    for sample in samples:
        kwargs: dict[str, Any] = {
            "league": league,
            "bet_type": bet_type,
            "blended_home": sample.get("blended_home", sample.get("home_prob", 50.0)),
            "model_margin": sample.get("model_margin", 0.0),
            "power_margin": sample.get("power_margin"),
            "power_home": sample.get("power_home"),
            "home_goals": sample["home_goals"],
            "away_goals": sample["away_goals"],
            "thresholds": thresholds,
        }
        if bet_type == "soccer_1x2":
            kwargs.update(
                {
                    "home_prob": sample["home_prob"],
                    "draw_prob": sample["draw_prob"],
                    "away_prob": sample["away_prob"],
                    "market_home_odds": sample.get("market_home_odds"),
                    "market_draw_odds": sample.get("market_draw_odds"),
                    "market_away_odds": sample.get("market_away_odds"),
                }
            )
        elif bet_type == "spread":
            kwargs.update(
                {
                    "market_spread": sample.get("market_spread"),
                    "home_spread_odds": sample.get("home_spread_odds"),
                    "away_spread_odds": sample.get("away_spread_odds"),
                }
            )
        else:
            kwargs.update(
                {
                    "market_home_odds": sample.get("market_home_odds"),
                    "market_away_odds": sample.get("market_away_odds"),
                }
            )
        graded = _evaluate_backtest_pick(**kwargs)
        if graded is None:
            continue
        unit_delta, result = graded
        bets += 1
        units_total += unit_delta
        unit_deltas.append(unit_delta)
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            pushes += 1

    roi_pct = (units_total / bets * 100.0) if bets else 0.0
    return {
        "bet_type": bet_type,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "units": round(units_total, 2),
        "roi_pct": round(roi_pct, 2),
        "max_drawdown": round(_max_drawdown(unit_deltas), 2),
        "thresholds": thresholds,
    }


def backtest_league_strategy(
    league: str,
    cutoff: str,
    thresholds: dict[str, Any] | None = None,
    *,
    min_train: int = 40,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bet_type = thresholds.get("bet_type") if thresholds else official_bet_type(league)
    if bet_type is None or bet_type == "none":
        return {"bet_type": "none", "bets": 0, "roi_pct": 0.0, "units": 0.0}

    thresholds = dict(thresholds or get_pick_thresholds(league))
    thresholds["bet_type"] = bet_type

    if samples is None:
        samples = _collect_backtest_samples(league, cutoff, min_train=min_train)
    return _backtest_samples(league, samples, thresholds, bet_type)  # type: ignore[arg-type]


def _strategy_objective(result: dict[str, Any]) -> float:
    """Risk-adjusted walk-forward score: units minus drawdown penalty."""
    if result["bets"] < MIN_TUNE_BETS:
        return float("-inf")
    units = float(result["units"])
    max_dd = float(result.get("max_drawdown", 0.0))
    score = units - DRAWDOWN_PENALTY * max_dd
    if score <= 0:
        return score
    return score + result["bets"] * 0.02


def _threshold_grid(bet_type: OfficialBetType, league: str | None = None) -> list[dict[str, float]]:
    if bet_type == "spread":
        return [
            {
                "min_edge": edge,
                "min_spread_point_edge": pts,
                "min_ev_pct": MIN_EXPECTED_VALUE_PCT,
                "min_profit_score": profit,
                "min_kelly_pct": kelly,
            }
            for edge in (35, 40, 45)
            for pts in (2.0, 2.5, 3.0)
            for profit in (0.0, 5.0)
            for kelly in (0.0, 2.0)
        ]
    if league == "mlb":
        return [
            {
                "min_edge": edge,
                "min_ev_pct": ev,
                "min_spread_point_edge": 2.0,
                "min_profit_score": profit,
                "min_kelly_pct": kelly,
            }
            for edge in (35, 40, 45, 50, 55)
            for ev in (5.0, 6.0, 7.0, 8.0, 10.0)
            for profit in (0.0, 3.0, 5.0, 8.0, 12.0)
            for kelly in (0.0, 1.0, 2.0, 3.0, 5.0)
        ]
    if bet_type == "soccer_1x2":
        return [
            {
                "min_edge": edge,
                "min_ev_pct": ev,
                "min_spread_point_edge": 2.0,
                "min_profit_score": profit,
                "min_kelly_pct": kelly,
            }
            for edge in (35, 40, 45, 50)
            for ev in (5.0, 6.0, 7.0, 8.0)
            for profit in (0.0, 5.0, 8.0)
            for kelly in (0.0, 2.0, 3.0)
        ]
    return [
        {
            "min_edge": edge,
            "min_ev_pct": ev,
            "min_spread_point_edge": 2.0,
            "min_profit_score": profit,
            "min_kelly_pct": kelly,
        }
        for edge in (30, 35, 40)
        for ev in (4.0, 5.0, 6.0)
        for profit in (0.0, 5.0)
        for kelly in (0.0, 2.0)
    ]


def _refine_thresholds(
    league: str,
    samples: list[dict[str, Any]],
    bet_type: OfficialBetType,
    base: dict[str, Any],
) -> dict[str, Any]:
    current = dict(base)
    best = _backtest_samples(league, samples, current, bet_type)
    best_score = _strategy_objective(best)

    if bet_type == "spread":
        specs: list[tuple[str, float, float, float]] = [
            ("min_edge", 25.0, 55.0, 2.5),
            ("min_spread_point_edge", 1.0, 4.0, 0.25),
            ("min_profit_score", 0.0, 12.0, 1.0),
            ("min_kelly_pct", 0.0, 6.0, 0.5),
        ]
    elif league == "mlb":
        specs = [
            ("min_edge", 35.0, 70.0, 2.5),
            ("min_ev_pct", 4.0, 14.0, 0.5),
            ("min_profit_score", 0.0, 20.0, 1.0),
            ("min_kelly_pct", 0.0, 10.0, 0.5),
        ]
    elif bet_type == "soccer_1x2":
        specs = [
            ("min_edge", 30.0, 60.0, 2.5),
            ("min_ev_pct", 4.0, 12.0, 0.5),
            ("min_profit_score", 0.0, 15.0, 1.0),
            ("min_kelly_pct", 0.0, 8.0, 0.5),
        ]
    else:
        specs = [
            ("min_edge", 20.0, 50.0, 2.5),
            ("min_ev_pct", 2.0, 10.0, 0.5),
            ("min_profit_score", 0.0, 12.0, 1.0),
            ("min_kelly_pct", 0.0, 6.0, 0.5),
        ]

    for _ in range(1):
        improved = True
        while improved:
            improved = False
            for key, low, high, step in specs:
                value = float(current[key])
                for trial in (value - step, value + step):
                    if trial < low or trial > high:
                        continue
                    trial_thresholds = {**current, key: round(trial, 2)}
                    result = _backtest_samples(league, samples, trial_thresholds, bet_type)
                    score = _strategy_objective(result)
                    if score > best_score:
                        best_score = score
                        best = result
                        current = trial_thresholds
                        improved = True
    return best


def _enforce_positive_roi(
    league: str,
    samples: list[dict[str, Any]],
    bet_type: OfficialBetType,
    best: dict[str, Any],
) -> dict[str, Any]:
    """Tighten gates until walk-forward ROI is non-negative, or disable official picks."""
    if (
        best["roi_pct"] >= 0
        and float(best["units"]) >= 0
        and best["bets"] >= MIN_TUNE_BETS
    ):
        th = dict(best["thresholds"])
        th["enabled"] = True
        best["thresholds"] = th
        return best

    current = dict(best["thresholds"])
    best_result = best
    for edge in (55, 60, 65):
        for ev in (8.0, 11.0, 14.0):
            for profit in (10.0, 15.0, 20.0):
                for kelly in (3.0, 6.0):
                    trial = {
                        **current,
                        "min_edge": float(edge),
                        "min_ev_pct": float(ev),
                        "min_profit_score": float(profit),
                        "min_kelly_pct": float(kelly),
                        "bet_type": bet_type,
                    }
                    result = _backtest_samples(league, samples, trial, bet_type)
                    if (
                        result["roi_pct"] >= 0
                        and float(result["units"]) >= 0
                        and result["bets"] >= MIN_TUNE_BETS
                    ):
                        th = dict(result["thresholds"])
                        th["enabled"] = True
                        result["thresholds"] = th
                        return result
                    if _strategy_objective(result) > _strategy_objective(best_result):
                        best_result = result

    th = dict(best_result["thresholds"])
    th["enabled"] = (
        best_result["roi_pct"] >= 0
        and float(best_result["units"]) >= 0
        and best_result["bets"] >= MIN_TUNE_BETS
    )
    best_result["thresholds"] = th
    return best_result


def _tune_samples(samples: list[dict[str, Any]], max_samples: int = 1200) -> list[dict[str, Any]]:
    """Cap walk-forward pool for tuning speed while keeping recent history."""
    if len(samples) <= max_samples:
        return samples
    return samples[-max_samples:]


def tune_league_thresholds(league: str, cutoff: str) -> dict[str, Any]:
    bet_type = official_bet_type(league)
    if bet_type == "none":
        return _default_entry("none")

    samples = _tune_samples(
        _collect_backtest_samples(league, cutoff, use_sport_layer=True)
    )
    best: dict[str, Any] | None = None
    best_score = float("-inf")

    for params in _threshold_grid(bet_type, league):
        thresholds = {**DEFAULT_THRESHOLDS, **params, "bet_type": bet_type}
        result = _backtest_samples(league, samples, thresholds, bet_type)  # type: ignore[arg-type]
        score = _strategy_objective(result)
        if score > best_score:
            best_score = score
            best = result

    if not best or best_score == float("-inf"):
        thresholds = {**DEFAULT_THRESHOLDS, "bet_type": bet_type}
        best = _backtest_samples(league, samples, thresholds, bet_type)  # type: ignore[arg-type]
    else:
        best = _refine_thresholds(league, samples, bet_type, best["thresholds"])  # type: ignore[arg-type]

    best = _enforce_positive_roi(league, samples, bet_type, best)  # type: ignore[arg-type]
    th = best["thresholds"]
    return {
        "bet_type": bet_type,
        "min_edge": th["min_edge"],
        "min_ev_pct": th["min_ev_pct"],
        "min_spread_point_edge": th["min_spread_point_edge"],
        "min_profit_score": th.get("min_profit_score", 0.0),
        "min_kelly_pct": th.get("min_kelly_pct", 0.0),
        "enabled": bool(th.get("enabled", True)),
        "backtest_games": len(samples),
        "backtest_samples": len(samples),
        "backtest_roi_pct": best["roi_pct"],
        "backtest_bets": best["bets"],
        "backtest_wins": best["wins"],
        "backtest_losses": best["losses"],
        "backtest_units": best["units"],
        "backtest_max_drawdown": best.get("max_drawdown"),
    }


def evaluate_soccer_official_picks_for_game(
    *,
    league: str,
    away_name: str,
    home_name: str,
    away_slug: str,
    home_slug: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    away_proj: int,
    draw_proj: int,
    home_proj: int,
    away_market: int | None,
    draw_market: int | None,
    home_market: int | None,
    expected_home_goals: float | None = None,
    expected_away_goals: float | None = None,
    base_home_prob: float | None = None,
    base_draw_prob: float | None = None,
    base_away_prob: float | None = None,
) -> list[BetPick]:
    """Official 1X2 picks: Hubáček decorrelation gap vs market, honest EV."""
    thresholds = get_pick_thresholds(league)
    picks = evaluate_soccer_picks(
        away_name=away_name,
        home_name=home_name,
        away_slug=away_slug,
        home_slug=home_slug,
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
        away_proj=away_proj,
        draw_proj=draw_proj,
        home_proj=home_proj,
        away_market=away_market,
        draw_market=draw_market,
        home_market=home_market,
        base_home_prob=base_home_prob,
        base_draw_prob=base_draw_prob,
        base_away_prob=base_away_prob,
        hubacek_only=True,
        min_market_gap_pp=thresholds["min_market_gap_pp"],
        min_win_confidence_pp=thresholds["min_win_confidence_pp"],
        min_ev_pct=thresholds["min_ev_pct"],
    )
    allowed_sides = thresholds.get("allowed_sides")
    if allowed_sides:
        allowed = {str(side).lower() for side in allowed_sides}
        picks = [pick for pick in picks if pick.side in allowed]
    for pick in picks:
        pick.bet_type = "soccer_1x2"
    return picks


def evaluate_official_picks_for_game(
    *,
    league: str,
    away_name: str,
    home_name: str,
    away_slug: str,
    home_slug: str,
    total_score: float,
    win_probability: float,
    blended: dict[str, Any],
    away_market: int | None,
    home_market: int | None,
    consensus_spread: float | None,
    away_spread_odds: int | None,
    home_spread_odds: int | None,
    home_prob: float | None = None,
    away_prob: float | None = None,
) -> list[BetPick]:
    """Official picks: Hubáček decorrelation gap vs market (no flat EV% bar)."""
    thresholds = get_pick_thresholds(league)
    bet_type = thresholds["bet_type"]

    if bet_type == "spread":
        if consensus_spread is None:
            return []
        picks = evaluate_spread_picks(
            league=league,
            away_name=away_name,
            home_name=home_name,
            away_slug=away_slug,
            home_slug=home_slug,
            total_score=total_score,
            win_probability=win_probability,
            consensus_spread=consensus_spread,
            away_spread_odds=away_spread_odds,
            home_spread_odds=home_spread_odds,
            model_margin_home=blended_home_spread_margin(blended, league),
            hubacek_only=True,
            blended=blended,
            min_cover_gap_pp=thresholds["min_spread_cover_gap_pp"],
            min_win_confidence_pp=thresholds["min_spread_confidence_pp"],
        )
        return picks

    if bet_type == "moneyline":
        ml_away, ml_home, base_away, base_home = official_pick_binary_prob_sets(
            blended,
            total_score,
            league=league,
            away_market=away_market,
            home_market=home_market,
            consensus_spread=consensus_spread,
        )
        picks = evaluate_picks(
            away_name=away_name,
            home_name=home_name,
            away_slug=away_slug,
            home_slug=home_slug,
            total_score=total_score,
            win_probability=win_probability,
            away_market=away_market,
            home_market=home_market,
            away_prob=ml_away,
            home_prob=ml_home,
            base_away_prob=base_away,
            base_home_prob=base_home,
            hubacek_only=True,
            min_ev_pct=thresholds["min_ev_pct"],
            min_market_gap_pp=thresholds["min_market_gap_pp"],
            min_win_confidence_pp=thresholds["min_win_confidence_pp"],
            ml_lo=thresholds.get("ml_lo"),
            ml_hi=thresholds.get("ml_hi"),
        )
        return picks

    return []
