"""Daily slate analysis and bet recommendations across all supported leagues."""

from __future__ import annotations

import io
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORONTO = ZoneInfo("America/Toronto")
logger = logging.getLogger(__name__)


def _toronto_today() -> date:
    return datetime.now(TORONTO).date()


def _line_shopping_status() -> str:
    from web.live_odds_enrichment import line_shopping_status

    return line_shopping_status()


def _consensus_spread_for_preds(
    scoreboard_spread: float | None,
    book_enrichment: dict[str, Any] | None,
) -> float | None:
    """Prefer multi-book consensus home spread when finite; else scoreboard."""
    raw = (book_enrichment or {}).get("consensus_home_spread")
    # bool is a subclass of int; False→0.0 must not invent a pick'em line.
    if raw is not None and not isinstance(raw, bool):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None
        else:
            if math.isfinite(value):
                return value
    if scoreboard_spread is None or isinstance(scoreboard_spread, bool):
        return None
    try:
        board = float(scoreboard_spread)
    except (TypeError, ValueError):
        return None
    return board if math.isfinite(board) else None


from web.bet_advisor import (
    ensure_hubacek_in_blend,
    model_moneylines,
    official_pick_binary_probs,
    pick_to_dict,
    projections_from_win_probs,
    soccer_model_moneylines,
)
from web.baseball_pred_model import get_baseball_pred_context, is_baseball_league  # noqa: E402
from web.basketball_pred_model import get_basketball_pred_context, is_basketball_league  # noqa: E402
from web.hockey_pred_model import is_hockey_league  # noqa: E402
from web.football_pred_model import get_football_pred_context, is_football_league  # noqa: E402
from web.league_profiles import is_soccer_league  # noqa: E402
from web.soccer_pred_model import get_soccer_pred_context  # noqa: E402
from web.blend_service import blend_predictions, compute_model_agreement  # noqa: E402
from web.ensemble_ml import apply_ensemble_ml  # noqa: E402
from web.season_games import prewarm_league_power  # noqa: E402
from web.espn_client import (  # noqa: E402
    ScheduledGame,
    current_season_year,
    fetch_scoreboard,
    iso_to_project_date,
)
from web.hubacek_picks import (
    HUBACEK_MIN_MARKET_GAP_PP,
    official_hubacek_thresholds,
    passes_hubacek_tracked_pick,
)
from web.league_profiles import (  # noqa: E402
    LEAGUE_PROFILES,
    SUPPORTED_LEAGUES,
    eligible_for_official_picks,
    get_algo_league,
)
from web.league_readiness import assess_league_readiness  # noqa: E402
from web.live_data import load_live_team_data, resolve_team  # noqa: E402
from web.pick_strategy import (
    evaluate_official_picks_for_game,
    evaluate_soccer_official_picks_for_game,
    get_pick_thresholds,
    official_bet_type,
)
from web.predict_service import FACTOR_LABELS  # noqa: E402

SINGLE_MODEL_BLEND_MODES = frozenset(
    {
        "basketball_matrix",
        "mlb_runcast",
        "mlb_v2",
        "soccer_path_a",
        "soccer_v2",
        "wnba_v2",
        "nba_v2",
        "nhl_v2",
        "nfl_v2",
        "cfb_v2",
        "cbb_torvik",
        "cbb_v2",
    }
)


def _factor_entry(
    key: str,
    label: str,
    value: float,
    *,
    positive_favors: str = "home",
) -> dict[str, Any]:
    """Build a factor row. Sport margins are home-positive; Algo totals are away-positive."""
    if value == 0:
        favors = "neutral"
    elif value > 0:
        favors = positive_favors
    else:
        favors = "away" if positive_favors == "home" else "home"
    return {
        "key": key,
        "label": label,
        "value": value,
        "favors": favors,
    }


def _sport_layer_factors(blended: dict[str, Any], league: str) -> list[dict[str, Any]]:
    """Factors from the active sport model (not legacy Algo V2)."""
    league = league.lower()
    factors: list[dict[str, Any]] = []

    if is_hockey_league(league):
        pred = blended.get("hockey_pred") or {}
        home_xg = pred.get("expected_home_goals")
        away_xg = pred.get("expected_away_goals")
        if home_xg is not None and away_xg is not None:
            factors.append(
                _factor_entry(
                    "xg_margin",
                    "Expected goals (home − away)",
                    float(home_xg) - float(away_xg),
                )
            )
        margin = pred.get("predicted_margin")
        if margin is not None:
            factors.append(
                _factor_entry("score_margin", "Projected goal margin (home)", float(margin))
            )
        home_elo = pred.get("home_elo")
        away_elo = pred.get("away_elo")
        if home_elo is not None and away_elo is not None:
            factors.append(
                _factor_entry(
                    "elo_edge",
                    "Elo edge (home − away)",
                    round(float(home_elo) - float(away_elo), 1),
                )
            )
        home_xgf = pred.get("home_xgf_pg")
        away_xgf = pred.get("away_xgf_pg")
        if home_xgf is not None and away_xgf is not None:
            factors.append(
                _factor_entry(
                    "xgf_edge",
                    "xGF/60 edge (home − away)",
                    round(float(home_xgf) - float(away_xgf), 3),
                )
            )
        home_rest = pred.get("home_rest_days")
        away_rest = pred.get("away_rest_days")
        if home_rest is not None and away_rest is not None:
            rest_edge = float(home_rest) - float(away_rest)
            if abs(rest_edge) >= 1.0:
                factors.append(
                    _factor_entry("rest_edge", "Rest days edge (home − away)", rest_edge)
                )
        if pred.get("market_decorrelated") or blended.get("market_decorrelated"):
            factors.append(_factor_entry("decorrelation", "Market decorrelation applied", 1.0))
        return factors

    if league == "mlb":
        pred = blended.get("baseball_pred") or {}
        margin = pred.get("predicted_margin")
        if margin is not None:
            factors.append(
                _factor_entry("run_margin", "Projected run margin (home)", float(margin))
            )
        pitcher = pred.get("pitcher_margin")
        if pitcher is not None:
            factors.append(
                _factor_entry("starter_edge", "Probable starter edge", float(pitcher) * 12.0)
            )
        home_runs = pred.get("predicted_home_runs")
        away_runs = pred.get("predicted_away_runs")
        if home_runs is not None and away_runs is not None:
            factors.append(
                _factor_entry(
                    "projected_runs",
                    "Projected runs (home − away)",
                    float(home_runs) - float(away_runs),
                )
            )
        raw = pred.get("raw_home_win_probability")
        calibrated = pred.get("home_win_probability")
        if raw is not None and calibrated is not None:
            factors.append(
                _factor_entry(
                    "calibration_shift",
                    "Calibration shift (pp)",
                    float(calibrated) - float(raw),
                )
            )
        if pred.get("market_decorrelated") or blended.get("market_decorrelated"):
            factors.append(_factor_entry("decorrelation", "Market decorrelation applied", 1.0))
        return factors

    if is_basketball_league(league):
        pred = blended.get("basketball_pred") or {}
        home_or = pred.get("predicted_home_offensive_rating")
        away_or = pred.get("predicted_away_offensive_rating")
        pace = pred.get("predicted_pace")
        if home_or is not None and away_or is not None:
            factors.append(
                _factor_entry(
                    "offensive_rating",
                    "Offensive rating (home − away)",
                    float(home_or) - float(away_or),
                )
            )
        if pace is not None:
            factors.append(_factor_entry("pace", "Projected pace (possessions)", float(pace)))
        margin = pred.get("predicted_margin")
        if margin is not None:
            factors.append(
                _factor_entry("score_margin", "Projected score margin (home)", float(margin))
            )
        home_score = pred.get("predicted_home_score")
        away_score = pred.get("predicted_away_score")
        if home_score is not None and away_score is not None:
            factors.append(
                _factor_entry(
                    "projected_total",
                    "Projected total points",
                    float(home_score) + float(away_score),
                )
            )
        home_elo = pred.get("home_elo")
        away_elo = pred.get("away_elo")
        if home_elo is not None and away_elo is not None:
            factors.append(
                _factor_entry(
                    "elo_edge",
                    "Elo edge (home − away)",
                    round(float(home_elo) - float(away_elo), 1),
                )
            )
        home_net = pred.get("home_net_rtg")
        away_net = pred.get("away_net_rtg")
        if home_net is not None and away_net is not None:
            factors.append(
                _factor_entry(
                    "net_rating",
                    "Net rating edge (home − away)",
                    round(float(home_net) - float(away_net), 2),
                )
            )
        if pace is None and pred.get("home_pace") is not None and pred.get("away_pace") is not None:
            factors.append(
                _factor_entry(
                    "pace",
                    "Projected pace (possessions)",
                    round((float(pred["home_pace"]) + float(pred["away_pace"])) / 2.0, 1),
                )
            )
        home_rest = pred.get("home_rest_days")
        away_rest = pred.get("away_rest_days")
        if home_rest is not None and away_rest is not None:
            rest_edge = float(home_rest) - float(away_rest)
            if abs(rest_edge) >= 1.0 or pred.get("home_b2b") or pred.get("away_b2b"):
                factors.append(
                    _factor_entry("rest_edge", "Rest days edge (home − away)", rest_edge)
                )
        return factors

    if is_soccer_league(league):
        pred = blended.get("soccer_pred") or {}
        home_xg = pred.get("expected_home_goals")
        away_xg = pred.get("expected_away_goals")
        if home_xg is not None and away_xg is not None:
            factors.append(
                _factor_entry("xg_margin", "Expected goals (home − away)", float(home_xg) - float(away_xg))
            )
        pi_gd = pred.get("pi_expected_gd")
        if pi_gd is not None:
            factors.append(_factor_entry("pi_edge", "Pi-rating expected GD", float(pi_gd)))
        if pred.get("market_decorrelated"):
            factors.append(_factor_entry("decorrelation", "Market decorrelation applied", 1.0))
        raw_home = pred.get("raw_home_win_probability")
        calibrated = pred.get("home_win_probability")
        if raw_home is not None and calibrated is not None:
            factors.append(
                _factor_entry(
                    "calibration_shift",
                    "Calibration shift home (pp)",
                    float(calibrated) - float(raw_home),
                )
            )
        return factors

    return factors


def _build_model_factors(
    blended: dict[str, Any],
    algo_data: dict[str, Any],
    league: str,
) -> list[dict[str, Any]]:
    if (blended.get("blend_mode") or "") in SINGLE_MODEL_BLEND_MODES:
        return _sport_layer_factors(blended, league)
    factors: list[dict[str, Any]] = []
    for key, label in FACTOR_LABELS.items():
        if key not in algo_data:
            continue
        # Algo convention: positive factor totals favor the away team.
        factors.append(
            _factor_entry(key, label, float(algo_data[key]), positive_favors="away")
        )
    return factors


def _ensure_project_root() -> None:
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _today_cutoff(game: ScheduledGame) -> str:
    if game.start_time:
        return iso_to_project_date(game.start_time)
    today = _toronto_today()
    return f"{today.month}-{today.day}-{today.year}"


def _iso_match_date(start_time: str | None) -> str:
    """Kickoff calendar date as YYYY-MM-DD in America/Toronto (soccer rest features)."""
    if not start_time:
        return ""
    try:
        parsed = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(TORONTO).date().isoformat()


def _is_actionable_soon(game: ScheduledGame, horizon_days: int = 3) -> bool:
    if not game.start_time:
        return True
    start = datetime.fromisoformat(game.start_time.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return now - timedelta(days=1) <= start <= now + timedelta(days=horizon_days)


def _season_year_from_cutoff(league: str, cutoff_date: str) -> str:
    month, day, year = cutoff_date.split("-")
    season = current_season_year(league, date(int(year), int(month), int(day)))
    return str(season)


def _enrich_pick_with_team_abbr(
    pick_dict: dict[str, Any], matchup: dict[str, Any]
) -> dict[str, Any]:
    side = pick_dict.get("side")
    if side in ("away", "home"):
        abbr = (matchup.get(side) or {}).get("abbr")
        if abbr:
            return {**pick_dict, "team_abbr": abbr}
    return pick_dict


def predict_live_game(
    game: ScheduledGame,
    headlines: list[str] | None = None,
) -> dict[str, Any]:
    _ensure_project_root()
    from algo import Algo
    from odds_calculator import Odds_Calculator

    away = resolve_team(game.league, game.away_abbr, game.away_name)
    home = resolve_team(game.league, game.home_abbr, game.home_name)
    if not away or not home:
        raise ValueError(
            f"Unknown teams for {game.league}: {game.away_abbr} @ {game.home_abbr}"
        )

    cutoff = _today_cutoff(game)
    season_year = _season_year_from_cutoff(game.league, cutoff)

    data_away = load_live_team_data(game.league, away, game.away_espn_id, cutoff)
    data_home = load_live_team_data(game.league, home, game.home_espn_id, cutoff)

    if not data_away or not data_home:
        raise ValueError(
            f"Insufficient season data for {away[1]} or {home[1]} before {cutoff}."
        )

    # Multi-book enrichment fetched ONCE per game (needs only league +
    # event_id). Opens feed the steam context signal below and the same
    # payload is reused for the final market dict — never fetched twice.
    book_enrichment: dict[str, Any] = {}
    try:
        from web.live_odds_enrichment import fetch_multi_book_odds

        book_enrichment = fetch_multi_book_odds(game.league, game.event_id) or {}
    except Exception:  # noqa: BLE001 — never block slate on books
        book_enrichment = {}

    consensus_spread = _consensus_spread_for_preds(
        game.market.spread, book_enrichment
    )

    algo_league = get_algo_league(game.league)
    odds_calculator = Odds_Calculator(algo_league)
    algo = Algo(algo_league)

    with redirect_stdout(io.StringIO()):
        returned_away = odds_calculator.analyze2(away, home, data_away, "away")
        returned_home = odds_calculator.analyze2(home, away, data_home, "home")
        if is_hockey_league(game.league):
            algo_data = algo.calculate(cutoff, returned_away, returned_home)
        else:
            algo_data = algo.calculate_V2(cutoff, returned_away, returned_home)

    legacy_total = float(algo_data["total"])
    if is_hockey_league(game.league):
        legacy_win_probability = float(odds_calculator.get_odds(legacy_total))
    else:
        legacy_win_probability = abs(legacy_total)

    blended = blend_predictions(
        legacy_total_score=legacy_total,
        legacy_win_probability=legacy_win_probability,
        league=game.league,
        cutoff_date=cutoff,
        home_abbr=home[0],
        away_abbr=away[0],
        home_name=game.home_name,
        away_name=game.away_name,
        event_id=game.event_id,
        home_espn_id=game.home_espn_id,
        away_espn_id=game.away_espn_id,
        home_slug=home[1],
        away_slug=away[1],
        consensus_spread=consensus_spread,
        home_moneyline=game.market.home_moneyline,
        away_moneyline=game.market.away_moneyline,
        draw_moneyline=game.market.draw_moneyline,
        headlines=headlines,
        match_date=_iso_match_date(game.start_time),
        kickoff_iso=game.start_time or "",
    )
    # Context layer hook: specialized blend paths (NBA/MLB/NHL/soccer) skip
    # `_with_db_rating_layer`; apply once here when not already attached.
    # Soccer v2 / Path-A ESPN context lock probs via context_adjustment_pp.
    if (
        blended.get("context_adjustment_pp") is None
        and not blended.get("context_adjusted")
        and (blended.get("blend_mode") or "") != "soccer_v2"
    ):
        from web.context_signals import apply_context_to_blend

        context_market: dict[str, Any] = {
            "home_moneyline": game.market.home_moneyline,
            "away_moneyline": game.market.away_moneyline,
            "draw_moneyline": game.market.draw_moneyline,
        }
        # Multi-book consensus opens activate steam_line_movement_shift.
        if book_enrichment.get("open_home_moneyline") is not None:
            context_market["open_home_moneyline"] = book_enrichment["open_home_moneyline"]
        if book_enrichment.get("open_away_moneyline") is not None:
            context_market["open_away_moneyline"] = book_enrichment["open_away_moneyline"]
        blended = apply_context_to_blend(
            blended,
            market=context_market,
            headlines=headlines,
            home_names=[game.home_name, home[0]],
            away_names=[game.away_name, away[0]],
            league=game.league,
        )
    blended = apply_ensemble_ml(
        blended,
        game.league,
        consensus_spread=consensus_spread,
        home_moneyline=game.market.home_moneyline,
        away_moneyline=game.market.away_moneyline,
        draw_moneyline=game.market.draw_moneyline,
    )
    blended = ensure_hubacek_in_blend(
        blended,
        league=game.league,
        away_market=game.market.away_moneyline,
        home_market=game.market.home_moneyline,
        consensus_spread=consensus_spread,
    )

    if is_soccer_league(game.league):
        from web.soccer_paper_tracking import maybe_record_from_blend

        maybe_record_from_blend(
            blended,
            league=game.league,
            event_id=game.event_id,
            home_abbr=home[0],
            away_abbr=away[0],
            home_name=game.home_name,
            away_name=game.away_name,
            game_date=_iso_match_date(game.start_time) or _toronto_today().isoformat(),
            home_ml=game.market.home_moneyline,
            draw_ml=game.market.draw_moneyline,
            away_ml=game.market.away_moneyline,
        )

    total = float(blended["total_score"])
    win_probability = float(blended["win_probability"])
    favorite_side = blended["favorite_side"]

    if blended.get("threeway"):
        home_prob = float(blended["home_win_probability"])
        draw_prob = float(blended["draw_probability"])
        away_prob = float(blended["away_win_probability"])
        away_proj, draw_proj, home_proj = soccer_model_moneylines(
            home_prob, draw_prob, away_prob
        )
    elif blended.get("blended_home_win_probability") is not None:
        home_prob = float(blended["blended_home_win_probability"])
        away_prob = 100.0 - home_prob
        away_proj, home_proj = projections_from_win_probs(home_prob, away_prob)
        draw_proj = None
    else:
        away_proj, home_proj = model_moneylines(total)
        draw_proj = None

    factors = _build_model_factors(blended, algo_data, game.league)

    market_payload = {
        "away_moneyline": game.market.away_moneyline,
        "home_moneyline": game.market.home_moneyline,
        "draw_moneyline": game.market.draw_moneyline,
        "spread": game.market.spread,
        "away_spread_odds": game.market.away_spread_odds,
        "home_spread_odds": game.market.home_spread_odds,
    }
    model_agreement = compute_model_agreement(
        blended, game.league, market=market_payload
    )
    pick_thresholds = get_pick_thresholds(game.league)
    ml_away_prob, ml_home_prob = official_pick_binary_probs(
        blended,
        total,
        league=game.league,
        away_market=game.market.away_moneyline,
        home_market=game.market.home_moneyline,
        consensus_spread=game.market.spread,
    )

    model_payload: dict[str, Any] = {
        **blended,
        "model_agreement": model_agreement,
        "pick_strategy": pick_thresholds,
        "away_projection": away_proj,
        "home_projection": home_proj,
        "factors": factors,
    }
    if draw_proj is not None:
        model_payload["draw_projection"] = draw_proj
    if (
        (blended.get("hockey_pred") or {}).get("live_inputs_stale")
        or (blended.get("soccer_pred") or {}).get("live_inputs_stale")
        or (blended.get("baseball_pred") or {}).get("live_inputs_stale")
        or (blended.get("basketball_pred") or {}).get("live_inputs_stale")
        or (blended.get("football_pred") or {}).get("live_inputs_stale")
    ):
        model_payload["live_inputs_stale"] = True

    if is_soccer_league(game.league):
        soccer_pred = blended.get("soccer_pred")
        # Unavailable Path A stub invents flat 33/33/33 for UI only — never
        # emit Hubáček official picks from a null model.
        if (
            blended.get("blend_mode") == "soccer_path_a_unavailable"
            or not isinstance(soccer_pred, dict)
            or int(blended.get("blend_layers") or 0) <= 0
        ):
            picks = []
        else:
            pick_home = float(
                soccer_pred.get("pick_home_win_probability", home_prob)
            )
            pick_draw = float(soccer_pred.get("pick_draw_probability", draw_prob))
            pick_away = float(
                soccer_pred.get("pick_away_win_probability", away_prob)
            )
            pick_away_proj, pick_draw_proj, pick_home_proj = soccer_model_moneylines(
                pick_home, pick_draw, pick_away
            )
            from web.context_signals import games_played_proxy_from_blend

            picks = evaluate_soccer_official_picks_for_game(
                league=game.league,
                away_name=game.away_name,
                home_name=game.home_name,
                away_slug=away[1],
                home_slug=home[1],
                home_prob=pick_home,
                draw_prob=pick_draw,
                away_prob=pick_away,
                away_proj=pick_away_proj,
                draw_proj=pick_draw_proj,
                home_proj=pick_home_proj,
                away_market=game.market.away_moneyline,
                draw_market=game.market.draw_moneyline,
                home_market=game.market.home_moneyline,
                expected_home_goals=blended.get("expected_home_goals")
                or soccer_pred.get("expected_home_goals"),
                expected_away_goals=blended.get("expected_away_goals")
                or soccer_pred.get("expected_away_goals"),
                base_home_prob=home_prob,
                base_draw_prob=draw_prob,
                base_away_prob=away_prob,
                games_played_proxy=games_played_proxy_from_blend(blended),
            )
    else:
        # Unavailable sport stubs invent 50/50 for UI only — never emit Hubáček
        # official picks from a null model (decorrelation can fake a gap vs book).
        blend_mode = str(blended.get("blend_mode") or "")
        layers = blended.get("blend_layers")
        unavailable = blend_mode.endswith("_unavailable")
        if not unavailable and layers is not None:
            try:
                unavailable = int(layers) <= 0
            except (TypeError, ValueError):
                unavailable = True
        if unavailable:
            picks = []
        else:
            picks = evaluate_official_picks_for_game(
                league=game.league,
                away_name=game.away_name,
                home_name=game.home_name,
                away_slug=away[1],
                home_slug=home[1],
                total_score=total,
                win_probability=win_probability,
                blended=blended,
                away_market=game.market.away_moneyline,
                home_market=game.market.home_moneyline,
                consensus_spread=consensus_spread,
                away_spread_odds=game.market.away_spread_odds,
                home_spread_odds=game.market.home_spread_odds,
                home_prob=ml_home_prob,
                away_prob=ml_away_prob,
            )

    # Full model picks stay on the game card for every league. Official tracking
    # and the slate recommended_bets rollup still gate on eligible_for_official_picks.
    model_picks = picks

    matchup = {
        "away": {"abbr": away[0], "slug": away[1], "name": game.away_name},
        "home": {"abbr": home[0], "slug": home[1], "name": game.home_name},
    }

    market_dict: dict[str, Any] = {
        "provider": game.market.provider,
        "away_moneyline": game.market.away_moneyline,
        "home_moneyline": game.market.home_moneyline,
        "draw_moneyline": game.market.draw_moneyline,
        "spread": consensus_spread,
        "away_spread_odds": game.market.away_spread_odds,
        "home_spread_odds": game.market.home_spread_odds,
        "over_under": game.market.over_under,
    }
    try:
        from web.live_odds_enrichment import apply_enrichment_to_market

        # Reuse the enrichment fetched at the top of this function.
        market_dict = apply_enrichment_to_market(market_dict, book_enrichment)
    except Exception:  # noqa: BLE001 — never block slate on books
        pass

    def _enrich_pick(pick_dict: dict[str, Any]) -> dict[str, Any]:
        from web.live_odds_enrichment import line_shopping_fields_for_pick

        enriched = _enrich_pick_with_team_abbr(pick_dict, matchup)
        # Honest EV stays on ESPN; parallel EV at the shopped price uses the
        # calibrated (base) probability when available.
        model_prob = enriched.get("base_win_probability")
        if model_prob is None:
            model_prob = enriched.get("win_probability")
        shopping = line_shopping_fields_for_pick(
            market_dict,
            side=str(enriched.get("side") or ""),
            bet_type=str(enriched.get("bet_type") or "moneyline"),
            model_prob_pct=float(model_prob) if model_prob is not None else None,
        )
        return {**enriched, **shopping} if shopping else enriched

    return {
        "event_id": game.event_id,
        "league": game.league,
        "league_name": LEAGUE_PROFILES[game.league]["name"],
        "name": game.name,
        "start_time": game.start_time,
        "status": game.status,
        "status_detail": game.status_detail,
        "cutoff_date": cutoff,
        "season_year": season_year,
        "matchup": matchup,
        "market": market_dict,
        "model": model_payload,
        "eligible_for_official_picks": eligible_for_official_picks(game.league),
        "official_bet_type": official_bet_type(game.league),
        "recommendations": [_enrich_pick(pick_to_dict(pick)) for pick in model_picks],
        "top_pick": _enrich_pick(pick_to_dict(model_picks[0])) if model_picks else None,
    }


def _slate_cutoff_date() -> str:
    today = _toronto_today()
    return f"{today.month}-{today.day}-{today.year}"


def _actionable_games(scheduled: list[ScheduledGame]) -> list[ScheduledGame]:
    return [
        game
        for game in scheduled
        if game.status not in {"in", "post"} and _is_actionable_soon(game)
    ]


# Max headlines fed into the news context signal per league.
_NEWS_HEADLINES_PER_LEAGUE = 40

# Parallel ESPN scoreboard workers (I/O bound; throttle still spaces starts).
_SCOREBOARD_WORKERS = 6

# Safe, non-exception strings for the public slate `errors` payload.
_PUBLIC_SLATE_ERRORS = {
    "scoreboard": "Scoreboard unavailable.",
    "prewarm": "Model prewarm failed.",
    "power_prewarm": "Power ratings prewarm failed.",
    "v2_prewarm": "GradientBoost v2 prewarm failed.",
    "sport_prewarm": "Sport model prewarm failed.",
    "predict": "Game prediction failed.",
    "league": "League slate failed.",
    "not_ready": "League not ready for predictions.",
}


def _public_error(
    league: str,
    kind: str,
    *,
    game: str | None = None,
) -> dict[str, str]:
    row: dict[str, str] = {
        "league": league,
        "error": _PUBLIC_SLATE_ERRORS.get(kind, "Processing failed."),
    }
    if game:
        row["game"] = game
    return row


def _prewarm_league_models(
    league: str,
    cutoffs: set[str],
    errors: list[dict[str, str]],
) -> None:
    for cutoff in cutoffs:
        try:
            prewarm_league_power(league, cutoff)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Power prewarm failed for %s (%s): %s", league, cutoff, exc)
            errors.append(_public_error(league, "power_prewarm"))
        if is_basketball_league(league):
            v2_ready = False
            if league.lower() == "wnba":
                try:
                    from web.hockey_pred_model import _cutoff_to_iso as _wnba_cutoff_to_iso
                    from web.wnba_v2.live import artifacts_available as _wnba_v2_available
                    from web.wnba_v2.live import get_live_context as _wnba_live_context

                    day_iso = _wnba_cutoff_to_iso(cutoff)
                    if day_iso and _wnba_v2_available():
                        v2_ready = _wnba_live_context(day_iso) is not None
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "WNBA v2 prewarm failed for %s (%s): %s", league, cutoff, exc
                    )
                    errors.append(_public_error(league, "v2_prewarm"))
            elif league.lower() == "nba":
                try:
                    from web.hockey_pred_model import _cutoff_to_iso as _nba_cutoff_to_iso
                    from web.nba_v2.live import artifacts_available as _nba_v2_available
                    from web.nba_v2.live import get_live_context as _nba_live_context

                    day_iso = _nba_cutoff_to_iso(cutoff)
                    if day_iso and _nba_v2_available():
                        v2_ready = _nba_live_context(day_iso) is not None
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "NBA v2 prewarm failed for %s (%s): %s", league, cutoff, exc
                    )
                    errors.append(_public_error(league, "v2_prewarm"))
            if not v2_ready:
                try:
                    get_basketball_pred_context(league, cutoff)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Basketball matrix prewarm failed for %s (%s): %s",
                        league,
                        cutoff,
                        exc,
                    )
                    errors.append(_public_error(league, "sport_prewarm"))
        if is_baseball_league(league):
            try:
                get_baseball_pred_context(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Baseball model prewarm failed for %s (%s): %s", league, cutoff, exc
                )
                errors.append(_public_error(league, "sport_prewarm"))
        if league.lower() == "mlb":
            try:
                from web.mlb_pred_model import _cutoff_to_iso
                from web.mlb_v2.live import get_live_context

                day_iso = _cutoff_to_iso(cutoff)
                if day_iso:
                    get_live_context(day_iso)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MLB v2 prewarm failed for %s (%s): %s", league, cutoff, exc
                )
                errors.append(_public_error(league, "v2_prewarm"))
        if league.lower() == "nhl":
            try:
                from web.hockey_pred_model import _cutoff_to_iso as _nhl_cutoff_to_iso
                from web.nhl_v2.live import get_live_context as _nhl_live_context

                day_iso = _nhl_cutoff_to_iso(cutoff)
                if day_iso:
                    _nhl_live_context(day_iso)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NHL v2 prewarm failed for %s (%s): %s", league, cutoff, exc
                )
                errors.append(_public_error(league, "v2_prewarm"))
        if is_football_league(league):
            try:
                get_football_pred_context(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Football model prewarm failed for %s (%s): %s", league, cutoff, exc
                )
                errors.append(_public_error(league, "sport_prewarm"))
        if is_soccer_league(league):
            soccer_v2_ready = False
            if league.lower() in ("epl", "bundesliga", "laliga", "seriea", "ligue1"):
                try:
                    from web.soccer_pred_model import _cutoff_to_iso as _soc_cutoff_to_iso
                    from web.soccer_v2.live import artifacts_available
                    from web.soccer_v2.live import get_live_context as _soc_live_context

                    day_iso = _soc_cutoff_to_iso(cutoff)
                    if day_iso and artifacts_available():
                        soccer_v2_ready = _soc_live_context(day_iso) is not None
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Soccer v2 prewarm failed for %s (%s): %s", league, cutoff, exc
                    )
                    errors.append(_public_error(league, "v2_prewarm"))
            if not soccer_v2_ready:
                try:
                    get_soccer_pred_context(league, cutoff)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Soccer model prewarm failed for %s (%s): %s",
                        league,
                        cutoff,
                        exc,
                    )
                    errors.append(_public_error(league, "sport_prewarm"))


def _news_signals_enabled() -> bool:
    """NEWS_SIGNALS env flag; default on, "0"/"false"/"no"/"off" disables."""
    flag = (os.environ.get("NEWS_SIGNALS") or "").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _league_news_headlines(
    league: str,
    limit: int = _NEWS_HEADLINES_PER_LEAGUE,
) -> list[str]:
    """One soft-fail ESPN news fetch per league; plain headline strings.

    Returns [] on any error or when NEWS_SIGNALS is disabled — league news
    must never break or noticeably slow the slate build.
    """
    if not _news_signals_enabled():
        return []
    try:
        from web.sports_db.espn_fetch import fetch_league_news

        payload = fetch_league_news(league, limit=limit)
    except Exception:  # noqa: BLE001 — news never blocks the slate
        return []
    if not payload:
        return []
    headlines: list[str] = []
    for article in payload.get("articles") or []:
        if not isinstance(article, dict):
            continue
        headline = article.get("headline") or article.get("title")
        if isinstance(headline, str) and headline.strip():
            headlines.append(headline.strip())
        if len(headlines) >= limit:
            break
    return headlines


def get_daily_slate(days_ahead: int = 0) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    all_games: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    leagues_not_ready: list[dict[str, str]] = []
    slate_cutoff = _slate_cutoff_date()
    logger.info("Building daily slate (days_ahead=%s)", days_ahead)

    # Fresh multi-book wall-time budget for this build (long-lived uvicorn
    # otherwise permanently exhausts the process-global accumulator).
    try:
        from web.live_odds_enrichment import reset_enrichment_budget

        reset_enrichment_budget()
    except Exception:  # noqa: BLE001 — never block the slate on budget reset
        pass

    # Fetch all league scoreboards in parallel (I/O-bound). Processing stays
    # sequential so model caches and pick logic remain deterministic.
    def _fetch_one(league: str) -> tuple[str, Any]:
        try:
            # Match date_label / slate cutoffs (America/Toronto), not UTC server today.
            return league, fetch_scoreboard(
                league, on_date=_toronto_today(), days_ahead=days_ahead
            )
        except Exception as exc:  # noqa: BLE001
            return league, exc

    scoreboards: dict[str, Any] = {}
    workers = max(1, min(_SCOREBOARD_WORKERS, len(SUPPORTED_LEAGUES)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_one, league) for league in SUPPORTED_LEAGUES]
        for fut in as_completed(futures):
            league, payload = fut.result()
            scoreboards[league] = payload

    for league in SUPPORTED_LEAGUES:
        # Soft-fail every league independently: scoreboard, readiness, prewarm,
        # or a single game must never abort the rest of the slate.
        scheduled_or_exc = scoreboards.get(league)
        if isinstance(scheduled_or_exc, BaseException):
            errors.append(_public_error(league, "scoreboard"))
            logger.warning(
                "Scoreboard failed for %s: %s", league, scheduled_or_exc
            )
            continue
        if scheduled_or_exc is None:
            errors.append(_public_error(league, "scoreboard"))
            continue

        scheduled = scheduled_or_exc
        try:
            actionable = _actionable_games(scheduled)
            if not actionable:
                continue

            readiness = assess_league_readiness(league, slate_cutoff)
            if not readiness.get("ready"):
                reason = str(
                    readiness.get("reason") or "Insufficient season data for predictions."
                )
                leagues_not_ready.append({"league": league, "reason": reason})
                logger.info(
                    "Skipping %s — not ready (%s)", league, reason
                )
                continue

            power_cutoffs = {_today_cutoff(game) for game in actionable}
            try:
                _prewarm_league_models(league, power_cutoffs, errors)
            except Exception as exc:  # noqa: BLE001 — prewarm must not skip games
                errors.append(_public_error(league, "prewarm"))
                logger.warning("Model prewarm failed for %s: %s", league, exc)

            headlines = _league_news_headlines(league)
            league_games = 0

            for game in actionable:
                try:
                    all_games.append(predict_live_game(game, headlines=headlines))
                    league_games += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        _public_error(league, "predict", game=game.name)
                    )
                    logger.warning(
                        "Game predict failed for %s (%s): %s",
                        league,
                        game.name,
                        exc,
                    )
            if league_games:
                logger.info("Slate %s: %s game(s)", league, league_games)
        except Exception as exc:  # noqa: BLE001
            errors.append(_public_error(league, "league"))
            logger.warning("League slate failed for %s: %s", league, exc)
            continue

    recommendations = []
    for game in all_games:
        # Fail closed: missing eligibility must not enter the official book.
        if not game.get("eligible_for_official_picks", False):
            continue
        for pick in game.get("recommendations") or []:
            recommendations.append(
                {
                    **pick,
                    "league": game["league"],
                    "league_name": game["league_name"],
                    "event_id": game["event_id"],
                    "matchup": f"{game['matchup']['away']['name']} @ {game['matchup']['home']['name']}",
                    "start_time": game["start_time"],
                    "tracked": True,
                }
            )

    model_analysis = []
    for game in all_games:
        if game.get("eligible_for_official_picks", False):
            continue
        for pick in game.get("recommendations") or []:
            model_analysis.append(
                {
                    **pick,
                    "league": game["league"],
                    "league_name": game["league_name"],
                    "event_id": game["event_id"],
                    "matchup": f"{game['matchup']['away']['name']} @ {game['matchup']['home']['name']}",
                    "start_time": game["start_time"],
                    "tracked": False,
                }
            )

    def _best_picks_by_event(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best_by_event: dict[str, dict[str, Any]] = {}
        for rec in candidates:
            event_id = rec.get("event_id") or ""
            if not event_id:
                continue
            current = best_by_event.get(event_id)
            if current is None or (
                rec.get("profit_score", 0),
                rec.get("ev_pct", 0),
                rec.get("edge", 0),
            ) > (
                current.get("profit_score", 0),
                current.get("ev_pct", 0),
                current.get("edge", 0),
            ):
                best_by_event[event_id] = rec
        return sorted(
            best_by_event.values(),
            key=lambda item: (
                item.get("profit_score", 0),
                item.get("ev_pct", 0),
                item.get("edge", 0),
            ),
            reverse=True,
        )

    def _meets_recommendation_threshold(_game: dict[str, Any], rec: dict[str, Any]) -> bool:
        return passes_hubacek_tracked_pick(rec)

    games_by_event = {game["event_id"]: game for game in all_games}
    recommendations = _best_picks_by_event(recommendations)
    model_analysis = _best_picks_by_event(model_analysis)

    qualifying = [
        r
        for r in recommendations
        if _meets_recommendation_threshold(games_by_event.get(r.get("event_id", ""), {}), r)
    ]
    model_analysis_qualifying = [
        r
        for r in model_analysis
        if _meets_recommendation_threshold(games_by_event.get(r.get("event_id", ""), {}), r)
    ]

    # Match tracking_service.record_from_slate stake math so board units equal
    # the official book (multi-pick same-slate correlation haircut).
    from web.tracking_service import stake_units_from_kelly

    slate_correlation_penalty = 0.15 if len(qualifying) > 1 else 0.0
    for rec in qualifying:
        rec["stake_units"] = stake_units_from_kelly(
            rec.get("kelly_pct"),
            ev_pct=rec.get("ev_pct"),
            correlation_penalty=slate_correlation_penalty,
        )

    if errors:
        logger.warning(
            "Daily slate complete with %s error(s); %s games analyzed",
            len(errors),
            len(all_games),
        )
    else:
        logger.info(
            "Daily slate complete: %s games, %s official picks, %s model-only",
            len(all_games),
            len(qualifying),
            len(model_analysis_qualifying),
        )

    return {
        "generated_at": generated_at,
        "date_label": _toronto_today().isoformat(),
        "summary": {
            "games_analyzed": len(all_games),
            "recommended_bets": len(qualifying),
            "model_analysis_bets": len(model_analysis_qualifying),
            **official_hubacek_thresholds(),
            "leagues": list({game["league"] for game in all_games}),
            "error_count": len(errors),
            "line_shopping": _line_shopping_status(),
            "leagues_not_ready": leagues_not_ready,
        },
        "recommended_bets": qualifying[:20],
        "model_analysis_bets": model_analysis_qualifying[:20],
        "min_market_gap_pp": HUBACEK_MIN_MARKET_GAP_PP,
        "pick_system": "hubacek",
        "games": all_games,
        "errors": errors,
    }
