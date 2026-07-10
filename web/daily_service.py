"""Daily slate analysis and bet recommendations across all supported leagues."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORONTO = ZoneInfo("America/Toronto")


def _toronto_today() -> date:
    return datetime.now(TORONTO).date()

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
from web.league_readiness import is_league_ready_for_daily_slate  # noqa: E402
from web.live_data import load_live_team_data, resolve_team  # noqa: E402
from web.pick_strategy import (
    evaluate_official_picks_for_game,
    evaluate_soccer_official_picks_for_game,
    get_pick_thresholds,
    official_bet_type,
)
from web.predict_service import FACTOR_LABELS  # noqa: E402

SINGLE_MODEL_BLEND_MODES = frozenset(
    {"basketball_matrix", "mlb_runcast", "soccer_path_a"}
)


def _factor_entry(key: str, label: str, value: float) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "value": value,
        "favors": "home" if value > 0 else "away" if value < 0 else "neutral",
    }


def _sport_layer_factors(blended: dict[str, Any], league: str) -> list[dict[str, Any]]:
    """Factors from the active sport model (not legacy Algo V2)."""
    league = league.lower()
    factors: list[dict[str, Any]] = []

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
        factors.append(_factor_entry(key, label, float(algo_data[key])))
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


def predict_live_game(game: ScheduledGame) -> dict[str, Any]:
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
        consensus_spread=game.market.spread,
        home_moneyline=game.market.home_moneyline,
        away_moneyline=game.market.away_moneyline,
        draw_moneyline=game.market.draw_moneyline,
    )
    blended = apply_ensemble_ml(
        blended,
        game.league,
        consensus_spread=game.market.spread,
        home_moneyline=game.market.home_moneyline,
        away_moneyline=game.market.away_moneyline,
        draw_moneyline=game.market.draw_moneyline,
    )
    blended = ensure_hubacek_in_blend(
        blended,
        league=game.league,
        away_market=game.market.away_moneyline,
        home_market=game.market.home_moneyline,
        consensus_spread=game.market.spread,
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
            game_date=cutoff,
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

    if is_soccer_league(game.league):
        soccer_pred = blended.get("soccer_pred") or {}
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
        )
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
            consensus_spread=game.market.spread,
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

    def _enrich_pick(pick_dict: dict[str, Any]) -> dict[str, Any]:
        return _enrich_pick_with_team_abbr(pick_dict, matchup)

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
        "market": {
            "provider": game.market.provider,
            "away_moneyline": game.market.away_moneyline,
            "home_moneyline": game.market.home_moneyline,
            "draw_moneyline": game.market.draw_moneyline,
            "spread": game.market.spread,
            "away_spread_odds": game.market.away_spread_odds,
            "home_spread_odds": game.market.home_spread_odds,
            "over_under": game.market.over_under,
        },
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


def _prewarm_league_models(
    league: str,
    cutoffs: set[str],
    errors: list[dict[str, str]],
) -> None:
    for cutoff in cutoffs:
        try:
            prewarm_league_power(league, cutoff)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {"league": league, "error": f"Power prewarm failed ({cutoff}): {exc}"}
            )
        if is_basketball_league(league):
            try:
                get_basketball_pred_context(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"Basketball matrix prewarm failed ({cutoff}): {exc}",
                    }
                )
        if is_baseball_league(league):
            try:
                get_baseball_pred_context(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"Baseball model prewarm failed ({cutoff}): {exc}",
                    }
                )
        if league.lower() == "mlb":
            try:
                from web.mlb_pred_model import _cutoff_to_iso
                from web.mlb_v2.live import get_live_context

                day_iso = _cutoff_to_iso(cutoff)
                if day_iso:
                    get_live_context(day_iso)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"MLB v2 prewarm failed ({cutoff}): {exc}",
                    }
                )
        if league.lower() == "nhl":
            try:
                from web.hockey_pred_model import _cutoff_to_iso as _nhl_cutoff_to_iso
                from web.nhl_v2.live import get_live_context as _nhl_live_context

                day_iso = _nhl_cutoff_to_iso(cutoff)
                if day_iso:
                    _nhl_live_context(day_iso)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"NHL v2 prewarm failed ({cutoff}): {exc}",
                    }
                )
        if is_football_league(league):
            try:
                get_football_pred_context(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"Football model prewarm failed ({cutoff}): {exc}",
                    }
                )
        if is_soccer_league(league):
            try:
                get_soccer_pred_context(league, cutoff)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "error": f"Soccer model prewarm failed ({cutoff}): {exc}",
                    }
                )


def get_daily_slate(days_ahead: int = 0) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    all_games: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    slate_cutoff = _slate_cutoff_date()

    for league in SUPPORTED_LEAGUES:
        try:
            scheduled = fetch_scoreboard(league, days_ahead=days_ahead)
        except Exception as exc:  # noqa: BLE001
            errors.append({"league": league, "error": str(exc)})
            continue

        actionable = _actionable_games(scheduled)
        if not actionable:
            continue

        if not is_league_ready_for_daily_slate(league, slate_cutoff):
            continue

        power_cutoffs = {_today_cutoff(game) for game in actionable}
        _prewarm_league_models(league, power_cutoffs, errors)

        for game in actionable:
            try:
                all_games.append(predict_live_game(game))
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "league": league,
                        "game": game.name,
                        "error": str(exc),
                    }
                )

    recommendations = []
    for game in all_games:
        if not game.get("eligible_for_official_picks", True):
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
        if game.get("eligible_for_official_picks", True):
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

    return {
        "generated_at": generated_at,
        "date_label": _toronto_today().isoformat(),
        "summary": {
            "games_analyzed": len(all_games),
            "recommended_bets": len(qualifying),
            "model_analysis_bets": len(model_analysis_qualifying),
            **official_hubacek_thresholds(),
            "leagues": list({game["league"] for game in all_games}),
        },
        "recommended_bets": qualifying[:20],
        "model_analysis_bets": model_analysis_qualifying[:20],
        "min_market_gap_pp": HUBACEK_MIN_MARKET_GAP_PP,
        "pick_system": "hubacek",
        "games": all_games,
        "errors": errors,
    }
